"""Predict one registered RGB/depth/NIR observation with a frozen TE-Fed checkpoint."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .data import SampleRecord, load_sample, map_boxes
from .engine import apply_input_policy, assess_frame, load_checkpoint, write_json
from .state import validate_state


def _file(value, name):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{name} file not found: {path}')
    return path


def _sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _module_digest(module):
    """Cover parameters and persistent buffers without keeping another model copy."""
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _names(value, count):
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        value = json.loads(_file(value, 'class names').read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise ValueError('class_names must be a JSON object mapping class IDs to names')
    result = {}
    for key, name in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError('class_names keys must be integer class IDs') from None
        if isinstance(key, bool) or str(index) != str(key) or not 0 <= index < count:
            raise ValueError(f'class_names contains an invalid class ID: {key!r}')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('class_names values must be nonempty strings')
        result[index] = name.strip()
    return result


def _check_nir(path):
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.dtype != np.uint8 and not (
        np.issubdtype(array.dtype, np.floating) and np.isfinite(array).all()
        and np.all((array >= 0) & (array <= 1))
    ):
        raise ValueError('NIR must be uint8 or [0,1] floating-point .npy. Raw high-bit-depth NIR '
                         'requires the fixed preprocessing recipe used for training; '
                         'do not substitute a contrast-stretched preview image.')


def _draw(rgb, predictions, path, threshold, names):
    with Image.open(rgb) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 36)))
    for item in predictions:
        if item['score'] < threshold:
            continue
        box = item['box_original_xyxy']
        cls = int(item['class_id'])
        label = f'{names.get(cls, f"class_{cls}")} {item["score"]:.3f}'
        color = (255, 190, 40)
        draw.rectangle(box, outline=color, width=width)
        bounds = draw.textbbox((0, 0), label, font=font)
        height = bounds[3] - bounds[1] + 4
        x = min(max(0, box[0]), max(0, image.width - (bounds[2] - bounds[0]) - 4))
        y = min(max(0, box[1] - height), max(0, image.height - height))
        draw.rectangle((x, y, x + bounds[2] - bounds[0] + 4, y + height), fill=color)
        draw.text((x + 2, y + 2 - bounds[1]), label, fill=(0, 0, 0), font=font)
    image.save(path)


def run_inference(checkpoint, calibration, rgb, depth, nir, sample_id, client_id,
                  output, device='cpu', display_confidence=.25, class_names=None, round_id=0):
    """Write predictions and a display overlay; no training, state update or GT metrics.

    Input modalities must already be registered and use the checkpoint's training
    preprocessing. ``display_confidence`` filters only the overlay. All candidates
    returned by the existing evaluator's assessment path remain in predictions.json.
    """
    sample_id, client_id = str(sample_id).strip(), str(client_id).strip()
    if not sample_id or not client_id:
        raise ValueError('sample_id and client_id must be nonempty')
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError('round_id must be a nonnegative integer')
    if isinstance(display_confidence, bool) or not math.isfinite(float(display_confidence)) or not 0 <= float(display_confidence) <= 1:
        raise ValueError('display_confidence must be finite and in [0,1]')
    display_confidence = float(display_confidence)
    paths = {name: _file(value, name) for name, value in (
        ('checkpoint', checkpoint), ('calibration', calibration), ('rgb', rgb), ('depth', depth), ('nir', nir))}
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Output must be a new directory: {destination}')
    if any(destination == path or destination in path.parents for path in paths.values()):
        raise ValueError('Output directory must not contain or overwrite input files')
    _check_nir(paths['nir'])
    artifact = json.loads(paths['calibration'].read_text(encoding='utf-8'))
    if not isinstance(artifact, dict):
        raise ValueError('calibration must contain a JSON object')
    temperature = artifact.get('T_C')
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('calibration requires an explicit finite positive T_C')
    inputs = {name: {'path': str(path), 'sha256': _sha256(path)} for name, path in paths.items()}
    model, router, payload = load_checkpoint(paths['checkpoint'], device=device)
    config = deepcopy(payload['config'])
    if config.get('experiment', {}).get('method') != 'tefed':
        raise ValueError('Single-observation inference requires a checkpoint with experiment.method=tefed')
    if config.get('task', 'tomato') != 'tomato' or config.get('model', {}).get('specialists', 5) != 5:
        raise ValueError('This entry point requires the tomato TE-Fed checkpoint with five specialists')
    if config.get('model', {}).get('disable_auxiliary') or config.get('model', {}).get('detector_rgb_only'):
        raise ValueError('This entry point requires the trimodal detector, not an RGB-only control')
    states = payload.get('states', {})
    if client_id not in states:
        raise ValueError(f'Client {client_id!r} has no stored checkpoint state; choose a stored client ID')
    state = validate_state(states[client_id])
    frozen_state = deepcopy(state)
    states_before = deepcopy(states)
    config.setdefault('experiment', {})['device'] = str(device)
    names = _names(class_names, int(config['model'].get('classes', 2)))
    record = SampleRecord(sample_id, client_id, 'unlabeled', paths['rgb'], paths['depth'], paths['nir'])
    sample = load_sample(record, size=config['model'].get('image_size', 640),
                         require_modalities=('depth', 'nir'), include_labels=False)
    sample = apply_input_policy(sample, config)
    model.eval()
    router.eval()
    before = (_module_digest(model), _module_digest(router))
    with torch.no_grad():
        assessment = assess_frame(model, router, sample, state, artifact, config,
                                  round_id=round_id, experts=True)
    if before != (_module_digest(model), _module_digest(router)) or state != frozen_state or states != states_before:
        raise RuntimeError('Inference changed model/router parameters, buffers or frozen client state')
    predictions = []
    original_size = np.asarray([*sample['meta']['original_size']] * 2, dtype=float)
    for candidate in assessment['candidates']:
        item = deepcopy(candidate)
        original = map_boxes([item['box']], sample['meta'], inverse=True)[0]
        item['box_letterbox_normalized_xyxy'] = list(item['box'])
        item['box_original_normalized_xyxy'] = original.tolist()
        item['box_original_xyxy'] = (original * original_size).tolist()
        item['class_name'] = names.get(int(item['class_id']), f'class_{item["class_id"]}')
        predictions.append(item)
    for name, path in paths.items():
        if _sha256(path) != inputs[name]['sha256']:
            raise RuntimeError(f'Input file changed during inference: {name}')
    test_only = bool(config['experiment'].get('test_mode') or artifact.get('data_origin') == 'software_fixture')
    provenance = {
        'sample_id': sample_id, 'client_id': client_id, 'inputs': inputs,
        'config_source': 'checkpoint.config', 'config_overrides': {'experiment.device': str(device)},
        'config': config, 'checkpoint_round_index': payload.get('round_index'),
        'checkpoint_completed_rounds': payload.get('completed_rounds'),
        'checkpoint_model_schema_version': payload.get('model_schema_version'),
        'checkpoint_implementation_status': payload.get('implementation_status'),
        'calibration_source': 'explicit file; calibration is not embedded in this checkpoint format',
        'calibration_binding_verified': False,
        'state_source': 'checkpoint.states', 'state': frozen_state, 'state_frozen': True,
        'model_router_state_unchanged': True, 'round_id': round_id,
        'round_id_protocol': '0 matches the current evaluate path; it seeds the fixed reference augmentation',
        'sample_split': 'unlabeled', 'ground_truth_used': False, 'ap_evaluated': False,
        'letterbox': sample['meta'], 'display_confidence': display_confidence,
        'display_filter_applies_to': 'rgb_with_detections.png only; all assessment candidates are exported',
        'class_names': names, 'software_test_only': test_only,
        'evidence_status': 'software interface test only' if test_only else 'inference with the supplied checkpoint',
        'paper_results_reproduced': False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'predictions.json', {
        'sample_id': sample_id, 'client_id': client_id, 'letterbox': sample['meta'],
        'box_coordinate_system': 'box is letterbox-normalized xyxy; explicit original-image coordinates accompany it',
        'predictions': predictions,
    })
    _draw(paths['rgb'], predictions, destination / 'rgb_with_detections.png', display_confidence, names)
    write_json(destination / 'provenance.json', provenance)
    return {'output_dir': str(destination), 'predictions': len(predictions),
            'displayed': sum(float(item['score']) >= display_confidence for item in predictions),
            'software_test_only': test_only}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'calibration', 'rgb', 'depth', 'nir', 'sample-id', 'client-id', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--display-confidence', type=float, default=.25)
    parser.add_argument('--class-names', help='JSON object file mapping class IDs to names; otherwise class_N')
    parser.add_argument('--round-id', type=int, default=0, help='Reference-augmentation seed round; 0 matches evaluate')
    args = parser.parse_args(argv)
    try:
        result = run_inference(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
