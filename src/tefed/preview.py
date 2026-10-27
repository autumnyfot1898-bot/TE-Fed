"""Preview aligned input images without loading a detector or changing raw data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_OUTPUT_NAMES = {
    'rgb': 'rgb.png',
    'depth_display': 'depth_display.png',
    'nir_display': 'nir_display.png',
    'combined': 'rgb_depth_nir.png',
    'metadata': 'preview.json',
}


def _local_file(example_dir, relative_path, name):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f'{name} must specify a relative file path')
    relative = Path(relative_path)
    path = (example_dir / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(example_dir):
        raise ValueError(f'{name} must stay inside the example directory')
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name}: {path}')
    return path


def _read_input(example_dir, spec, name):
    if not isinstance(spec, dict):
        raise ValueError(f'Manifest files.{name} must be an object')
    path = _local_file(example_dir, spec.get('path'), name)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = spec.get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError(f'Manifest files.{name} needs a SHA-256 hash')
    if digest != expected.lower():
        raise ValueError(f'SHA-256 mismatch for {name}: {path}')
    if payload[:8] != b'\x89PNG\r\n\x1a\n' or len(payload) < 26:
        raise ValueError(f'{name} must be a PNG image')
    with Image.open(path) as image:
        image.load()
        array = np.array(image)
        mode = image.mode
    # Some Pillow versions expose 16-bit PNG samples as int32. The PNG header
    # records the original bit depth, independently of the in-memory dtype.
    bit_depth = payload[24]
    if name == 'rgb':
        if bit_depth != 8 or array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('RGB must be an 8-bit, three-channel PNG')
    else:
        if bit_depth != 16 or array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f'{name} must be a single-channel, 16-bit PNG')
        if np.any(array < 0) or np.any(array > 65535):
            raise ValueError(f'{name} contains values outside uint16 range')
        array = array.astype(np.uint16, copy=False)
    if spec.get('dtype') != str(array.dtype):
        raise ValueError(f'Manifest dtype does not match {name}: {array.dtype}')
    if spec.get('shape') != list(array.shape):
        raise ValueError(f'Manifest shape does not match {name}: {list(array.shape)}')
    details = {
        **spec, 'resolved_path': str(path), 'verified_sha256': digest,
        'png_bit_depth': bit_depth, 'png_mode': mode,
    }
    return array, payload, details


def _palette(example_dir):
    path = _local_file(example_dir, 'display_palette.json', 'display palette')
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    colors = raw.get('viridis') if isinstance(raw, dict) else raw
    try:
        values = np.asarray(colors, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('display_palette.json needs 256 RGB colors for viridis') from error
    if (values.shape != (256, 3) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 255)
            or not np.equal(values, np.rint(values)).all()):
        raise ValueError('display_palette.json needs 256 integer RGB colors in [0,255]')
    return values.astype(np.uint8), {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _display(array, spec, name, palette):
    if not isinstance(spec, dict):
        raise ValueError(f'Manifest display_mappings.{name} must be an object')
    try:
        low, high = float(spec['vmin']), float(spec['vmax'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{name} needs fixed vmin and vmax display limits') from error
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low:
        raise ValueError(f'{name} display limits must satisfy 0 < vmin < vmax')
    if spec.get('zero_display') != 'black':
        raise ValueError(f'{name} must explicitly map zero values to black')
    scaled = np.clip((array.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    indices = np.rint(scaled * 255).astype(np.uint8)
    colormap = spec.get('colormap')
    if colormap == 'viridis':
        if palette is None:
            raise ValueError('Viridis display requires display_palette.json')
        display = palette[indices].copy()
    elif colormap == 'gray':
        display = np.repeat(indices[:, :, None], 3, axis=2)
    else:
        raise ValueError(f'Unsupported {name} display colormap: {colormap!r}')
    display[array == 0] = 0
    return display


def preview_example(example_dir, output_dir):
    """Write an input preview using the example's fixed display limits.

    The output directory must be new. RGB is copied byte for byte; Depth and
    NIR are read as uint16 and never rewritten. Display colors are unrelated to
    model input preprocessing. No checkpoint, training code or torch is loaded.
    The returned summary has absolute paths; preview.json uses portable names.
    """
    example = Path(example_dir).expanduser().resolve()
    if not example.is_dir():
        raise FileNotFoundError(f'Example directory does not exist: {example}')
    output = Path(output_dir).expanduser().resolve()
    if output == example:
        raise ValueError('Output directory must differ from the source example directory')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory; refusing to overwrite: {output}')
    manifest_path = _local_file(example, 'manifest.json', 'manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise ValueError('Manifest must contain a files object')
    arrays, sources, rgb_bytes = {}, {}, None
    for name in ('rgb', 'depth', 'nir'):
        array, payload, details = _read_input(example, manifest['files'].get(name), name)
        arrays[name], sources[name] = array, details
        if name == 'rgb':
            rgb_bytes = payload
    height, width = arrays['rgb'].shape[:2]
    if any(array.shape[:2] != (height, width) for array in arrays.values()):
        raise ValueError('RGB, Depth and NIR dimensions must match')
    if manifest.get('image_size') != [width, height]:
        raise ValueError('Manifest image_size must match the input width and height')
    mappings = manifest.get('display_mappings')
    if not isinstance(mappings, dict):
        raise ValueError('Manifest must contain fixed display_mappings')
    needs_palette = any(isinstance(mappings.get(name), dict)
                        and mappings[name].get('colormap') == 'viridis'
                        for name in ('depth', 'nir'))
    palette, palette_source = _palette(example) if needs_palette else (None, None)
    displays = {name: _display(arrays[name], mappings.get(name), name, palette)
                for name in ('depth', 'nir')}

    # Keep all panels at the original resolution; labels sit below the images.
    label_height, gutter = 96, 16
    labels = ['(a) RGB', '(b) Depth', '(c) NIR']
    combined_width = 3 * width + 2 * gutter
    combined = Image.new('RGB', (combined_width, height + label_height), 'white')
    draw = ImageDraw.Draw(combined)
    font_size = min(64, max(10, width // 5))
    font = ImageFont.load_default(size=font_size)
    for index, (label, array) in enumerate(zip(labels, (arrays['rgb'], displays['depth'], displays['nir']))):
        panel_left = index * (width + gutter)
        combined.paste(Image.fromarray(array), (panel_left, 0))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text((panel_left + (width - right + left) / 2,
                   height + (label_height - bottom + top) / 2 - top), label, font=font, fill='black')
    paths = {name: str(output / filename) for name, filename in _OUTPUT_NAMES.items()}
    summary = {
        'display_only': True,
        'inference_executed': False,
        'purpose': 'Input visualization; display mappings are not model preprocessing.',
        'example_dir': str(example),
        'source_manifest': str(manifest_path),
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_metadata': manifest,
        'source_files': sources,
        'raw_inputs_modified': False,
        'rgb_copy': 'byte-for-byte copy of the source PNG',
        'image_size': [width, height],
        'combined_image_size': [combined_width, height + label_height],
        'label_band_height': label_height,
        'label_position': 'below images',
        'panel_labels': labels,
        'label_font_size': font_size,
        'gutter_width': gutter,
        'display_mappings': mappings,
        'mapping_applied': 'clip((value-vmin)/(vmax-vmin),0,1); round to 0..255; set zero pixels to black',
        'display_palette': palette_source,
        'outputs': paths,
    }
    # Validation and rendering finish before creating any output files.
    output.mkdir(parents=True, exist_ok=False)
    with Path(paths['rgb']).open('xb') as stream:
        stream.write(rgb_bytes)
    for name in ('depth', 'nir'):
        with Path(paths[f'{name}_display']).open('xb') as stream:
            Image.fromarray(displays[name]).save(stream, format='PNG')
    with Path(paths['combined']).open('xb') as stream:
        combined.save(stream, format='PNG')
    # Keep the saved record portable while returning absolute paths to callers.
    portable = deepcopy(summary)
    portable['example_dir'] = 'Directory supplied with --example; source paths are relative to that directory.'
    portable['source_manifest'] = 'manifest.json'
    for spec in portable['source_files'].values():
        spec.pop('resolved_path', None)
    if portable['display_palette'] is not None:
        portable['display_palette']['path'] = 'display_palette.json'
    portable['outputs'] = dict(_OUTPUT_NAMES)
    with Path(paths['metadata']).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(portable, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', required=True, help='Directory containing the example manifest and input PNGs')
    parser.add_argument('--output', required=True, help='New directory for display images and preview.json')
    args = parser.parse_args(argv)
    try:
        summary = preview_example(args.example, args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f'Preview failed: {error}\n')
    print(json.dumps({
        'display_only': summary['display_only'],
        'inference_executed': summary['inference_executed'],
        'outputs': summary['outputs'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
