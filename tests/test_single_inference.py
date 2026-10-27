"""Single-observation boundaries and one native software-fixture integration test."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tefed import inference
from tefed.data import letterbox_parameters, load_manifest, map_boxes


STATE = {'q': .6, 'p': .7, 'h': [.2] * 5}


class SingleInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='tefed-single-inference-',
                                             dir=os.environ.get('TEFED_TEST_ARTIFACT_ROOT'))
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / 'checkpoint.pt'
        self.checkpoint.write_bytes(b'mocked-checkpoint-not-model-evidence')
        self.calibration = self.root / 'calibration.json'
        self.calibration.write_text(json.dumps({'T_C': 1., 'data_origin': 'software_fixture'}), encoding='utf-8')
        self.rgb, self.depth, self.nir = [self.root / f'{name}.png' for name in ('rgb', 'depth', 'nir')]
        Image.fromarray(np.full((20, 40, 3), 80, np.uint8)).save(self.rgb)
        Image.fromarray(np.full((20, 40), 700, np.uint16)).save(self.depth)
        Image.fromarray(np.full((20, 40), 120, np.uint8)).save(self.nir)
        self.model, self.router = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        self.payload = {'config': {'task': 'tomato', 'experiment': {'method': 'tefed', 'test_mode': True},
                                  'model': {'image_size': 64, 'depth_scale': 65535., 'classes': 2, 'specialists': 5}},
                        'states': {'client0': deepcopy(STATE)}, 'round_index': 1, 'completed_rounds': 2,
                        'model_schema_version': 2, 'implementation_status': 'software-fixture'}
        transform = letterbox_parameters((40, 20), 64)
        boxes = map_boxes([[.1, .2, .8, .9], [.45, .3, .55, .6]], transform)
        self.candidates = [dict(box=box.tolist(), class_id=cls, score=score,
                                route=[0, 1, 2], weights=[.2, .5, .3],
                                disposition=disposition, R=.7)
                           for box, cls, score, disposition in zip(boxes, [0, 1], [.95, .1], ['qualified', 'suppressed'])]

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, **overrides):
        arguments = dict(checkpoint=self.checkpoint, calibration=self.calibration, rgb=self.rgb,
                         depth=self.depth, nir=self.nir, sample_id='observation-1', client_id='client0',
                         output=self.root / 'prediction', display_confidence=.5)
        arguments.update(overrides)
        return inference.run_inference(**arguments)

    def loaded(self):
        return patch.object(inference, 'load_checkpoint', return_value=(self.model, self.router, self.payload))

    def test_unlabeled_inference_exports_all_candidates_and_inverse_coordinates(self):
        original_config = deepcopy(self.payload['config'])

        def assess(model, router, sample, state, artifact, config, **kwargs):
            self.assertFalse(model.training)
            self.assertFalse(router.training)
            self.assertEqual(sample['targets'], [])
            self.assertTrue(sample['ignore_labels'])
            self.assertEqual(state, STATE)
            self.assertEqual(kwargs, {'round_id': 0, 'experts': True})
            self.assertEqual(config['experiment']['device'], 'cpu')
            return {'candidates': deepcopy(self.candidates)}

        with self.loaded(), patch.object(inference, 'assess_frame', side_effect=assess):
            result = self.call(class_names={'1': 'provided_name'})
        self.assertEqual(result['predictions'], 2)
        self.assertEqual(result['displayed'], 1)
        output = Path(result['output_dir'])
        saved = json.loads((output / 'predictions.json').read_text())['predictions']
        self.assertEqual(saved[1]['disposition'], 'suppressed')
        self.assertEqual(saved[1]['route'], [0, 1, 2])
        self.assertEqual(saved[0]['class_name'], 'class_0')
        self.assertEqual(saved[1]['class_name'], 'provided_name')
        np.testing.assert_allclose(saved[0]['box_original_xyxy'], [4., 4., 32., 18.], atol=1e-5)
        np.testing.assert_allclose(saved[0]['box'], self.candidates[0]['box'])
        np.testing.assert_allclose(map_boxes([saved[0]['box_original_normalized_xyxy']],
                                           letterbox_parameters((40, 20), 64))[0], saved[0]['box'], atol=1e-6)
        provenance = json.loads((output / 'provenance.json').read_text())
        self.assertTrue(provenance['software_test_only'])
        self.assertTrue(provenance['state_frozen'])
        self.assertFalse(provenance['ground_truth_used'])
        self.assertFalse(provenance['ap_evaluated'])
        self.assertEqual(provenance['inputs']['nir']['sha256'], hashlib.sha256(self.nir.read_bytes()).hexdigest())
        self.assertEqual(provenance['state_source'], 'checkpoint.states')
        self.assertEqual(self.payload['config'], original_config)
        with Image.open(output / 'rgb_with_detections.png') as image:
            self.assertEqual(image.size, (40, 20))

    def test_raw_sixteen_bit_nir_rejected_before_loading_checkpoint(self):
        Image.fromarray(np.full((20, 40), 2000, np.uint16)).save(self.nir)
        with self.loaded() as load, self.assertRaisesRegex(ValueError, 'fixed preprocessing recipe used for training'):
            self.call()
        load.assert_not_called()
        self.assertFalse((self.root / 'prediction').exists())

    def test_unknown_client_fails_without_prior_fallback(self):
        with self.loaded(), self.assertRaisesRegex(ValueError, 'no stored checkpoint state'):
            self.call(client_id='unknown')
        self.assertFalse((self.root / 'prediction').exists())

    def test_missing_checkpoint_does_not_create_output(self):
        with self.loaded() as load, self.assertRaisesRegex(FileNotFoundError, 'checkpoint file not found'):
            self.call(checkpoint=self.root / 'missing.pt')
        load.assert_not_called()
        self.assertFalse((self.root / 'prediction').exists())

    def test_existing_output_is_not_overwritten(self):
        output = self.root / 'prediction'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('unchanged')
        with self.loaded() as load, self.assertRaises(FileExistsError):
            self.call()
        load.assert_not_called()
        self.assertEqual(marker.read_text(), 'unchanged')

    def test_general_or_rgb_only_checkpoint_is_not_a_trimodal_demo(self):
        self.payload['config']['experiment']['method'] = 'server_moe'
        with self.loaded(), self.assertRaisesRegex(ValueError, 'method=tefed'):
            self.call()
        self.payload['config']['experiment']['method'] = 'tefed'
        self.payload['config']['model']['detector_rgb_only'] = True
        with self.loaded(), self.assertRaisesRegex(ValueError, 'RGB-only control'):
            self.call()

    def test_mutating_parameters_or_state_aborts_before_output(self):
        def mutate_model(*args, **kwargs):
            self.model.weight.add_(1)
            return {'candidates': []}

        with self.loaded(), patch.object(inference, 'assess_frame', side_effect=mutate_model), self.assertRaisesRegex(RuntimeError, 'changed model/router'):
            self.call()
        self.assertFalse((self.root / 'prediction').exists())

        def mutate_state(model, router, sample, state, *args, **kwargs):
            state['q'] = .9
            return {'candidates': []}

        with self.loaded(), patch.object(inference, 'assess_frame', side_effect=mutate_state), self.assertRaisesRegex(RuntimeError, 'frozen client state'):
            self.call()
        self.assertEqual(self.payload['states']['client0'], STATE)

    def test_explicit_round_and_class_name_file_are_recorded(self):
        names = self.root / 'names.json'
        names.write_text(json.dumps({'0': 'supplied_class'}), encoding='utf-8')
        with self.loaded(), patch.object(inference, 'assess_frame', return_value={'candidates': []}) as assess:
            self.call(round_id=7, class_names=names)
        self.assertEqual(assess.call_args.kwargs['round_id'], 7)
        saved = json.loads((self.root / 'prediction' / 'provenance.json').read_text())
        self.assertEqual(saved['round_id'], 7)
        self.assertEqual(saved['class_names'], {'0': 'supplied_class'})

    def test_native_checkpoint_uses_only_generated_software_fixture(self):
        from tefed.engine import save_checkpoint
        from tefed.model import RoutedYOLO, Router
        from tefed.software_fixture import create_fixture

        torch.set_num_threads(2)
        torch.manual_seed(37)
        config = create_fixture(self.root / 'native-software-fixture')
        config['evaluation']['prediction_confidence'] = 0.
        config['evaluation']['max_detections'] = 2
        model, router = RoutedYOLO(config), Router(.5)
        checkpoint = self.root / 'native-software-fixture.pt'
        # This initialized fixture has completed no training rounds.
        save_checkpoint(checkpoint, model, router, config, {'client0': deepcopy(STATE)}, [], -1)
        record = load_manifest(config['experiment']['manifest']).by_split('unlabeled', 'client0')[0]
        result = self.call(checkpoint=checkpoint, calibration=config['experiment']['calibration'],
                           rgb=record.rgb, depth=record.depth, nir=record.nir,
                           sample_id=record.id, output=self.root / 'native-test-output')
        provenance = json.loads((Path(result['output_dir']) / 'provenance.json').read_text())
        self.assertTrue(result['software_test_only'])
        self.assertGreater(result['predictions'], 0)
        self.assertTrue(provenance['model_router_state_unchanged'])
        self.assertEqual(provenance['config']['model']['image_size'], 64)
        self.assertEqual(provenance['checkpoint_completed_rounds'], 0)
        self.assertEqual(provenance['evidence_status'], 'software interface test only')
        self.assertFalse(provenance['paper_results_reproduced'])


if __name__ == '__main__':
    unittest.main()
