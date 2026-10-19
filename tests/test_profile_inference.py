"""Check measurement device and output scope without running a detector."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import profile_inference


class InferenceProfileTests(unittest.TestCase):
    def _record(self, directory, cuda):
        root = Path(directory)
        calibration = root / 'calibration.json'
        calibration.write_text('{}', encoding='utf-8')
        output = root / 'latency.csv'
        device = 'cuda:1' if cuda else 'cpu'
        parameter = SimpleNamespace(device=device, is_cuda=cuda)
        model = Mock()
        model.parameters.side_effect = lambda: iter([parameter])
        model.capacity.side_effect = lambda active=None: {
            'active_heads': list(active or (0, 1, 2)), 'active_prediction_parameters': 123}
        router = Mock()
        router.parameters.side_effect = lambda: iter([])
        cfg = {'experiment': {'device': device, 'manifest': 'manifest.json', 'calibration': str(calibration)}}
        payload = {'config': {'experiment': {'method': 'tefed'}}, 'states': {'client0': {}}}
        record = SimpleNamespace(id='sample0')
        manifest = Mock()
        manifest.by_split.return_value = [record]
        arguments = ['profile_inference.py', '--config', 'config.toml', '--checkpoint', 'final.pt',
                     '--node', 'node0', '--client', 'client0', '--requests', '200', '--warmup', '1',
                     '--output', str(output)]
        assessment = {'candidates': [{'route': [0, 2, 4]}, {'route': [0, 1, 2]}]}
        with patch('sys.argv', arguments), \
                patch('tefed.engine.load_config', return_value=cfg), \
                patch('tefed.engine.load_checkpoint', return_value=(model, router, payload)), \
                patch('tefed.data.load_manifest', return_value=manifest), \
                patch('tefed.engine.prepared_sample', return_value={}), \
                patch('tefed.engine.assess_frame', return_value=assessment) as assess, \
                patch('torch.cuda.synchronize') as synchronize:
            profile_inference.main()
        with output.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        scope = json.loads(output.with_suffix('.scope.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 200)
        self.assertEqual(assess.call_count, 201)
        self.assertEqual(scope['observed_frame_head_sets'], {'0,1,2,4': 200})
        self.assertTrue(all(row['active_heads'] == '0,1,2,4' for row in rows))
        self.assertEqual(scope['device'], device)
        return synchronize, device

    def test_cuda_measurements_synchronize_the_model_device(self):
        with TemporaryDirectory() as directory:
            synchronize, device = self._record(directory, cuda=True)
        self.assertEqual(synchronize.call_count, 402)
        self.assertTrue(all(call.args == (device,) for call in synchronize.call_args_list))

    def test_cpu_measurements_do_not_touch_cuda(self):
        with TemporaryDirectory() as directory:
            synchronize, _ = self._record(directory, cuda=False)
        synchronize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
