"""Check display-only previews and preservation of source sensor images."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from tefed.preview import preview_example


class TrimodalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.example = self.root / 'example'
        (self.example / 'inputs').mkdir(parents=True)
        self.output = self.root / 'preview'
        self.width, self.height = 96, 24
        self.rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.rgb[:] = [19, 71, 143]
        self.rgb[:, self.width // 2:] = [211, 89, 31]
        self.depth = np.tile(np.array([0, 50, 100, 550, 1000, 65535], dtype=np.uint16),
                             (self.height, self.width // 6))
        self.nir = np.tile(np.array([0, 5, 10, 60, 110, 60000], dtype=np.uint16),
                           (self.height, self.width // 6))
        self.palette = [[value, 255 - value, 17] for value in range(256)]
        self.manifest = {
            'status': 'TEST_FIXTURE_INPUT_MODALITIES',
            'image_size': [self.width, self.height],
            'inference_executed': False,
            'display_mappings': {
                'depth': {'vmin': 100, 'vmax': 1000, 'colormap': 'viridis',
                          'zero_display': 'black', 'raw_dtype': 'uint16'},
                'nir': {'vmin': 10, 'vmax': 110, 'colormap': 'gray',
                        'zero_display': 'black', 'raw_dtype': 'uint16'},
            },
            'files': {},
        }
        for name, array in [('rgb', self.rgb), ('depth', self.depth), ('nir', self.nir)]:
            path = self.example / 'inputs' / f'{name}.png'
            Image.fromarray(array).save(path)
            self.manifest['files'][name] = {
                'path': f'inputs/{name}.png',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dtype': str(array.dtype),
                'shape': list(array.shape),
            }
        (self.example / 'display_palette.json').write_text(
            json.dumps({'viridis': self.palette}), encoding='utf-8')
        self.write_manifest()

    def write_manifest(self):
        (self.example / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_preview_preserves_sources_and_reports_display_only_outputs(self):
        before = {path: path.read_bytes() for path in self.example.rglob('*') if path.is_file()}

        summary = preview_example(self.example, self.output)

        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual((self.output / 'rgb.png').read_bytes(),
                         (self.example / 'inputs/rgb.png').read_bytes())
        for name, expected in [('depth', self.depth), ('nir', self.nir)]:
            source = self.example / 'inputs' / f'{name}.png'
            self.assertEqual(source.read_bytes()[24], 16, 'Original PNG must retain 16-bit samples')
            with Image.open(source) as image:
                np.testing.assert_array_equal(np.array(image), expected)
        self.assertEqual(set(path.name for path in self.output.iterdir()), {
            'rgb.png', 'depth_display.png', 'nir_display.png', 'rgb_depth_nir.png', 'preview.json'})
        self.assertTrue(summary['display_only'])
        self.assertFalse(summary['inference_executed'])
        self.assertFalse(summary['raw_inputs_modified'])
        self.assertEqual(summary['source_metadata'], self.manifest)
        self.assertEqual(summary['display_mappings'], self.manifest['display_mappings'])
        self.assertIn('not model preprocessing', summary['purpose'])
        saved = json.loads((self.output / 'preview.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['source_metadata'], self.manifest)
        self.assertEqual(saved['source_manifest'], 'manifest.json')
        self.assertEqual(saved['display_palette']['path'], 'display_palette.json')
        self.assertIn('--example', saved['example_dir'])
        self.assertTrue(all('resolved_path' not in spec for spec in saved['source_files'].values()))
        self.assertEqual(saved['outputs'], {name: Path(path).name for name, path in summary['outputs'].items()})
        self.assertNotIn(str(self.example), json.dumps(saved))
        self.assertNotIn(str(self.output), json.dumps(saved))
        self.assertTrue(all(Path(path).is_absolute() and Path(path).is_file()
                            for path in summary['outputs'].values()))

        with Image.open(self.output / 'depth_display.png') as image:
            depth_display = np.array(image)
        with Image.open(self.output / 'nir_display.png') as image:
            nir_display = np.array(image)
        # These endpoints distinguish fixed supplied limits from per-image scaling.
        np.testing.assert_array_equal(depth_display[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(depth_display[0, 1], self.palette[0])
        np.testing.assert_array_equal(depth_display[0, 4], self.palette[-1])
        np.testing.assert_array_equal(depth_display[0, 5], self.palette[-1])
        np.testing.assert_array_equal(nir_display[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(nir_display[0, 2], [0, 0, 0])
        np.testing.assert_array_equal(nir_display[0, 4], [255, 255, 255])
        np.testing.assert_array_equal(nir_display[0, 5], [255, 255, 255])

        with Image.open(self.output / 'rgb_depth_nir.png') as image:
            combined = np.array(image)
        gutter = summary['gutter_width']
        self.assertEqual(gutter, 16)
        self.assertEqual(combined.shape[1], 3 * self.width + 2 * gutter)
        band_height = combined.shape[0] - self.height
        self.assertEqual(band_height, 96)
        self.assertEqual(summary['label_position'], 'below images')
        self.assertEqual(summary['panel_labels'], ['(a) RGB', '(b) Depth', '(c) NIR'])
        for index, expected in enumerate([self.rgb, depth_display, nir_display]):
            panel_left = index * (self.width + gutter)
            panel = combined[:self.height, panel_left:panel_left + self.width]
            np.testing.assert_array_equal(panel, expected)
            footer = combined[self.height:, panel_left:panel_left + self.width]
            self.assertGreater(len(np.unique(footer.reshape(-1, 3), axis=0)), 1,
                               'Each panel needs a visible label below the image')
            if index < 2:
                white_gap = combined[:, panel_left + self.width:panel_left + self.width + gutter]
                self.assertTrue(np.all(white_gap == 255))

    def test_missing_input_creates_no_output(self):
        (self.example / 'inputs/depth.png').unlink()
        with self.assertRaises(FileNotFoundError):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_hash_creates_no_output(self):
        self.manifest['files']['nir']['sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_different_image_dimensions_create_no_output(self):
        path = self.example / 'inputs/nir.png'
        mismatched = self.nir[:, :-1]
        Image.fromarray(mismatched).save(path)
        self.manifest['files']['nir'].update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(mismatched.shape))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'dimensions must match'):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_source_directory_cannot_be_the_destination(self):
        before = {path: path.read_bytes() for path in self.example.rglob('*') if path.is_file()}
        with self.assertRaises(ValueError):
            preview_example(self.example, self.example)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertFalse((self.example / 'preview.json').exists())

    def test_existing_destination_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'rgb.png'
        sentinel.write_bytes(b'keep existing output')
        with self.assertRaises(FileExistsError):
            preview_example(self.example, self.output)
        self.assertEqual(sentinel.read_bytes(), b'keep existing output')
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_invalid_display_limits_create_no_output(self):
        self.manifest['display_mappings']['depth']['vmax'] = 100
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'display limits'):
            preview_example(self.example, self.output)
        self.assertFalse(self.output.exists())

    def test_import_does_not_load_torch(self):
        source_dir = Path(__file__).resolve().parents[1] / 'src'
        environment = dict(os.environ, PYTHONPATH=str(source_dir), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(
            [sys.executable, '-B', '-c',
             'import sys; import tefed.preview; assert "torch" not in sys.modules'],
            cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
