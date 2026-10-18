"""Validate run settings."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tefed.cli import parser
from tefed.engine import source_secret, validate_run_config, run_experiment


class ConfigurationTests(unittest.TestCase):
    def test_nonpositive_round_override_is_rejected(self):
        for value in ['0', '-1']:
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args(['train', '--config', 'local.toml', '--output', 'run', '--rounds', value])

    def test_invalid_run_parameters_are_rejected_before_io(self):
        for key, value in [('rounds', 0), ('local_epochs', -1), ('batch_size', 1.5)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_config({'experiment': {key: value}})
        for model in [{'image_size': 63}, {'head_loss_policy': 'equal_average'}, {'depth_scale': 0},
                      {'classes': 1}, {'classes': True}, {'classes': 2.5}]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_run_config({'model': model})
        validate_run_config({})

    def test_packaged_fixture_creation_is_independent_of_working_directory(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'fixture'
            code = ('import json; from pathlib import Path; '
                    'from tefed.software_fixture import create_fixture; '
                    f'c=create_fixture(Path({str(output)!r})); '
                    'print(json.dumps({"manifest": c["experiment"]["manifest"], '
                    '"test_mode": c["experiment"]["test_mode"]}))')
            result = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                    capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            self.assertTrue(payload['test_mode'])
            manifest = Path(payload['manifest'])
            self.assertTrue(manifest.is_file())
            self.assertEqual(json.loads(manifest.read_text(encoding='utf-8'))['data_origin'], 'generated')

    def test_actual_data_source_secret_missing_short_and_sufficient(self):
        config = {'experiment': {'test_mode': False}}
        for value in [None, '', 'x' * 31, ' ' * 40]:
            environment = {} if value is None else {'TEFED_SOURCE_SECRET': value}
            with self.subTest(value_length=None if value is None else len(value)), self.assertRaisesRegex(ValueError, 'TEFED_SOURCE_SECRET'):
                source_secret(config, environ=environment)
        self.assertEqual(source_secret(config, environ={'TEFED_SOURCE_SECRET': 'x' * 32}), b'x' * 32)

    def test_fixture_secret_is_fixed_and_never_uses_an_actual_data_key(self):
        config = {'experiment': {'test_mode': True}}
        first = source_secret(config, environ={})
        second = source_secret(config, environ={'TEFED_SOURCE_SECRET': 'private' * 10})
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 32)
        with self.assertRaises(ValueError):
            source_secret({'experiment': {'test_mode': 'false'}}, environ={})

    def test_missing_secret_fails_before_output_directory_or_manifest_read(self):
        with TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
            output = Path(directory) / 'must-not-exist'
            config = {'experiment': {'manifest': str(Path(directory) / 'missing.json')}, 'model': {}}
            with self.assertRaisesRegex(ValueError, 'TEFED_SOURCE_SECRET'):
                run_experiment(config, output_dir=output)
            self.assertFalse(output.exists())
