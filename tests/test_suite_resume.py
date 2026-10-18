"""Batch recovery, input identity and transactional metric publication."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from scripts import run_suite as suite
from tefed import engine
from tefed.resume import atomic_snapshot, file_journal, run_binding, run_lock
from tefed.software_fixture import create_fixture


class SuiteResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, root):
        config = create_fixture(root / 'fixture')
        config['experiment']['seeds'] = [0]
        return config

    def checkpoint(self, config, output, *, completed=True, next_round=None):
        """Write a real round snapshot without running detector training."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        rounds = config['experiment']['rounds']
        next_round = rounds if completed else 1 if next_round is None else next_round
        history = [{'round_index': index} for index in range(next_round)]
        engine.write_json(output / 'configuration.json', config)
        engine.write_json(output / 'calibration.json', {})
        engine.write_json(output / 'environment.json', {})
        (output / 'rounds.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        metrics = {'full': {'mAP50': .25, 'AP75': .125, 'mAP50_95': .1}}
        if completed:
            (output / 'final.pt').write_bytes(b'completed-detector')
            engine.write_json(output / 'metrics.json', metrics)
            engine.write_json(output / 'final.pool.json', [])
            engine.write_json(output / 'source_states.json', {})
        data = engine.load_manifest(config['experiment']['manifest'])
        binding = run_binding(config, data, labels_only=False, secret=engine.source_secret(config))
        atomic_snapshot(output / 'resume.pt', {'resume_schema_version': 1,
            'output_dir': str(output.resolve()), 'binding': binding,
            'phase': 'completed' if completed else 'training', 'next_round': next_round,
            'history': history, 'files': file_journal(output, completed=completed), 'metrics': metrics})
        return {'metrics': metrics}

    def runner(self, config, *, seed, output_dir, resume=None):
        return self.checkpoint(config, output_dir)

    def test_changed_plan_or_input_is_rejected_before_any_output_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            suite.run_suite(config, 'components', output)
            original = (output / 'plan.json').read_bytes()
            variants = []
            changed = deepcopy(config); changed['loss']['gamma'] = .123; variants.append(changed)
            changed = deepcopy(config); changed['experiment']['seeds'] = [0, 1]; variants.append(changed)
            for changed in variants:
                with self.assertRaisesRegex(ValueError, 'Suite plan'):
                    suite.run_suite(changed, 'components', output, execute=True, resume=True)
                self.assertEqual((output / 'plan.json').read_bytes(), original)
                self.assertFalse((output / 'execution.json').exists())
            image = engine.load_manifest(config['experiment']['manifest']).records[0].rgb
            image.write_bytes(image.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'input binding changed'):
                suite.run_suite(config, 'components', output, resume=True)
            self.assertEqual((output / 'plan.json').read_bytes(), original)

    def test_existing_output_requires_resume_and_original_plan_is_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            suite.run_suite(config, 'comparisons', output)
            original = (output / 'plan.json').read_bytes()
            with self.assertRaisesRegex(FileExistsError, '--resume'):
                suite.run_suite(config, 'comparisons', output, execute=True)
            self.assertEqual((output / 'plan.json').read_bytes(), original)

    def test_completed_jobs_are_skipped_and_result_tampering_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            with patch.object(suite, 'plans', return_value=[('server', {'experiment': {'method': 'server_moe'}})]):
                suite.run_suite(config, 'comparisons', output, execute=True, runner=self.runner)
                forbidden = Mock(side_effect=AssertionError('completed training was repeated'))
                suite.run_suite(config, 'comparisons', output, execute=True, resume=True, runner=forbidden)
                forbidden.assert_not_called()
                state_before = (output / 'execution.json').read_bytes()
                (output / 'server/seed_0/final.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'server seed 0 round 2.*final.pt'):
                    suite.run_suite(config, 'comparisons', output, execute=True, resume=True)
                self.assertEqual((output / 'execution.json').read_bytes(), state_before)

    def test_unfinished_job_resumes_and_later_jobs_start_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'; calls = []
            variants = [('one', {'experiment': {'method': 'server_moe'}}),
                        ('two', {'experiment': {'method': 'server_moe'}})]
            def interrupt(configuration, *, seed, output_dir, resume=None):
                calls.append((Path(output_dir).parent.name, resume))
                self.checkpoint(configuration, output_dir, completed=False)
                raise RuntimeError('power loss')
            def recover(configuration, *, seed, output_dir, resume=None):
                calls.append((Path(output_dir).parent.name, resume))
                return self.checkpoint(configuration, output_dir)
            with patch.object(suite, 'plans', return_value=variants):
                with self.assertRaisesRegex(RuntimeError, 'one seed 0 round 0.*power loss'):
                    suite.run_suite(config, 'comparisons', output, execute=True, runner=interrupt)
                first_history = json.loads((output / 'execution.json').read_text())['history']
                suite.run_suite(config, 'comparisons', output, execute=True, resume=True, runner=recover)
            self.assertEqual(calls, [('one', None), ('one', str(output / 'one/seed_0/resume.pt')), ('two', None)])
            state = json.loads((output / 'execution.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['history'][:len(first_history)], first_history)
            self.assertEqual([attempt['status'] for attempt in state['jobs'][0]['attempts']], ['failed', 'completed'])
            self.assertEqual(state['jobs'][0]['attempts'][1]['next_round'], 1)

    def test_factorial_publication_recovers_without_duplicate_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            original = suite._publish_factorial
            with patch.object(suite, '_publish_factorial', side_effect=OSError('interrupted metric publication')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted metric publication'):
                    suite.run_suite(config, 'factorial', output, execute=True, runner=self.runner)
            runner = Mock(side_effect=self.runner)
            with patch.object(suite, '_publish_factorial', wraps=original):
                suite.run_suite(config, 'factorial', output, execute=True, resume=True, runner=runner)
            self.assertEqual(runner.call_count, 7)
            files = {path.name: path.read_bytes() for path in output.glob('factorial_*.jsonl')}
            self.assertEqual(len(files), 4)
            for content in files.values():
                rows = [json.loads(row) for row in content.splitlines()]
                self.assertEqual(len(rows), 8)
                self.assertEqual(len({(row['seed'], tuple(sorted(row['factors'].items()))) for row in rows}), 8)
            suite.run_suite(config, 'factorial', output, execute=True, resume=True, runner=runner)
            self.assertEqual(runner.call_count, 7)
            self.assertEqual(files, {path.name: path.read_bytes() for path in output.glob('factorial_*.jsonl')})

    def test_global_output_lock_prevents_plan_writes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            with run_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    suite.run_suite(config, 'components', output)
            self.assertFalse(output.exists())

    def test_changed_factorial_operations_are_rejected_without_rewriting_plan(self):
        from tefed.experiments import FACTORIAL_DEFINITIONS
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            suite.run_suite(config, 'factorial', output)
            original = (output / 'plan.json').read_bytes()
            with patch.dict(FACTORIAL_DEFINITIONS['A'], {'off': 'changed operation'}):
                with self.assertRaisesRegex(ValueError, 'Suite plan'):
                    suite.run_suite(config, 'factorial', output, resume=True)
            self.assertEqual((output / 'plan.json').read_bytes(), original)
            self.assertFalse((output / 'execution.json').exists())

    def test_replay_plan_binds_the_pool_and_exposure_reference(self):
        from tests.test_experiment_protocols import setup_replay
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = setup_replay(root, native=True)
            output = root / 'suite'
            suite.run_suite(config, 'replay', output)
            original = (output / 'plan.json').read_bytes()
            protocol = Path(config['replay']['exposure_protocol'])
            protocol.write_bytes(protocol.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Suite plan'):
                suite.run_suite(config, 'replay', output, execute=True, resume=True)
            self.assertEqual((output / 'plan.json').read_bytes(), original)
            self.assertFalse((output / 'execution.json').exists())

    def test_budget_jobs_bind_active_renderer_inputs_from_zero_budget_base(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            scene = root / 'scene.blend'; scene.write_bytes(b'original scene')
            script = root / 'custom_renderer.py'; script.write_bytes(b'original script')
            config['renderer'] = {'enabled': True, 'blender': sys.executable,
                                  'scene': str(scene), 'script_path': str(script)}
            config['decision']['render_budget'] = 0
            suite.run_suite(config, 'budgets', output, execute=True, runner=self.runner)
            plan = json.loads((output / 'plan.json').read_text())
            for path in (scene, script, Path(sys.executable).resolve()):
                self.assertIn(str(path), plan['binding']['input_files'])
            forbidden = Mock(side_effect=AssertionError('completed budget job repeated'))
            suite.run_suite(config, 'budgets', output, execute=True, resume=True, runner=forbidden)
            forbidden.assert_not_called()
            plan_before = (output / 'plan.json').read_bytes()
            state_before = (output / 'execution.json').read_bytes()
            for path in (scene, script):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b' changed')
                    with self.assertRaisesRegex(ValueError, 'Suite plan.*input binding changed'):
                        suite.run_suite(config, 'budgets', output, execute=True, resume=True, runner=forbidden)
                    self.assertEqual((output / 'plan.json').read_bytes(), plan_before)
                    self.assertEqual((output / 'execution.json').read_bytes(), state_before)
                    path.write_bytes(original)

    def test_input_hashing_is_shared_by_seed_and_budget_jobs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root)
            config['experiment']['seeds'] = list(range(10))
            scene = root / 'scene.blend'; scene.write_bytes(b'scene')
            script = root / 'custom_renderer.py'; script.write_bytes(b'script')
            config['renderer'] = {'enabled': True, 'blender': sys.executable,
                                  'scene': str(scene), 'script_path': str(script)}
            config['decision']['render_budget'] = 0
            with patch('tefed.resume.run_binding', wraps=run_binding) as measured:
                manifest = suite.make_manifest(config, 'budgets', root / 'budgets')
                self.assertEqual(len(manifest['jobs']), 50)
                self.assertEqual(measured.call_count, 1)
            self.assertEqual(len({job['configuration_sha256'] for job in manifest['jobs']}), 50)
            config['decision']['render_budget'] = 1
            with patch('tefed.resume.run_binding', wraps=run_binding) as measured:
                manifest = suite.make_manifest(config, 'comparisons', root / 'comparisons')
                self.assertEqual(len(manifest['jobs']), 110)
                self.assertEqual(measured.call_count, 2)
            for path in (scene, script):
                self.assertIn(str(path), manifest['binding']['input_files'])

    def test_native_completed_run_passes_suite_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); config = self.config(root); output = root / 'suite'
            config['experiment']['rounds'] = 1
            with patch.object(suite, 'plans', return_value=[('server', {'experiment': {'method': 'server_moe'}})]):
                suite.run_suite(config, 'comparisons', output, execute=True)
                with patch('tefed.engine.RoutedYOLO', side_effect=AssertionError('completed model reconstructed')):
                    result = suite.run_suite(config, 'comparisons', output, execute=True, resume=True)
            self.assertEqual(result['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
