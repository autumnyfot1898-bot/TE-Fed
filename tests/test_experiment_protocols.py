"""Frozen-pool, strict-exposure and executable-factorial checks."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from tefed.experiments import (apply_qualification_control, artifact_bindings, factorial_manifest,
                               factorial_plans, validate_factorial_config)
from tefed.replay import (ExposureError, FrozenPool, ReplayController, build_exposure_protocol,
                          freeze_pool, validate_replay_after, validate_replay_before)
from tefed.software_fixture import create_fixture
from tests.test_core import STATE, descriptor
from scripts.run_suite import merge


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def training_ledger(cases=0, attempted=1, applied=1):
    return {'presentations': {'correction': cases}, 'attempted_optimizer_steps': attempted,
            'optimizer_steps': applied, 'amp_skipped_steps': attempted-applied}


def setup_replay(root, *, native=False):
    config = create_fixture(root/'fixture')
    config['experiment']['seeds'] = [0]
    config['experiment']['local_epochs'] = 1
    asset = root/'original.png'; asset.write_bytes(b'private-software-fixture')
    d = descriptor()
    case = {'id': 'case-1', 'round': 0, 'd': d, 'fidelity': .9, 'rgb': str(asset),
            'targets': [{'box': [.1, .1, .3, .3], 'class_id': 0}]}
    if native:
        record = json.loads(Path(config['experiment']['manifest']).read_text())['records'][0]
        case.update(rgb=record['rgb'],depth=record['depth'],nir=record['nir'],
                    depth_unit_conversion=1.,targets=record['labels'],fixture_only=True)
        config['decision']['reuse_gate'] = 0.
    pool = freeze_pool([case], {d['source_state_key']: STATE}, root/'pool')
    reference = root/'reference'; reference.mkdir()
    write_json(reference/'environment.json', {'seed': 0, **artifact_bindings(config)})
    write_json(reference/'configuration.json', config)
    write_json(reference/'status.json', {'status': 'completed'})
    clients = ('client0','client1') if native else ('client1',)
    rows = [{'round_index': r, 'clients': {client:training_ledger(r) for client in clients},
             'reuse_selected_ids': {client:[] if r==0 else ['case-1'] for client in clients}} for r in range(2)]
    (reference/'rounds.jsonl').write_text('\n'.join(map(json.dumps, rows))+'\n', encoding='utf-8')
    protocol = build_exposure_protocol(pool, [reference], root/'exposure.json')
    config['replay'] = {'pool_manifest': str(pool), 'exposure_protocol': str(protocol), 'arm': 'matched'}
    return config, case, rows


class ReplayProtocolTests(unittest.TestCase):
    def test_frozen_copy_and_binding_reject_asset_or_manifest_changes(self):
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            pool = FrozenPool(config['replay']['pool_manifest'])
            returned = pool.cases; returned[0]['d']['attrs'][0] = .99
            self.assertNotEqual(returned[0]['d']['attrs'], pool.cases[0]['d']['attrs'])
            Path(directory, 'original.png').write_bytes(b'original-source-changed')
            pool.verify()
            Path(pool.cases[0]['rgb']).write_bytes(b'tampered-bundle')
            with self.assertRaisesRegex(ValueError, 'Frozen asset'):
                pool.verify()

    def test_selectors_share_pool_and_preserve_next_round_eligibility(self):
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            controllers = []
            for arm in ('matched', 'independent'):
                config['replay']['arm'] = arm
                controller = ReplayController(config, ['client1'], 2, 0)
                self.assertEqual(controller.select(STATE, 'client1', 0)['selected'], [])
                chosen = controller.select(STATE, 'client1', 1, {'reuse_gate': 0})
                self.assertEqual([case['id'] for case in chosen['selected']], ['case-1'])
                controllers.append(controller)
            self.assertEqual(controllers[0].metadata()['pool_sha256'], controllers[1].metadata()['pool_sha256'])
            self.assertEqual(controllers[0].metadata()['protocol_sha256'], controllers[1].metadata()['protocol_sha256'])
            with self.assertRaisesRegex(ExposureError, 'no filling'):
                controllers[0].select(STATE, 'client1', 1, {'reuse_gate': 1})

    def test_exact_assembled_counts_and_amp_applied_updates_are_required(self):
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            controller = ReplayController(config, ['client1'], 2, 0)
            chosen = controller.select(STATE, 'client1', 1, {'reuse_gate': 0})
            items = [{'kind': 'labeled'}, {'kind': 'correction', 'case': chosen['selected'][0]}]
            validate_replay_before(items, config, chosen['control'])
            record = validate_replay_after(training_ledger(1), chosen['control'])
            self.assertTrue(record['matched'])
            with self.assertRaisesRegex(ExposureError, 'cases differ'):
                validate_replay_before(items+[items[1]], config, chosen['control'])
            with self.assertRaisesRegex(ExposureError, 'pre-training'):
                validate_replay_before(items+[{'kind':'labeled'}]*4, config, chosen['control'])
            with self.assertRaisesRegex(ExposureError, 'AMP'):
                validate_replay_after(training_ledger(1, 1, 0), chosen['control'])

    def test_resume_reconstructs_accounting_without_mutable_counters(self):
        with TemporaryDirectory() as directory:
            config, _, rows = setup_replay(Path(directory))
            controller = ReplayController(config, ['client1'], 2, 0)
            for row in rows:
                selected = controller.select(STATE, 'client1', row['round_index'], {'reuse_gate': 0})
                row['clients']['client1']['replay'] = validate_replay_after(row['clients']['client1'], selected['control'])
            state = controller.state_dict()
            restored = ReplayController(config, ['client1'], 2, 0)
            restored.load_state_dict(state); restored.verify_history(rows)
            changed = deepcopy(rows); changed[1]['clients']['client1']['optimizer_steps'] = 0
            changed[1]['clients']['client1']['amp_skipped_steps'] = 1
            with self.assertRaises(ExposureError):
                restored.verify_history(changed)
            state['pool_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'Resume replay'):
                restored.load_state_dict(state)

    def test_protocol_requires_complete_reference_and_unchanged_inputs(self):
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            Path(config['experiment']['calibration']).write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'calibration or initial'):
                ReplayController(config, ['client1'], 2, 0)

    def test_protocol_binds_training_settings_beyond_exposure_counts(self):
        with TemporaryDirectory() as directory:
            config, _, _ = setup_replay(Path(directory))
            config['loss']['beta'] = .25
            with self.assertRaisesRegex(ValueError, 'training configuration'):
                ReplayController(config, ['client1'], 2, 0)

    def test_native_nonempty_selector_replay_and_interrupted_resume(self):
        from tefed import client
        from tefed.engine import run_experiment
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = setup_replay(root, native=True)
            matched = run_experiment(config, seed=0, output_dir=root/'matched')
            config['replay']['arm'] = 'independent'
            original = client.execute_client_round
            def interrupted(*args, **kwargs):
                if kwargs['round_index']==1:
                    raise RuntimeError('test round boundary interruption')
                return original(*args, **kwargs)
            with patch('tefed.client.execute_client_round', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'test round boundary'):
                    run_experiment(config, seed=0, output_dir=root/'independent')
            independent = run_experiment(config, seed=0, output_dir=root/'independent',resume=root/'independent'/'resume.pt')
            for left,right in zip(matched['rounds'],independent['rounds']):
                self.assertEqual(left['replay']['pool_sha256'],right['replay']['pool_sha256'])
                for name in ('client0','client1'):
                    self.assertEqual(left['clients'][name]['replay']['actual'],right['clients'][name]['replay']['actual'])
            self.assertEqual(matched['rounds'][1]['clients']['client0']['presentations']['correction'],1)
            self.assertEqual(independent['rounds'][1]['clients']['client1']['presentations']['correction'],1)
            final = torch.load(independent['checkpoint'],map_location='cpu',weights_only=True)
            self.assertTrue(all(torch.isfinite(value).all() for value in final['model'].values()))


class FactorialProtocolTests(unittest.TestCase):
    def test_eight_cells_use_existing_methods_and_explicit_operations(self):
        jobs = factorial_plans()
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({tuple(patch['factorial']['factors'].values()) for _, patch in jobs}), 8)
        for _, patch in jobs:
            config = merge({'experiment': {'test_mode': True}}, patch)
            levels = validate_factorial_config(config)
            method = config['experiment']['method']
            self.assertEqual(method, 'server_moe' if not levels['C'] else 'tefed' if levels['B'] else 'generic_hc')

    def test_A_off_only_changes_qualification_and_keeps_validity_and_weights(self):
        off = next(patch for _, patch in factorial_plans() if patch['factorial']['factors']=={'A':0,'B':1,'C':1})
        config = merge({'experiment': {'test_mode': True}}, off)
        candidate = {'valid': True, 'R': .61, 'D_E': .5, 'posterior': [.8, .2],
                     'cues': {'C':.7,'G':.8}, 'disposition':'correction', 'route':[0,1,2], 'weights':[.5,.3,.2]}
        result = apply_qualification_control(candidate, config)
        self.assertEqual(result['disposition'], 'qualified')
        for key in ('R','route','weights'):
            self.assertEqual(result[key], candidate[key])
        self.assertEqual(candidate['disposition'], 'correction')
        self.assertEqual(apply_qualification_control({**candidate,'valid':False}, config)['disposition'], 'suppressed')
        self.assertEqual(apply_qualification_control({**candidate,'posterior':[.6,.4]}, config)['disposition'], 'correction')
        full = merge(config, dict(factorial=next(patch['factorial'] for _, patch in factorial_plans() if patch['factorial']['factors']=={'A':1,'B':1,'C':1})))
        self.assertIs(apply_qualification_control(candidate, full), candidate)

    def test_factorial_manifest_binds_seed_operations_and_artifacts(self):
        with TemporaryDirectory() as directory:
            config = create_fixture(Path(directory)/'fixture')
            manifest = factorial_manifest(config, [0,1], Path(directory)/'suite')
            self.assertEqual(len(manifest['jobs']), 16)
            self.assertEqual(manifest['scope'], 'software_fixture')
            self.assertIn('no training effect', manifest['structural_null'])
            self.assertTrue(all(job['bindings']==manifest['bindings'] for job in manifest['jobs']))
            changed = deepcopy(config); changed['loss']['beta'] = 0.
            self.assertNotEqual(manifest['protocol_sha256'], factorial_manifest(changed,[0,1],Path(directory)/'suite')['protocol_sha256'])
            with self.assertRaises(ValueError):
                factorial_manifest(config, [0,0], Path(directory)/'suite')

    def test_operation_tampering_and_inactive_real_correction_are_rejected(self):
        config = next(patch for _, patch in factorial_plans() if patch['factorial']['factors']=={'A':1,'B':1,'C':1})
        config = merge(config, {'renderer': {'enabled':False}})
        with self.assertRaisesRegex(ValueError, 'C=1'):
            validate_factorial_config(config)
        config['experiment']['test_mode'] = True
        config['factorial']['definitions_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'definitions'):
            validate_factorial_config(config)

    def test_remote_factorial_fails_before_worker_or_model_construction(self):
        from tefed.engine import run_experiment
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root/'fixture')
            config = merge(config, factorial_plans()[-1][1])
            config['communication'] = {'mode':'http','workers':{}}
            with patch('tefed.engine.RoutedYOLO') as model, patch('tefed.worker.RemoteClientExecutor') as remote:
                with self.assertRaisesRegex(ValueError, 'factorial requires communication.mode=local'):
                    run_experiment(config, output_dir=root/'must-not-exist')
                model.assert_not_called(); remote.assert_not_called()
            self.assertFalse((root/'must-not-exist').exists())

    def test_native_eight_cells_and_full_match_regular_tefed(self):
        from tefed.engine import run_experiment
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root/'fixture')
            config['experiment'].update(rounds=1, seeds=[0])
            results = {}
            for variant, patch in factorial_plans():
                result = run_experiment(merge(config, patch), seed=0, output_dir=root/variant)
                self.assertTrue(Path(result['checkpoint']).is_file())
                self.assertEqual(len(result['rounds']), 1)
                results[variant] = result
            regular = run_experiment(config, seed=0, output_dir=root/'regular')
            full = torch.load(results['A1_B1_C1']['checkpoint'], map_location='cpu', weights_only=True)
            baseline = torch.load(regular['checkpoint'], map_location='cpu', weights_only=True)
            self.assertEqual(results['A1_B1_C1']['metrics'], regular['metrics'])
            for name in full['model']:
                self.assertTrue(torch.equal(full['model'][name], baseline['model'][name]), name)


if __name__=='__main__':
    unittest.main()
