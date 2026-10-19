"""Pure recorded-data regressions; no model, training, or optimizer execution."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tefed.baselines import policy_schedule
from tefed.experiments import artifact_bindings
from tefed.offline import paired_run_differences
from tefed.replay import (
    ReplayController, _training_binding, build_exposure_protocol, freeze_pool,
)
from tefed.reporting import plot_runs
from tests.test_offline_diagnostics import completed_run


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def recorded_run(root, name, seed, method, *, proximal_mu=.01, offset=0.):
    path = completed_run(root / name, seed, method, offset)
    config_path = path / 'configuration.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if method == 'fedprox':
        config['baselines'] = {'fedprox': {'proximal_mu': proximal_mu}}
    config['resolved_baseline_policies'] = policy_schedule(method, config)
    write_json(config_path, config)
    ledger_path = path / 'rounds.jsonl'
    rows = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        for ledger in row['clients'].values():
            ledger['mean_step_loss'] = 1. - .1 * seed
    ledger_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def replay_reference(root):
    """Bind a recorded reference with an empty pool and zero correction exposure."""
    manifest = root / 'manifest.json'
    calibration = root / 'calibration.json'
    write_json(manifest, {'records': []})
    write_json(calibration, {'T_C': 1.})
    config = {
        'experiment': {'method': 'tefed', 'rounds': 1, 'local_epochs': 2,
                       'batch_size': 4, 'seed': 0, 'test_mode': True,
                       'manifest': str(manifest), 'calibration': str(calibration)},
        'model': {'classes': 2},
        'optimizer': {'learning_rate': .01},
        'baselines': {'tefed': {'consistency_weight': .4}},
    }
    saved = deepcopy(config)
    saved['resolved_baseline_policies'] = policy_schedule('tefed', config)
    reference = root / 'reference'
    reference.mkdir()
    write_json(reference / 'configuration.json', saved)
    write_json(reference / 'environment.json', {'seed': 0, **artifact_bindings(config)})
    write_json(reference / 'status.json', {'status': 'completed'})
    row = {
        'round_index': 0,
        'clients': {'client1': {'presentations': {'correction': 0},
                               'attempted_optimizer_steps': 1, 'optimizer_steps': 1,
                               'amp_skipped_steps': 0}},
        'reuse_selected_ids': {'client1': []},
    }
    (reference / 'rounds.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    pool = freeze_pool([], {}, root / 'pool')
    protocol = build_exposure_protocol(pool, [reference], root / 'exposure.json')
    config['replay'] = {'pool_manifest': str(pool), 'exposure_protocol': str(protocol),
                        'arm': 'matched'}
    saved['replay'] = deepcopy(config['replay'])
    return config, saved


class PolicyRecordComparisonTests(unittest.TestCase):
    def test_cross_method_pairing_retains_recorded_policy_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = [recorded_run(root, f'avg{seed}', seed, 'fedavg') for seed in (0, 1)]
            right = [recorded_run(root, f'prox{seed}', seed, 'fedprox', offset=.1) for seed in (0, 1)]
            result = paired_run_differences(left, right, expected_seeds=(0, 1))
            score = next(row for row in result['summary'] if row['metric'] == 'mAP50')
            self.assertAlmostEqual(score['mean'], .1)
            settings = result['method_specific_settings']
            self.assertEqual(settings['left']['resolved_baseline_policies'][0]['method'], 'fedavg')
            self.assertEqual(settings['right']['resolved_baseline_policies'][0]['proximal_mu'], .01)
            self.assertIn('resolved_baseline_policies', result['allowed_method_sections'])

    def test_cross_method_plots_export_recorded_metrics_and_losses(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [recorded_run(root, f'{method}{seed}', seed, method)
                    for method in ('fedavg', 'fedprox') for seed in (0, 1)]
            output = root / 'plots'
            result = plot_runs(runs, output, seeds=[0, 1])
            self.assertEqual(result['metric_groups'], 6)
            self.assertEqual(result['series'], 8)
            for filename in ('metric_comparison.svg', 'training_curves.svg'):
                self.assertIn('<svg', (output / filename).read_text(encoding='utf-8'))
            values = (output / 'plot_values.csv').read_text(encoding='utf-8')
            self.assertIn('fedavg', values)
            self.assertIn('fedprox', values)

    def test_within_method_pairing_rejects_different_effective_policies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = [recorded_run(root, f'avg{seed}', seed, 'fedavg') for seed in (0, 1)]
            right = [recorded_run(root, f'prox{seed}', seed, 'fedprox', proximal_mu=.01 + .01 * seed)
                     for seed in (0, 1)]
            with self.assertRaisesRegex(ValueError, 'fixed protocol'):
                paired_run_differences(left, right, expected_seeds=(0, 1))

    def test_within_method_plots_reject_different_effective_policies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [recorded_run(root, f'prox{seed}', seed, 'fedprox', proximal_mu=.01 + .01 * seed)
                    for seed in (0, 1)]
            with self.assertRaisesRegex(ValueError, 'Within-method'):
                plot_runs(runs, root / 'plots', seeds=[0, 1])
            self.assertFalse((root / 'plots').exists())


class PolicyRecordReplayTests(unittest.TestCase):
    def test_raw_and_saved_training_bindings_match_without_mutating_inputs(self):
        raw = {'experiment': {'method': 'tefed', 'rounds': 1, 'local_epochs': 2},
               'model': {}, 'baselines': {'tefed': {'consistency_weight': .4}}}
        saved = deepcopy(raw)
        saved['resolved_baseline_policies'] = policy_schedule('tefed', raw)
        before = deepcopy(saved)
        self.assertEqual(_training_binding(raw), _training_binding(saved))
        self.assertEqual(saved, before)

    def test_saved_exposure_reference_accepts_raw_and_saved_replay_configurations(self):
        with TemporaryDirectory() as directory:
            raw, saved = replay_reference(Path(directory))
            for config in (raw, saved):
                for arm in ('matched', 'independent'):
                    with self.subTest(recorded='resolved_baseline_policies' in config, arm=arm):
                        config['replay']['arm'] = arm
                        controller = ReplayController(config, ['client1'], 1, 0)
                        self.assertEqual(controller.select({'q': .5, 'p': .5, 'h': [.2] * 5},
                                                           'client1', 0)['selected'], [])

    def test_replay_still_rejects_real_optimizer_and_baseline_changes(self):
        with TemporaryDirectory() as directory:
            raw, saved = replay_reference(Path(directory))
            for field in ('optimizer', 'baseline'):
                for config in (raw, saved):
                    with self.subTest(field=field, recorded='resolved_baseline_policies' in config):
                        changed = deepcopy(config)
                        if field == 'optimizer':
                            changed['optimizer']['learning_rate'] = .02
                        else:
                            changed['baselines']['tefed']['consistency_weight'] = .7
                            if 'resolved_baseline_policies' in changed:
                                changed['resolved_baseline_policies'] = policy_schedule('tefed', changed)
                        self.assertNotEqual(_training_binding(raw), _training_binding(changed))
                        with self.assertRaisesRegex(ValueError, 'training configuration differs'):
                            ReplayController(changed, ['client1'], 1, 0)

    def test_forged_recorded_policies_are_rejected(self):
        with TemporaryDirectory() as directory:
            raw, saved = replay_reference(Path(directory))
            for mutation in ('coefficient', 'method', 'missing_epoch'):
                with self.subTest(mutation=mutation):
                    forged = deepcopy(saved)
                    if mutation == 'coefficient':
                        forged['resolved_baseline_policies'][0]['consistency_weight'] = .9
                    elif mutation == 'method':
                        forged['resolved_baseline_policies'][0]['method'] = 'fedavg'
                    else:
                        forged['resolved_baseline_policies'].pop()
                    with self.assertRaisesRegex(ValueError, 'Recorded baseline policies differ'):
                        _training_binding(forged)
                    with self.assertRaisesRegex(ValueError, 'Recorded baseline policies differ'):
                        ReplayController(forged, ['client1'], 1, 0)
            self.assertEqual(_training_binding(raw), _training_binding(saved))


if __name__ == '__main__':
    unittest.main()
