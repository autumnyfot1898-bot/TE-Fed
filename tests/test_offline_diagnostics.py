"""Hand-computed review, route, and same-seed diagnostic checks."""
from copy import deepcopy
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_diagnostics import main
from tefed.offline import (FULL_BIN_EDGES, input_records, paired_run_differences,
                           reliability_diagnostics, routing_diagnostics, write_new_json)
from tests.test_analysis import write_run


def review(identity, reliability, correct):
    return {'id': identity, 'reliability': reliability, 'reviewed_correct': correct,
            'review_reference': {'kind': 'human_review', 'id': 'review-v1'}}


def candidate(identity='a', client='client0', round_index=0, route=None, weights=None):
    return {'client_id': client, 'round_index': round_index, 'local_identity': identity,
            'id': '0', 'route': route or [0, 1, 2], 'weights': weights or [.6, .3, .1],
            'rule_flags': [0, 0, 0, 0, 1]}


def condition(row, conditions):
    return {key: row[key] for key in ('client_id', 'round_index', 'local_identity', 'id')} | {
        'conditions': conditions, 'condition_source': 'manifest', 'reference_id': 'conditions-v1'}


def completed_run(path, seed, method, offset=0.):
    write_run(path, seed)
    config_path = path / 'configuration.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['experiment']['method'] = method
    config_path.write_text(json.dumps(config), encoding='utf-8')
    metric_path = path / 'metrics.json'
    metrics = json.loads(metric_path.read_text(encoding='utf-8'))
    for view in metrics.values():
        for key in view:
            view[key] += offset
    metric_path.write_text(json.dumps(metrics), encoding='utf-8')
    (path / 'status.json').write_text(json.dumps({'status': 'completed', 'completed_rounds': 1}), encoding='utf-8')
    (path / 'final.pt').write_bytes(b'completed-checkpoint-identity')
    return path


class ReliabilityDiagnosticsTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_paper_bin_boundaries(self):
        rows = [review('a', .4, False), review('b', .5, True), review('c', 1., True)]
        rows[1].pop('reviewed_correct')
        rows[1].update(reviewed_class_match=True, reviewed_iou=.5)
        result = reliability_diagnostics(rows)
        self.assertEqual([b['count'] for b in result['bins']], [1, 1, 0, 0, 0, 1])
        self.assertAlmostEqual(result['ECE'], .3)
        self.assertAlmostEqual(result['Brier'], (.16 + .25) / 3)
        self.assertAlmostEqual(result['NLL'], (-log(.6) - log(.5) - log(1 - 1e-6)) / 3)
        self.assertIsNone(result['bins'][2]['accuracy'])

    def test_zero_samples_are_undefined_and_expansion_is_explicit(self):
        result = reliability_diagnostics([])
        self.assertEqual(result['status'], 'empty')
        self.assertIsNone(result['ECE'])
        self.assertIsNone(result['Brier'])
        self.assertIsNone(result['NLL'])
        with self.assertRaisesRegex(ValueError, 'expanded'):
            reliability_diagnostics([review('a', 0., False)])
        result = reliability_diagnostics([review('a', 0., False)], bin_edges=FULL_BIN_EDGES, bin_policy='full_unit_interval')
        self.assertEqual(result['bins'][0]['count'], 1)
        self.assertAlmostEqual(result['NLL'], -log(1 - 1e-6))

    def test_correctness_requires_class_match_and_iou(self):
        rows = []
        for i, (matched, iou) in enumerate(((True, .49), (False, 1.), (True, .5))):
            row = review(str(i), .8, True)
            row.pop('reviewed_correct')
            row.update(reviewed_class_match=matched, reviewed_iou=iou)
            rows.append(row)
        result = reliability_diagnostics(rows)
        self.assertAlmostEqual(result['bins'][4]['accuracy'], 1 / 3)
        rows[-1]['reviewed_correct'] = False
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            reliability_diagnostics(rows)

    def test_review_rejects_missing_renderer_truth_nonfinite_and_duplicates(self):
        good = review('a', .7, True)
        bad_rows = [good | {'reliability': float('nan')}, good | {'reliability': True},
                    good | {'reviewed_correct': 1}, good | {'review_reference': {'kind': 'renderer', 'id': 'r'}},
                    good | {'review_reference': {'kind': 'human_review', 'id': 'r', 'independent': False}},
                    good | {'review_reference': {'kind': 'human_review', 'id': 'r', 'renderer_reference': True}},
                    {key: value for key, value in good.items() if key != 'reviewed_correct'}]
        for bad in bad_rows:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reliability_diagnostics([bad])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reliability_diagnostics([good, deepcopy(good)])
        for edges in ((.4, .4, 1.), (.4, float('inf')), (.4,)):
            with self.assertRaises(ValueError):
                reliability_diagnostics([good], bin_edges=edges)

    def test_optional_client_round_context_and_reliability_aliases(self):
        row = review('a', .6, True) | {'client_id': 'client0', 'round_index': 2}
        self.assertEqual(reliability_diagnostics([row])['count'], 1)
        with self.assertRaisesRegex(ValueError, 'aliases'):
            reliability_diagnostics([row | {'R': .7}])


class RoutingDiagnosticsTests(unittest.TestCase):
    def test_first_manifest_label_and_specialist_only_top1(self):
        a, b, c = candidate('a'), candidate('b', route=[0, 3, 4], weights=[.7, .2, .1]), candidate('c')
        result = routing_diagnostics([a, b, c], [condition(a, ['occlusion', 'illumination']),
                                                condition(b, ['clutter']), condition(c, ['nominal'])])
        self.assertEqual(result['correspondence'], {'count': 2, 'matches': 1, 'fraction': .5,
                                                     'excluded_nominal': 1, 'excluded_missing': 0})
        self.assertEqual(result['clients']['client0']['top1_specialist_counts']['1'], 2)
        self.assertAlmostEqual(result['groups']['context']['mean_route_weights'][0], .7)
        self.assertEqual(result['count'], 3)

    def test_missing_labels_and_no_specialist_are_counted(self):
        a, b = candidate('a'), candidate('b', route=[0], weights=[1.])
        result = routing_diagnostics([a, b], [condition(a, [])])
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['groups']['__missing__']['count'], 2)
        self.assertEqual(result['correspondence']['excluded_missing'], 2)
        self.assertIsNone(result['correspondence']['fraction'])
        self.assertEqual(result['clients']['client0']['top1_specialist_counts']['none'], 1)

    def test_scoped_identity_join_does_not_mix_clients_or_rounds(self):
        a, b, c = candidate('a'), candidate('a', client='client1'), candidate('a', round_index=1)
        labels = condition(a, ['illumination'])
        labels.pop('id')
        result = routing_diagnostics([a, b, c], [labels], join_on='identity')
        self.assertEqual(result['correspondence']['count'], 1)
        self.assertEqual(len(result['missing_conditions']), 2)

    def test_client_round_and_event_id_join_modes_and_ties(self):
        a = candidate('a', route=[0, 3, 1], weights=[.6, .2, .2])
        labels = condition(a, ['illumination'])
        labels.pop('local_identity'); labels.pop('id')
        result = routing_diagnostics([a], [labels], join_on='client_round')
        self.assertEqual(result['correspondence']['matches'], 1)
        a['event_id'] = 'unique-event'
        labels = {'event_id': 'unique-event', 'conditions': ['illumination'],
                  'condition_source': 'independent_annotation', 'reference_id': 'v1'}
        self.assertEqual(routing_diagnostics([a], [labels], join_on='event_id')['correspondence']['matches'], 1)

    def test_rule_flags_never_supply_conditions_and_bad_routes_fail(self):
        a = candidate()
        result = routing_diagnostics([a], [])
        self.assertEqual(len(result['missing_conditions']), 1)
        with self.assertRaisesRegex(ValueError, 'condition_source'):
            routing_diagnostics([a], [condition(a, ['sensing']) | {'condition_source': 'rule_flags'}])
        for bad in (a | {'weights': [.6, .3, .3]}, a | {'route': [0, 1, 1]}, a | {'weights': [.6, float('nan'), .4]}):
            with self.assertRaises(ValueError):
                routing_diagnostics([bad], [])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            routing_diagnostics([a, a], [])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            routing_diagnostics([a], [condition(a, ['sensing'])] * 2)


class PairedDiagnosticsTests(unittest.TestCase):
    def _runs(self, folder):
        root = Path(folder)
        left = [completed_run(root / f'left{seed}', seed, 'server_moe') for seed in (0, 1)]
        right = [completed_run(root / f'right{seed}', seed, 'tefed', .1 + .1 * seed) for seed in (0, 1)]
        return left, right

    def test_actual_same_seed_differences_and_sample_sd(self):
        with TemporaryDirectory() as folder:
            left, right = self._runs(folder)
            for path in right:
                file = path / 'configuration.json'
                config = json.loads(file.read_text(encoding='utf-8'))
                config['decision']['render_budget'] = 576
                config['loss']['beta'] = .2
                file.write_text(json.dumps(config), encoding='utf-8')
            result = paired_run_differences(left, right[::-1], expected_seeds=(0, 1))
            row = next(row for row in result['summary'] if row['metric'] == 'mAP50')
            self.assertAlmostEqual(row['mean'], .15)
            self.assertAlmostEqual(row['sample_sd'], .1 / sqrt(2))
            self.assertEqual([x['seed'] for x in row['per_seed']], ['0', '1'])
            self.assertAlmostEqual(row['per_seed'][0]['right_minus_left'], .1)
            self.assertIn('final.pt', result['left']['runs'][0]['files_sha256'])
            self.assertIn('no confidence interval', result['inference'])

    def test_rejects_missing_or_duplicate_seed_and_single_sample(self):
        with TemporaryDirectory() as folder:
            left, right = self._runs(folder)
            for a, b, seeds in ((left, right[:1], (0, 1)), (left, right + right[:1], (0, 1)),
                                (left[:1], right[:1], (0,))):
                with self.assertRaises(ValueError):
                    paired_run_differences(a, b, expected_seeds=seeds)

    def test_old_seed_mixed_protocol_rejected_before_subtraction(self):
        with TemporaryDirectory() as folder:
            left, right = self._runs(folder)
            file = right[1] / 'configuration.json'
            config = json.loads(file.read_text(encoding='utf-8'))
            config['decision']['render_budget'] = 577
            file.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fixed protocol'):
                paired_run_differences(left, right, expected_seeds=(0, 1))

    def test_cross_method_must_share_inputs_model_and_training(self):
        changes = [('environment.json', 'initial_checkpoint_sha256', 'd' * 64),
                   ('environment.json', 'calibration_sha256', 'd' * 64),
                   ('environment.json', 'manifest_sha256', 'd' * 64),
                   ('configuration.json', 'optimizer.learning_rate', .02),
                   ('configuration.json', 'model.classes', 3),
                   ('configuration.json', 'evaluation.views', ['participating'])]
        for filename, key, value in changes:
            with self.subTest(key=key), TemporaryDirectory() as folder:
                left, right = self._runs(folder)
                for path in right:
                    file = path / filename
                    data = json.loads(file.read_text(encoding='utf-8'))
                    if '.' in key:
                        section, name = key.split('.')
                        data[section][name] = value
                    else:
                        data[key] = value
                    file.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'different data'):
                    paired_run_differences(left, right, expected_seeds=(0, 1))

    def test_unfinished_runs_and_absent_initial_hash_rejected(self):
        for field in ('status', 'status_round_type', 'checkpoint', 'calibration_snapshot', 'initial_hash'):
            with self.subTest(field=field), TemporaryDirectory() as folder:
                left, right = self._runs(folder)
                for path in right:
                    if field == 'status':
                        (path / 'status.json').write_text('{"status":"running","completed_rounds":1}', encoding='utf-8')
                    elif field == 'status_round_type':
                        (path / 'status.json').write_text('{"status":"completed","completed_rounds":true}', encoding='utf-8')
                    elif field == 'checkpoint':
                        (path / 'final.pt').unlink()
                    elif field == 'calibration_snapshot':
                        (path / 'calibration.json').unlink()
                    else:
                        file = path / 'environment.json'
                        data = json.loads(file.read_text(encoding='utf-8'))
                        data['initial_checkpoint_sha256'] = None
                        file.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    paired_run_differences(left, right, expected_seeds=(0, 1))


class OfflineCliTests(unittest.TestCase):
    def test_hashes_match_parsed_input_and_new_atomic_output(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'reviews.jsonl'
            path.write_text(json.dumps(review('a', .8, True)) + '\n', encoding='utf-8')
            rows, manifest = input_records(path)
            self.assertEqual(manifest['sha256'], sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(rows), 1)
            output = Path(folder) / 'result.json'
            self.assertEqual(main(['reliability', '--records', str(path), '--out', str(output)]), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['inputs']['review_records']['sha256'], manifest['sha256'])
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new_json(output, {'changed': True})
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_atomic_invalid_json_never_publishes_result(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'invalid.json'
            with self.assertRaises(ValueError):
                write_new_json(output, {'value': float('nan')})
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_cli_full_bins_routing_and_paired_paths(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            reviews = root / 'review.jsonl'
            reviews.write_text(json.dumps(review('a', 0., False)), encoding='utf-8')
            self.assertEqual(main(['reliability', '--records', str(reviews), '--bins', 'full', '--out', str(root / 'review.json')]), 0)
            a = candidate()
            for name, row in (('candidates', a), ('conditions', condition(a, ['illumination']))):
                (root / f'{name}.jsonl').write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(main(['routing', '--candidates', str(root / 'candidates.jsonl'), '--conditions', str(root / 'conditions.jsonl'), '--out', str(root / 'routes.json')]), 0)
            left = [completed_run(root / f'a{seed}', seed, 'server_moe') for seed in (0, 1)]
            right = [completed_run(root / f'b{seed}', seed, 'tefed', .1) for seed in (0, 1)]
            self.assertEqual(main(['paired', '--left', *map(str, left), '--right', *map(str, right), '--seeds', '0', '1', '--out', str(root / 'paired.json')]), 0)


if __name__ == '__main__':
    unittest.main()
