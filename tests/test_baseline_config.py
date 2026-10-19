"""Baseline overrides must reach training, selection, aggregation and records."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tefed import engine
from tefed.baselines import (
    METHOD_REGISTRY, adaptive_aggregation_weights, make_ema_teacher,
    policy_dict, policy_schedule, update_ema_teacher, validate_baseline_config,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Identity()
        self.heads = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        with torch.no_grad():
            self.heads[0].weight.copy_(torch.tensor([[1., .2], [.4, 1.]]))

    def forward(self, x, active_heads):
        return {head: self.heads[head](x) for head in active_heads}

    def decode_head(self, raw, head, image_shape):
        return raw


class TinyCriterion:
    def __call__(self, raw, targets, weights, device):
        return raw.square().mean(), {}

    def epoch_update(self):
        pass


def tiny_config(method, **overrides):
    return {
        'experiment': {'method': method, 'rounds': 1, 'local_epochs': 2, 'batch_size': 1},
        'model': {}, 'loss': {'rho': 0.},
        'optimizer': {'amp': False, 'learning_rate': .01, 'final_learning_rate': .01,
                      'warmup_rounds': 0, 'momentum': 0., 'weight_decay': 0.},
        'baselines': {method: overrides},
    }


class BaselineConfigurationTests(unittest.TestCase):
    def test_empty_overrides_preserve_default_schedule(self):
        for method in METHOD_REGISTRY:
            for index in [(0, 0), (12, 1), (29, 1)]:
                self.assertEqual(policy_dict(method, *index), policy_dict(method, *index, {'baselines': {method: {}}}))
        self.assertEqual(policy_dict('fedprox', 0, 0)['proximal_mu'], .01)
        self.assertEqual(policy_dict('fedsto', 0, 0)['orthogonal_coefficient'], .05)
        self.assertEqual(policy_dict('fsslod', 0, 0)['elastic_coefficient'], .1)
        self.assertEqual(policy_dict('consistent_teacher_fedawa', 0, 0)['reference_weighting'], 'uniform')

    def test_invalid_tables_are_rejected_without_lock_or_output(self):
        invalid = [[], {'unknown': {}}, {'fedprox': []}, {'fedprox': {'unknown': 1}},
                   {'fedavg': {'pseudo_threshold': .8}}, {'confidence': {'teacher_decay': .9}},
                   {'tefed': {'pseudo_threshold': .8}}, {'fedsto': {'proximal_mu': .1}},
                   {'fsslod': {'teacher_decay': .9, 'teacher_decay_start': .8}},
                   {'fsslod': {'teacher_decay_start': .9999}},
                   {'consistent_teacher_fedawa': {'reference_weighting': 'samples'}},
                   {'consistent_teacher_fedawa': {'reference_weighting': []}}]
        for key in ['pseudo_threshold', 'teacher_decay', 'consistency_weight']:
            for value in [True, '0.8', None, -1, float('nan'), float('inf')]:
                invalid.append({'fedsto': {key: value}})
        invalid.append({'confidence': {'pseudo_threshold': 1.01}})
        for key in ['aggregation_temperature', 'aggregation_cap']:
            invalid.append({'consistent_teacher_fedawa': {key: 0}})
        invalid.append({'consistent_teacher_fedawa': {'aggregation_cap': 1.1}})
        invalid.extend([
            {'consistent_teacher_fedawa': {'aggregation_mode': 'unknown'}},
            {'consistent_teacher_fedawa': {'aggregation_mode': []}},
            {'consistent_teacher_fedawa': {'fedawa_steps': 1}},
            {'consistent_teacher_fedawa': {'aggregation_mode': 'fedawa_paper', 'aggregation_cap': .8}},
            {'consistent_teacher_fedawa': {'aggregation_mode': 'fedawa_paper', 'reference_weighting': 'uniform'}},
        ])
        for key, values in [('fedawa_steps', [True, 1.5, 0, -1]),
                            ('fedawa_learning_rate', [True, '0.1', 0, -1, float('nan')])]:
            for value in values:
                invalid.append({'consistent_teacher_fedawa': {'aggregation_mode': 'fedawa_paper', key: value}})
        for value in [True, 1.5, -1, 2]:
            invalid.append({'fedsto': {'head_only_epochs': value}})
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            for table in invalid:
                config = {'experiment': {'method': 'fedavg'}, 'baselines': table}
                with self.subTest(table=table), self.assertRaises(ValueError):
                    engine.run_experiment(config, output_dir=parent / 'run')
                self.assertEqual(list(parent.iterdir()), [])

    def test_scheduled_and_constant_ema_and_explicit_stage_allocation(self):
        config = tiny_config('fsslod', teacher_decay_start=.8, teacher_decay_end=.95)
        config['experiment'].update(rounds=4, local_epochs=3)
        self.assertEqual(policy_dict('fsslod', 0, 0, config)['teacher_decay'], .8)
        self.assertEqual(policy_dict('fsslod', 3, 2, config)['teacher_decay'], .95)
        config['baselines']['fsslod'] = {'teacher_decay': .7}
        self.assertEqual(policy_dict('fsslod', 3, 2, config)['teacher_decay'], .7)
        config = tiny_config('fedsto', head_only_epochs=0)
        self.assertFalse(policy_dict('fedsto', 0, 0, config)['freeze_trunk'])
        config['experiment']['local_epochs'] = 1
        config['baselines']['fedsto'] = {}
        self.assertTrue(policy_dict('fedsto', 0, 0, config)['freeze_trunk'])
        config['baselines']['fedsto'] = {'head_only_epochs': 1}
        with self.assertRaises(ValueError):
            validate_baseline_config(config)

    def test_policy_indices_follow_the_actual_training_dimensions(self):
        config = {'experiment': {'rounds': 41, 'local_epochs': 4}}
        for method in METHOD_REGISTRY:
            with self.subTest(method=method):
                policy = policy_dict(method, 40, 3, config)
                self.assertEqual(policy['round_index'], 40)
                self.assertEqual(policy['epoch_index'], 3)
                for round_index, epoch_index in [(41, 0), (0, 4), (-1, 0), (0, -1), (True, 0), (0, .5)]:
                    with self.assertRaises(ValueError):
                        policy_dict(method, round_index, epoch_index, config)
        self.assertAlmostEqual(policy_dict('fsslod', 40, 3, config)['teacher_decay'], .999)
        self.assertEqual(policy_dict('fsslod', 0, 0, {'experiment': {'rounds': 1, 'local_epochs': 1}})['teacher_decay'], .99)
        from tefed.baselines import fedsto_set_trainable
        self.assertFalse(fedsto_set_trainable(TinyModel(), 3, head_only_epochs=2)['freeze_trunk'])
        self.assertTrue(fedsto_set_trainable(TinyModel(), 1, head_only_epochs=2)['freeze_trunk'])

    def test_directional_reference_changes_only_when_explicitly_weighted(self):
        vectors = [[1., 0.], [0., 1.], [-1., 0.]]
        uniform = adaptive_aggregation_weights(vectors, cap=.8)
        np.testing.assert_allclose(uniform, adaptive_aggregation_weights(vectors, sample_counts=[1, 1, 1], cap=.8))
        weighted = adaptive_aggregation_weights(vectors, sample_counts=[8, 1, 1], cap=.8)
        self.assertGreater(weighted[0], uniform[0])
        self.assertLess(weighted[2], uniform[2])

    def test_assessment_threshold_changes_candidate_disposition(self):
        class Detector(TinyModel):
            last_proposal_counts = {0: [1]}

            def features(self, x):
                return x

            def predict_heads(self, x, heads, **kwargs):
                return {0: [[{'box': [.1, .1, .9, .9], 'class_id': 0,
                             'posterior': [.8, .2], 'score': .8}]]}

        model = Detector()
        sample = {'id': 'test', 'rgb': np.zeros((32, 32, 3), dtype=np.uint8)}
        cue = {'attrs': [], 'availability': [], 'rule_flags': [], 'AM': 1., 'G': 1., 'S': 1.}
        with patch.object(engine, 'sample_tensor', return_value=torch.zeros(1, 5, 32, 32)), \
                patch('tefed.cues.extract_cues', return_value=cue), \
                patch('tefed.cues.routing_vector', return_value=[0.] * 12):
            default = engine.assess_frame(model, None, sample, {}, {}, tiny_config('confidence'), experts=False)
            configured = engine.assess_frame(model, None, sample, {}, {}, tiny_config('confidence', pseudo_threshold=.9), experts=False)
        self.assertEqual(default['candidates'][0]['disposition'], 'qualified')
        self.assertEqual(configured['candidates'][0]['disposition'], 'suppressed')
        self.assertEqual(configured['candidates'][0]['threshold'], .9)

    def train_tiny(self, method, overrides, *, teacher=False, pseudo=False):
        model = TinyModel()
        reference = deepcopy(model)
        with torch.no_grad():
            reference.heads[0].weight.zero_()
        ema = make_ema_teacher(model) if teacher else None
        sample = {'targets': [], 'rgb': np.zeros((32, 32, 3), dtype=np.uint8)}
        item = {'kind': 'labeled', 'sample': sample, 'route_weights': {0: 1.}}
        if pseudo:
            item = {'kind': 'pseudo', 'sample': sample, 'candidates': [{'R': 1., 'route': [0], 'weights': [1.]}]}
        with patch.object(engine, 'WeightedNativeLoss', side_effect=lambda *args: TinyCriterion()), \
                patch.object(engine, 'sample_tensor', return_value=torch.ones(1, 2)):
            ledger = engine.train_local(model, [item], tiny_config(method, **overrides),
                                        method=method, teacher=ema, reference=reference)
        return model, ema, ledger

    def test_each_regularizer_override_changes_actual_optimizer_result(self):
        for method, key in [('fedprox', 'proximal_mu'), ('fedsto', 'orthogonal_coefficient'), ('fsslod', 'elastic_coefficient')]:
            with self.subTest(method=method):
                disabled, _, _ = self.train_tiny(method, {key: 0.})
                enabled, _, ledger = self.train_tiny(method, {key: 2.})
                self.assertFalse(torch.allclose(disabled.heads[0].weight, enabled.heads[0].weight))
                self.assertTrue(all(policy[key] == 2. for policy in ledger['baseline_policies']))

    def test_ema_and_stage_records_match_executed_updates(self):
        with patch('tefed.baselines.update_ema_teacher', wraps=update_ema_teacher) as update:
            _, teacher, ledger = self.train_tiny('fsslod', {'teacher_decay_start': .8, 'teacher_decay_end': .95}, teacher=True)
        self.assertEqual(ledger['teacher_updates'], ledger['optimizer_steps'])
        self.assertEqual([call.args[2] for call in update.call_args_list],
                         [policy['teacher_decay'] for policy in ledger['baseline_policies']])
        self.assertFalse(teacher.training)
        _, _, ledger = self.train_tiny('fedsto', {'head_only_epochs': 0})
        self.assertEqual([stage['freeze_trunk'] for stage in ledger['stage_history']], [False, False])

    def test_consistency_weight_changes_actual_optimizer_result(self):
        def consistency(weak, strong, **kwargs):
            return strong.square().sum(), 1
        with patch.object(engine, 'soft_consistency', side_effect=consistency):
            disabled, _, no_loss = self.train_tiny('consistent_teacher_fedawa', {'consistency_weight': 0.}, teacher=True, pseudo=True)
            enabled, _, ledger = self.train_tiny('consistent_teacher_fedawa', {'consistency_weight': .8}, teacher=True, pseudo=True)
        self.assertFalse(torch.allclose(disabled.heads[0].weight, enabled.heads[0].weight))
        self.assertEqual(no_loss['consistency_pairs'], 0)
        self.assertEqual(ledger['consistency_pairs'], 2)
        self.assertTrue(all(policy['consistency_weight'] == .8 for policy in ledger['baseline_policies']))


class NativeBaselineConfigurationTests(unittest.TestCase):
    def test_paper_aggregation_resumes_logits_and_rejects_changed_state(self):
        from tefed.fedawa import paper_aggregation_weights
        from tefed.resume import atomic_snapshot, load_snapshot
        from tefed.software_fixture import create_fixture
        from tests.test_resume import stable_ledger
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            method = 'consistent_teacher_fedawa'
            config['experiment']['method'] = method
            config['baselines'] = {method: {'aggregation_mode': 'fedawa_paper',
                'pseudo_threshold': 0., 'fedawa_steps': 2, 'fedawa_learning_rate': .002}}
            with patch('tefed.fedawa.paper_aggregation_weights', wraps=paper_aggregation_weights) as aggregate:
                continuous = engine.run_experiment(config, output_dir=root / 'continuous')
            self.assertIsNone(aggregate.call_args_list[0].kwargs['logits'])
            self.assertEqual(aggregate.call_args_list[1].kwargs['logits'], continuous['rounds'][0]['aggregation']['logits'])
            self.assertEqual(aggregate.call_args.kwargs['steps'], 2)
            self.assertEqual(aggregate.call_args.kwargs['learning_rate'], .002)
            original = engine.train_local

            def interrupt(model, items, configuration, **kwargs):
                if kwargs['round_id'] == 1:
                    raise RuntimeError('test interruption')
                return original(model, items, configuration, **kwargs)

            output = root / 'interrupted'
            with patch.object(engine, 'train_local', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    engine.run_experiment(config, output_dir=output)
            partial = load_snapshot(output / 'resume.pt')
            self.assertEqual(partial['aggregation_state']['logits'], continuous['rounds'][0]['aggregation']['logits'])
            resumed = engine.run_experiment(config, output_dir=output, resume=output / 'resume.pt')
            self.assertEqual(stable_ledger(continuous['rounds']), stable_ledger(resumed['rounds']))
            expected = load_snapshot(root / 'continuous' / 'resume.pt')
            actual = load_snapshot(output / 'resume.pt')
            self.assertEqual(actual['aggregation_state'], expected['aggregation_state'])
            for key in actual['model']:
                self.assertTrue(torch.equal(actual['model'][key], expected['model'][key]))
            for mutate in ['ordering', 'logits']:
                changed = deepcopy(actual)
                if mutate == 'ordering':
                    changed['aggregation_state']['client_ids'].reverse()
                else:
                    changed['aggregation_state']['logits'][0] += .1
                atomic_snapshot(output / 'resume.pt', changed)
                with self.assertRaisesRegex(ValueError, 'FedAWA aggregation state'):
                    engine.run_experiment(config, output_dir=output, resume=output / 'resume.pt')

    def test_aggregation_policy_records_resume_and_transport_binding(self):
        from tefed.software_fixture import create_fixture
        from tefed.transport import shared_config_digest
        from tefed.worker import _variant_config
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = create_fixture(root / 'fixture')
            method = 'consistent_teacher_fedawa'
            config['experiment'].update(method=method, rounds=1)
            config['baselines'] = {method: {'pseudo_threshold': 0., 'teacher_decay': .8,
                'consistency_weight': .2, 'aggregation_temperature': .25, 'aggregation_cap': .8,
                'reference_weighting': 'sample_count'}}
            manifest_path = Path(config['experiment']['manifest'])
            manifest = json.loads(manifest_path.read_text())
            removed = next(row for row in manifest['records'] if row['client_id'] == 'client1' and row['split'] == 'labeled')
            manifest['records'].remove(removed)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with patch('tefed.baselines.adaptive_aggregation_weights', wraps=adaptive_aggregation_weights) as aggregate:
                result = engine.run_experiment(config, output_dir=root / 'run')
            call = aggregate.call_args
            self.assertEqual(len(call.args), 1)
            self.assertEqual(call.kwargs, {'sample_counts': [2, 1], 'temperature': .25, 'cap': .8})
            saved = json.loads((root / 'run' / 'configuration.json').read_text())
            self.assertEqual(saved['resolved_baseline_policies'], policy_schedule(method, config))
            row = result['rounds'][0]
            expected = adaptive_aggregation_weights(call.args[0], **call.kwargs)
            np.testing.assert_allclose(list(row['aggregation']['weights'].values()), expected)
            for client in row['clients'].values():
                self.assertEqual(client['baseline_policies'], saved['resolved_baseline_policies'])
                self.assertEqual(client['assessment_policy'], row['baseline_policy'])
                self.assertEqual(client['teacher_updates'], client['optimizer_steps'])
            worker = _variant_config(config, method, 0)
            self.assertEqual(shared_config_digest(worker), shared_config_digest({**config, 'model': {**config['model'], 'specialists': 0}}))
            changed = deepcopy(worker)
            changed['baselines'][method]['aggregation_cap'] = .9
            self.assertNotEqual(shared_config_digest(worker), shared_config_digest(changed))
            resumed = engine.run_experiment(config, output_dir=root / 'run', resume=root / 'run' / 'resume.pt')
            self.assertEqual(resumed['rounds'], result['rounds'])
            with self.assertRaisesRegex(ValueError, 'Resume protocol'):
                engine.run_experiment(changed, output_dir=root / 'run', resume=root / 'run' / 'resume.pt')


if __name__ == '__main__':
    unittest.main()
