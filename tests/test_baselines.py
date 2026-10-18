import copy
import unittest

import numpy as np

from tefed.baselines import (
    METHOD_REGISTRY, adaptive_aggregation_weights, batchnorm_state_keys,
    elastic_penalty, fedsto_set_trainable, flatten_update_vector,
    fsslod_ema_decay, is_batchnorm_key, make_ema_teacher,
    orthogonality_penalty, policy_dict, proximal_penalty,
    sample_weighted_aggregate, update_ema_teacher,
)

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def state(value):
    return {
        "backbone": {"trunk.conv.weight": np.array([value, 2 * value], dtype=np.float32),
                     "trunk.bn.weight": np.array([value], dtype=np.float32),
                     "trunk.bn.bias": np.array([value], dtype=np.float32),
                     "trunk.bn.running_mean": np.array([value], dtype=np.float32),
                     "trunk.bn.running_var": np.array([value], dtype=np.float32)},
        "heads": {"0": {"linear.weight": np.array([[value]], dtype=np.float64)}},
    }


class NumpyBaselineTests(unittest.TestCase):
    def test_fedavg_matches_sample_weighted_result_and_does_not_alias(self):
        a, b = state(0), state(10)
        average = sample_weighted_aggregate([a, b], [1, 3])
        np.testing.assert_allclose(average["backbone"]["trunk.conv.weight"], [7.5, 15])
        self.assertEqual(average["backbone"]["trunk.conv.weight"].dtype, np.float32)
        self.assertEqual(average["heads"]["0"]["linear.weight"].dtype, np.float64)
        average["backbone"]["trunk.conv.weight"][0] = -99
        self.assertEqual(float(b["backbone"]["trunk.conv.weight"][0]), 10)
        reverse = sample_weighted_aggregate([b, a], [3, 1])
        self.assertEqual(float(reverse["heads"]["0"]["linear.weight"][0, 0]), 7.5)

    def test_fedbn_keeps_affine_and_running_statistics_from_local_previous(self):
        local = state(2)
        aggregate = sample_weighted_aggregate([state(0), state(10)], [1, 1], previous=local, keep_predicate=is_batchnorm_key)
        for key in ("trunk.bn.weight", "trunk.bn.bias", "trunk.bn.running_mean", "trunk.bn.running_var"):
            np.testing.assert_array_equal(aggregate["backbone"][key], [2])
        np.testing.assert_array_equal(aggregate["backbone"]["trunk.conv.weight"], [5, 10])
        self.assertTrue(is_batchnorm_key("heads.2.bn.running_var"))
        self.assertFalse(is_batchnorm_key("trunk.conv.weight"))

    def test_fedavg_counter_policy_is_explicit_and_stable(self):
        a, b = state(1), state(3)
        a["backbone"]["counter"] = np.array(4, dtype=np.int64)
        b["backbone"]["counter"] = np.array(9, dtype=np.int64)
        self.assertEqual(int(sample_weighted_aggregate([a, b], [1, 2])["backbone"]["counter"]), 9)
        self.assertEqual(int(sample_weighted_aggregate([a, b], [1, 1])["backbone"]["counter"]), 4)

    def test_fedavg_rejects_bad_state_and_zero_mass(self):
        a, b = state(1), state(3)
        with self.assertRaises(ValueError):
            sample_weighted_aggregate([a, b], [0, 0])
        with self.assertRaises(ValueError):
            sample_weighted_aggregate([a, b], [1, -1])
        bad = copy.deepcopy(b)
        bad["heads"]["1"] = {}
        with self.assertRaises(ValueError):
            sample_weighted_aggregate([a, bad], [1, 1])
        bad = copy.deepcopy(b)
        bad["backbone"]["trunk.conv.weight"][0] = np.nan
        with self.assertRaises(ValueError):
            sample_weighted_aggregate([a, bad], [1, 1])
        with self.assertRaises(ValueError):
            sample_weighted_aggregate([a, b], [1, 1], keep_predicate=is_batchnorm_key)

    def test_flatten_shared_updates_excludes_heads_and_sorts_keys(self):
        update = {"backbone": {"z": np.array([3.0]), "a": np.array([[1.0, 2.0]])}, "heads": {"1": {"w": np.array([100.0])}}}
        np.testing.assert_array_equal(flatten_update_vector(update), [1, 2, 3])
        self.assertEqual(flatten_update_vector(update, shared_only=False).shape, (4,))

    def test_adaptive_weights_respect_cap_and_client_permutation(self):
        vectors = np.array([[1, 0], [1, 0], [-1, 0]], dtype=float)
        gains = np.array([0.9, 0, -0.9])
        weights = adaptive_aggregation_weights(vectors, gains)
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertLessEqual(float(weights.max()), 0.5 + 1e-12)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        order = [2, 0, 1]
        permuted = adaptive_aggregation_weights(vectors[order], gains[order])
        np.testing.assert_allclose(permuted, weights[order])

    def test_adaptive_two_client_cap_zero_updates_and_extreme_values(self):
        np.testing.assert_allclose(adaptive_aggregation_weights([[1, 0], [-1, 0]], [1, -1]), [0.5, 0.5])
        np.testing.assert_allclose(adaptive_aggregation_weights(np.zeros((4, 2))), [0.25] * 4)
        values = adaptive_aggregation_weights([[1e308, 1e308], [1e308, 0], [-1e308, 0]], temperature=1e-300)
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values.sum()), 1)
        with self.assertRaises(ValueError):
            adaptive_aggregation_weights([[1, 0]])
        with self.assertRaises(ValueError):
            adaptive_aggregation_weights([[1, 0], [0, 1]], cap=0.4)
        with self.assertRaises(ValueError):
            adaptive_aggregation_weights([[1, 0], [0, 1]], [2, 0])

    def test_policy_schedules_and_method_switches(self):
        self.assertEqual(len(METHOD_REGISTRY), 11)
        self.assertAlmostEqual(fsslod_ema_decay(0, 0), 0.99)
        self.assertAlmostEqual(fsslod_ema_decay(29, 1), 0.999)
        self.assertLess(fsslod_ema_decay(12, 1), fsslod_ema_decay(13, 0))
        first, second = policy_dict("fedsto", 4, 0), policy_dict("fedsto", 4, 1)
        self.assertTrue(first["reset_teacher_each_round"])
        self.assertTrue(first["freeze_trunk"])
        self.assertFalse(second["freeze_trunk"])
        self.assertEqual(first["orthogonal_coefficient"], 0.05)
        self.assertEqual(policy_dict("confidence", 0, 0)["pseudo_threshold"], 0.75)
        for method in ["local", "fedavg", "fedprox", "fedbn"]:
            self.assertFalse(policy_dict(method, 0, 0)["use_unlabeled"])
        for method in ["confidence", "consistent_teacher_fedawa", "fedsto", "fsslod", "server_moe", "generic_hc", "tefed"]:
            self.assertTrue(policy_dict(method, 0, 0)["use_unlabeled"])
        self.assertEqual(policy_dict("fedprox", 0, 0)["proximal_mu"], 0.01)
        self.assertEqual(policy_dict("fsslod", 0, 0)["elastic_coefficient"], 0.1)
        self.assertEqual(policy_dict("local", 0, 0)["aggregation"], "none")
        self.assertTrue(policy_dict("fedbn", 0, 0)["preserve_client_batchnorm"])
        self.assertFalse(policy_dict("server_moe", 0, 0)["use_corrections"])
        self.assertTrue(policy_dict("generic_hc", 0, 0)["use_corrections"])
        self.assertFalse(policy_dict("generic_hc", 0, 0)["use_source_verification"])
        self.assertTrue(policy_dict("generic_hc", 0, 0)["use_state_matched_reuse"])
        self.assertTrue(policy_dict("tefed", 0, 0)["use_state_matched_reuse"])
        with self.assertRaises(ValueError):
            policy_dict("unknown", 0, 0)
        with self.assertRaises(ValueError):
            fsslod_ema_decay(30, 0)


@unittest.skipUnless(TORCH_AVAILABLE, "Install training dependencies for tensor baseline tests")
class TensorBaselineTests(unittest.TestCase):
    def test_bn_exact_keys_handle_numeric_module_names(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        keys = batchnorm_state_keys(model)
        self.assertEqual(keys, {"1.weight", "1.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked"})
        self.assertTrue(is_batchnorm_key("1.weight", known_keys=keys))
        self.assertFalse(is_batchnorm_key("0.weight", known_keys=keys))

    def test_ema_float_states_and_integer_buffers(self):
        student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).double()
        with torch.no_grad():
            for p in student.parameters():
                p.fill_(2)
            student[1].running_mean.fill_(2)
            student[1].num_batches_tracked.fill_(4)
        teacher = make_ema_teacher(student)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        with torch.no_grad():
            for p in student.parameters():
                p.fill_(6)
            student[1].running_mean.fill_(6)
            student[1].num_batches_tracked.fill_(12)
        update_ema_teacher(teacher, student, 0.75)
        self.assertTrue(torch.all(teacher[0].weight == 3))
        self.assertEqual(teacher[0].weight.dtype, torch.float64)
        self.assertTrue(torch.all(teacher[1].running_mean == 3))
        self.assertEqual(int(teacher[1].num_batches_tracked), 12)
        self.assertTrue(torch.all(student[0].weight == 6))
        update_ema_teacher(teacher, student, 1)
        self.assertTrue(torch.all(teacher[0].weight == 3))
        update_ema_teacher(teacher, student, 0)
        self.assertTrue(torch.all(teacher[0].weight == 6))

    def test_proximal_and_elastic_penalties_have_correct_gradients(self):
        model = nn.Linear(2, 1, bias=False)
        anchor = {"weight": torch.zeros_like(model.weight)}
        with torch.no_grad():
            model.weight.fill_(1)
        prox = proximal_penalty(model, anchor, 0.01)
        self.assertAlmostEqual(float(prox), 0.01, places=7)
        prox.backward()
        self.assertTrue(torch.allclose(model.weight.grad, torch.full_like(model.weight, 0.01)))
        model.zero_grad(set_to_none=True)
        elastic = elastic_penalty(model, anchor, 0.1)
        self.assertAlmostEqual(float(elastic), 0.05, places=7)
        elastic.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertAlmostEqual(float(proximal_penalty(model, copy.deepcopy(model))), 0)

    def test_orthogonality_distinguishes_independent_and_duplicate_filters(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        self.assertAlmostEqual(float(orthogonality_penalty(model)), 0, places=7)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 0], [1.0, 0]]))
        value = orthogonality_penalty(model)
        self.assertGreater(float(value), 0)
        value.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_fedsto_stage_freezes_shared_gradients_and_bn_statistics(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion = nn.Linear(2, 2)
                self.trunk = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.heads = nn.ModuleList([nn.Linear(2, 1), nn.Linear(2, 1)])
            def forward(self, x):
                return self.heads[0](self.trunk(self.fusion(x)))
        model = Tiny().train()
        fedsto_set_trainable(model, 0, [0])
        count = model.trunk[1].num_batches_tracked.clone()
        model(torch.randn(8, 2)).sum().backward()
        self.assertEqual(int(model.trunk[1].num_batches_tracked), int(count))
        self.assertTrue(all(p.grad is None for p in model.trunk.parameters()))
        self.assertIsNotNone(model.heads[0].weight.grad)
        self.assertFalse(model.heads[1].weight.requires_grad)
        model.zero_grad(set_to_none=True)
        fedsto_set_trainable(model, 1, [0])
        model(torch.randn(8, 2)).sum().backward()
        self.assertGreater(int(model.trunk[1].num_batches_tracked), int(count))
        self.assertIsNotNone(model.trunk[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
