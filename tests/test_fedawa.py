"""Check the FedAWA paper objective against full parameter-space optimization."""
import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from tefed.fedawa import flatten_parameters, paper_aggregation_weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, 'Install training dependencies for FedAWA tests')
class FedAWATests(unittest.TestCase):
    def test_matches_direct_equation_three_and_post_step_weights(self):
        reference = np.array([1., 2., -.5, 1.5])
        clients = np.array([[1.2, 1.7, -.6, 1.6], [.7, 2.4, -.4, 1.2], [1.1, 2., -.7, 1.9]])
        initial = [.2, -.1, .5]
        raw = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        parameters = torch.from_numpy(clients)
        broadcast = torch.from_numpy(reference)
        optimizer = torch.optim.Adam([raw], lr=.02, betas=(.5, .999))

        def objective():
            weights = raw.softmax(0)
            updates = parameters - broadcast
            spread = (weights * torch.linalg.vector_norm(updates - weights @ updates, dim=1)).sum()
            return spread + 1 - F.cosine_similarity(weights @ parameters, broadcast, dim=0)

        expected_initial = float(objective().detach())
        for _ in range(7):
            optimizer.zero_grad()
            objective().backward()
            optimizer.step()
        actual = paper_aggregation_weights(clients, reference, logits=initial, steps=7, learning_rate=.02)
        np.testing.assert_allclose(actual['weights'], raw.detach().softmax(0).numpy(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(actual['logits'], raw.detach().numpy(), rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(actual['initial_objective'], expected_initial, places=12)
        self.assertAlmostEqual(actual['objective'], float(objective().detach()), places=12)
        self.assertGreater(np.max(np.abs(np.array(actual['weights']) - torch.tensor(initial).softmax(0).numpy())), .001)

    def test_zero_updates_and_resumable_logits_are_finite(self):
        for reference in (np.zeros(4), np.ones(4)):
            with self.subTest(reference=reference.tolist()):
                result = paper_aggregation_weights([reference] * 3, reference, steps=3)
                np.testing.assert_allclose(result['weights'], [1 / 3] * 3, atol=1e-12)
                self.assertTrue(np.isfinite(result['objective']))
        clients = [[1., 2.], [2., 1.], [2., 3.]]
        first = paper_aggregation_weights(clients, [1., 1.], steps=2)
        continued = paper_aggregation_weights(clients, [1., 1.], logits=first['logits'], steps=2)
        repeated = paper_aggregation_weights(clients, [1., 1.], logits=list(first['logits']), steps=2)
        self.assertEqual(continued, repeated)
        self.assertNotEqual(continued['logits'], first['logits'])

    def test_client_order_equivariance_and_no_directional_cap(self):
        clients = np.array([[1., 1.], [4., -1.], [-2., 3.]])
        initial = np.array([2., 0., -1.])
        first = paper_aggregation_weights(clients, [1., 0.], logits=initial, steps=3)
        order = [2, 0, 1]
        second = paper_aggregation_weights(clients[order], [1., 0.], logits=initial[order], steps=3)
        np.testing.assert_allclose(np.array(first['weights'])[order], second['weights'], atol=1e-10)
        self.assertGreater(first['weights'][0], .5)

    def test_initial_probabilities_follow_sample_fractions(self):
        result = paper_aggregation_weights([[1., 0.]] * 3, [1., 0.], sample_counts=[2, 3, 5])
        np.testing.assert_allclose(result['weights'], [.2, .3, .5], atol=1e-10)
        self.assertEqual(result['initialization'], 'sample_fraction_probabilities')

    def test_all_parameters_and_no_running_buffers_are_flattened(self):
        state = {'backbone': {'trunk.weight': np.array([1., 2.]), 'trunk.running_mean': np.array([99.])},
                 'heads': {'0': {'detect.weight': np.array([3.]), 'detect.bias': np.array([4.])}}}
        names = {'trunk.weight', 'heads.0.detect.weight', 'heads.0.detect.bias'}
        np.testing.assert_array_equal(flatten_parameters(state, names), [4., 3., 1., 2.])
        with self.assertRaises(ValueError):
            flatten_parameters(state, {'missing'})

    def test_invalid_vectors_and_optimizer_settings_are_rejected(self):
        for clients, reference in (([], [0.]), ([[1.]], [0., 1.]), ([[float('nan')]], [0.]),
                                   ([[1.]], [float('inf')])):
            with self.subTest(clients=clients), self.assertRaises(ValueError):
                paper_aggregation_weights(clients, reference)
        for settings in ({'steps': 0}, {'steps': True}, {'learning_rate': 0},
                         {'learning_rate': float('nan')}, {'logits': [1.]},
                         {'sample_counts': [0., 0.]}, {'sample_counts': [-1., 2.]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                paper_aggregation_weights([[1.], [2.]], [0.], **settings)


if __name__ == '__main__':
    unittest.main()
