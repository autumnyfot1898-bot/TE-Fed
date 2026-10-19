import unittest
import numpy as np

try:
    import torch
    from tefed.model import RoutedYOLO, Router
    from tefed.losses import WeightedNativeLoss, target_batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, 'Install training dependencies for model tests')
class ModelLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.model = RoutedYOLO({'model': {'architecture':'yolo26n.yaml','classes':2,'specialist_width':0.5}})

    def test_routes_and_shared_backbone(self):
        self.model.eval()
        predictions = self.model.predict_heads(torch.rand(1,5,64,64), (0,1,3), confidence=0)
        self.assertEqual(set(predictions), {0,1,3})
        for outputs in predictions.values():
            for p in outputs[0]:
                self.assertAlmostEqual(sum(p['posterior']),1,places=6)
        cap = self.model.capacity()
        self.assertEqual(len(cap['head_parameters']),6)
        self.assertGreater(cap['shared_parameters'],0)

    def test_weighted_loss_does_not_normalize_reliability_away(self):
        self.model.train()
        x = torch.rand(1,5,64,64)
        outputs = self.model(x,(0,))[0]
        targets = [{'box':[.15,.2,.7,.8],'class_id':0}, {'box':[.7,.1,.9,.3],'class_id':1}]
        native = self.model.make_criterion(0)
        expected = native(outputs,target_batch(targets,'cpu'))[0].sum()
        weighted = WeightedNativeLoss(self.model,0)
        one,_ = weighted(outputs,targets,[1,1],'cpu')
        half,_ = weighted(outputs,targets,[.5,.5],'cpu')
        zero,_ = weighted(outputs,targets,[0,0],'cpu')
        self.assertTrue(torch.allclose(expected,one,rtol=1e-5,atol=1e-5))
        self.assertTrue(torch.allclose(one*.5,half,rtol=1e-5,atol=1e-5))
        self.assertEqual(float(zero),0)
        self.model.zero_grad(set_to_none=True)
        half.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in self.model.heads[2].parameters()))

    def test_state_roundtrip_and_router(self):
        state = self.model.export_state()
        self.model.import_state(state)
        second = self.model.export_state()
        for k in state['backbone']:
            np.testing.assert_array_equal(state['backbone'][k],second['backbone'][k])
        router = Router()
        probs = router(torch.zeros(2,17))
        self.assertTrue(torch.allclose(probs.sum(-1),torch.ones(2)))

    def test_prediction_branch_matches_native_without_training_branch(self):
        self.model.eval()
        x = torch.rand(1, 5, 64, 64)
        with torch.no_grad():
            features = self.model.features(x)
            for index in (0, 1):
                head = self.model.heads[index]
                expected = self.model.decode_head(head(features), index, x.shape[-2:], branch='one2one')
                calls = []
                handle = head.detect.cv2[0].register_forward_hook(lambda *args: calls.append('one2many'))
                try:
                    actual = self.model.decode_head(head.predict_raw(features), index, x.shape[-2:])
                finally:
                    handle.remove()
                self.assertEqual(calls, [])
                for field in ('boxes', 'logits', 'scores'):
                    torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)

    def test_capacity_counts_executed_parameter_subset(self):
        cap = self.model.capacity((0, 1, 3))
        self.assertEqual(cap['total_parameters'], cap['shared_parameters'] + sum(cap['head_parameters']))
        self.assertEqual(cap['prediction_shared_parameters'],
                         cap['shared_parameters'] - cap['contrastive_parameters'])
        self.assertEqual(cap['active_prediction_parameters'], cap['prediction_shared_parameters'] +
                         sum(cap['prediction_head_parameters'][i] for i in (0, 1, 3)))
        self.assertLess(cap['active_prediction_parameters'], cap['all_heads_prediction_parameters'])
        self.assertLess(cap['all_heads_prediction_parameters'], cap['total_parameters'])
        self.assertEqual(cap['active_heads'], [0, 1, 3])
        for indices in ((), (0, 0), (-1,), (6,), (True,), ('0',)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.model.capacity(indices)


if __name__ == '__main__':
    unittest.main()
