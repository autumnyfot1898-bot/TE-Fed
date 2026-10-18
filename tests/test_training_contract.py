"""Numerical and native checks for the local training objective."""
from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from tefed.baselines import fedsto_set_trainable
    from tefed.engine import (_crop_embeddings, inherit_initial, load_checkpoint,
                              save_checkpoint, train_local)
    from tefed.losses import contrastive_alignment, soft_consistency
    from tefed.model import RoutedYOLO, Router, sample_tensor
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def predictions(boxes, probabilities, *, gradients=False, score=.8):
    return {'boxes': torch.tensor([boxes], dtype=torch.float32, requires_grad=gradients),
            'logits': torch.tensor([[np.log(p).tolist() for p in probabilities]],
                                   dtype=torch.float32, requires_grad=gradients),
            'scores': torch.full((1, len(boxes)), score)}


@unittest.skipUnless(AVAILABLE, 'Requires native training dependencies')
class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        cls.config = {'model': {'architecture': 'yolo26n.yaml', 'classes': 2,
                               'specialists': 0, 'contrastive_dim': 16}}
        cls.model = RoutedYOLO(cls.config)

    def test_consistency_matches_closed_form_kl_and_giou(self):
        weak = predictions([[0, 0, 1, 1]], [[.8, .2]], gradients=True)
        strong = predictions([[.1, 0, 1, 1]], [[.6, .4]], gradients=True)
        expected = .5 * (.8 * math.log(.8 / .6) + .2 * math.log(.2 / .4) + .1)
        loss = soft_consistency(weak, strong)
        self.assertAlmostEqual(float(loss), expected, places=6)
        loss.backward()
        self.assertIsNone(weak['logits'].grad)
        self.assertIsNone(weak['boxes'].grad)
        for key in ('logits', 'boxes'):
            self.assertTrue(torch.isfinite(strong[key].grad).all())
            self.assertGreater(float(strong[key].grad.abs().sum()), 0)

    def test_consistency_associates_different_anchors_one_to_one(self):
        a, b = [.05, .05, .45, .45], [.55, .55, .95, .95]
        weak = predictions([a, a, b], [[.8, .2], [.8, .2], [.1, .9]])
        strong = predictions([b, a], [[.1, .9], [.8, .2]], gradients=True)
        total, count = soft_consistency(weak, strong, return_details=True)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(total), 0., places=6)

    def test_consistency_requires_class_iou_and_explicit_confidence(self):
        weak = predictions([[0, 0, .4, .4]], [[.8, .2]], score=.2)
        cases = [predictions([[0, 0, .4, .4]], [[.2, .8]], gradients=True, score=.2),
                 predictions([[.8, .8, 1, 1]], [[.8, .2]], gradients=True, score=.2)]
        for strong in cases:
            total, count = soft_consistency(weak, strong, return_details=True)
            self.assertEqual(count, 0)
            total.backward()
            self.assertEqual(float(strong['boxes'].grad.abs().sum()), 0.)
        strong = predictions([[.02, 0, .4, .4]], [[.6, .4]], gradients=True, score=.2)
        self.assertEqual(soft_consistency(weak, strong, return_details=True)[1], 1)
        self.assertEqual(soft_consistency(weak, strong, confidence=.75, return_details=True)[1], 0)

    def test_consistency_uses_total_pairs_across_images(self):
        weak = {'boxes': torch.tensor([[[0., 0, 1, 1], [0, 0, 0, 0]],
                                       [[0., 0, 1, 1], [0, 0, 1, 1]]]),
                'logits': torch.log(torch.tensor([[[.8, .2], [.8, .2]],
                                                   [[.8, .2], [.8, .2]]])),
                'scores': torch.full((2, 2), .8)}
        strong = {'boxes': torch.tensor([[[.1, 0, 1, 1], [0, 0, 0, 0]],
                                         [[.2, 0, 1, 1], [.2, 0, 1, 1]]], requires_grad=True),
                  'logits': torch.log(torch.tensor([[[.6, .4], [.6, .4]],
                                                     [[.7, .3], [.7, .3]]])),
                  'scores': torch.full((2, 2), .8)}
        first = .5 * (.8 * math.log(.8 / .6) + .2 * math.log(.2 / .4) + .1)
        second = .5 * (.8 * math.log(.8 / .7) + .2 * math.log(.2 / .3) + .2)
        self.assertAlmostEqual(float(soft_consistency(weak, strong)), (first + 2 * second) / 3, places=6)

    def test_infonce_uses_all_ordered_pairs_and_destination_availability(self):
        views = [torch.tensor([[1., 0], [1, 0], [1, 0]], requires_grad=True) for _ in range(3)]
        available = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=torch.bool)
        loss = contrastive_alignment(views, availability=available)
        self.assertAlmostEqual(float(loss), (6 * math.log(2) + 4 * math.log(3)) / 10, places=6)
        loss.backward()
        self.assertTrue(all(torch.isfinite(view.grad).all() for view in views))

    def test_infonce_has_cross_candidate_negatives_and_empty_pair_zero(self):
        views = [torch.eye(2, requires_grad=True), torch.eye(2, requires_grad=True)]
        loss = contrastive_alignment(views)
        self.assertAlmostEqual(float(loss), math.log1p(math.exp(-10)), places=6)
        loss.backward()
        self.assertGreater(float(views[0].grad.abs().sum()), 0.)
        self.assertEqual(float(contrastive_alignment([view[:1] for view in views])), 0.)
        self.assertEqual(float(contrastive_alignment(views, availability=torch.zeros(2, 2, dtype=torch.bool))), 0.)

    def test_roi_align_preserves_subpixel_boxes_and_projection_gradients(self):
        model = deepcopy(self.model)
        channels = model.contrastive_projection.in_features
        with torch.no_grad():
            model.contrastive_projection.weight.zero_(); model.contrastive_projection.bias.zero_()
            model.contrastive_projection.weight[0, 0] = 1.
        feature = torch.zeros(1, channels, 4, 4)
        feature[0, 0] = torch.arange(4).repeat(4, 1)
        feature.requires_grad_()
        projected = model.project_regions(feature, torch.tensor([[0, .20, .20, .70, .70],
                                                                  [0, .22, .20, .72, .70]]))
        self.assertAlmostEqual(float(projected[0, 0]), 1.3, places=5)
        self.assertAlmostEqual(float(projected[1, 0] - projected[0, 0]), .08, places=5)
        projected.square().sum().backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.)
        self.assertGreater(float(model.contrastive_projection.weight.grad.abs().sum()), 0.)

    def test_modality_presence_includes_zero_sensors_and_respects_flags(self):
        model = deepcopy(self.model).eval()
        target = [{'box': [.1, .1, .8, .8], 'class_id': 0}]
        first = {'rgb': np.zeros((64, 64, 3), np.uint8), 'depth': np.zeros((64, 64), np.float32),
                 'nir': np.zeros((64, 64), np.uint8), 'targets': target, 'enabled_modalities': ['RGB', 'D', 'NIR']}
        second = {**first, 'enabled_modalities': ['RGB']}
        with torch.no_grad():
            views, available = _crop_embeddings(model, [sample_tensor(first), sample_tensor(second)], [first, second])
        self.assertIsNotNone(views)
        self.assertEqual(available.tolist(), [[True, True, True], [True, False, False]])
        self.assertTrue(torch.equal(views[1][1], torch.zeros_like(views[1][1])))
        self.assertIsNone(_crop_embeddings(model, [sample_tensor(first)], [first])[0])
        self.assertIsNone(_crop_embeddings(model, [sample_tensor(second)] * 2, [second] * 2)[0])

    def test_contrastive_bn_statistics_and_modes_are_preserved(self):
        model = deepcopy(self.model).train()
        normalizers = [layer for layer in model.trunk.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
        normalizers[0].eval()
        before = [(layer, layer.training, layer.running_mean.clone(), layer.running_var.clone()) for layer in normalizers]
        model._contrastive_features(torch.rand(2, 5, 64, 64)).square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNotNone(model.fusion.residual.weight.grad)
        for layer, training, mean, variance in before:
            self.assertEqual(layer.training, training)
            torch.testing.assert_close(layer.running_mean, mean)
            torch.testing.assert_close(layer.running_var, variance)
        with patch.object(model, 'features', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                model._contrastive_features(torch.zeros(2, 5, 64, 64))
        self.assertEqual([layer.training for layer in normalizers], [entry[1] for entry in before])
        model.eval()
        model._contrastive_features(torch.zeros(2, 5, 64, 64))
        self.assertTrue(all(not layer.training for layer in normalizers))

    def test_native_decoder_and_dfl_have_box_gradients(self):
        model = deepcopy(self.model)
        native = model(torch.rand(1, 5, 64, 64), (0,))[0]
        decoded = model.decode_head(native, 0, (64, 64))
        self.assertEqual(decoded['boxes'].shape[-1], 4)
        (decoded['boxes'].square().sum() + decoded['logits'].square().mean()).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.trunk.parameters()))
        head = model.heads[0].detect
        head.reg_max = 2
        distances = torch.tensor([[[0.], [1.], [1.], [0.], [0.], [1.], [1.], [0.]]], requires_grad=True)
        raw = {'scores': torch.zeros(1, 2, 1), 'boxes': distances, 'feats': [torch.zeros(1, 1, 1, 1)]}
        head.stride = torch.tensor([8.])
        box = model.decode_head(raw, 0, (32, 32))['boxes']
        box.sum().backward()
        self.assertGreater(float(distances.grad.abs().sum()), 0.)

    def test_projection_is_shared_frozen_and_checkpointed(self):
        model = deepcopy(self.model)
        parameters = sum(p.numel() for p in model.contrastive_projection.parameters())
        self.assertEqual(model.capacity()['contrastive_parameters'], parameters)
        expected = sum(p.numel() for module in (model.trunk, model.fusion, model.contrastive_projection) for p in module.parameters())
        self.assertEqual(model.capacity()['shared_parameters'], expected)
        state = model.export_state()
        self.assertIn('contrastive_projection.weight', state['backbone'])
        model.import_state(state)
        fedsto_set_trainable(model, 0)
        self.assertTrue(all(not p.requires_grad for p in model.contrastive_projection.parameters()))
        fedsto_set_trainable(model, 1)
        self.assertTrue(all(p.requires_grad for p in model.contrastive_projection.parameters()))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'current.pt'
            save_checkpoint(checkpoint, model, Router(), self.config, {}, [], 0)
            loaded, _, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload['model_schema_version'], 2)
            torch.testing.assert_close(loaded.contrastive_projection.weight, model.contrastive_projection.weight)
            target = deepcopy(model)
            with torch.no_grad():
                target.contrastive_projection.weight.zero_()
            inherit_initial(target, checkpoint)
            torch.testing.assert_close(target.contrastive_projection.weight, model.contrastive_projection.weight)
            payload.pop('model_schema_version'); torch.save(payload, checkpoint)
            for operation in (lambda: load_checkpoint(checkpoint), lambda: inherit_initial(target, checkpoint)):
                with self.assertRaisesRegex(ValueError, 'regenerate'):
                    operation()
            payload['model_schema_version'] = 2
            payload['model'].pop('contrastive_projection.weight'); torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, 'projection'):
                inherit_initial(target, checkpoint)

    def test_native_minibatch_exercises_both_auxiliary_losses(self):
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(31); np.random.seed(31)
                config = deepcopy(self.config)
                config.update(experiment={'rounds': 1, 'local_epochs': 1, 'batch_size': 2},
                              optimizer={'learning_rate': .001, 'momentum': 0., 'warmup_rounds': 0,
                                         'amp': device.startswith('cuda'), 'amp_init_scale': 128.},
                              loss={'rho': .1, 'gamma': .5, 'temperature': .1},
                              evaluation={'prediction_confidence': 0., 'max_detections': 20})
                model = RoutedYOLO(config).to(device)
                items = []
                for index in range(2):
                    rng = np.random.default_rng(index)
                    sample = {'rgb': rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                              'depth': rng.random((64, 64), dtype=np.float32),
                              'nir': rng.integers(0, 256, (64, 64), dtype=np.uint8),
                              'enabled_modalities': ['RGB', 'D', 'NIR'],
                              'targets': [{'box': [.15, .15, .75, .8], 'class_id': index}]}
                    items.append({'kind': 'pseudo', 'sample': sample,
                                  'candidates': [{'R': .8, 'route': [0], 'weights': [1.]}]})
                before = model.contrastive_projection.weight.detach().clone()
                ledger = train_local(model, items, config)
                self.assertEqual(ledger['contrastive_batches'], 1)
                self.assertGreater(ledger['mean_contrastive_loss'], 0.)
                self.assertGreater(ledger['consistency_pairs'], 0)
                self.assertGreater(ledger['mean_consistency_loss'], 0.)
                self.assertEqual(ledger['optimizer_steps'], 1)
                self.assertFalse(torch.equal(before, model.contrastive_projection.weight))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(float(model.fusion.residual.weight.grad.abs().sum()), 0.)
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.trunk.parameters()))

    def test_disabled_or_empty_alignment_does_not_update_projection(self):
        for mode in ('rho_zero', 'rgb_only', 'one_candidate', 'missing_auxiliary'):
            with self.subTest(mode=mode):
                config = deepcopy(self.config)
                config.update(experiment={'rounds': 1, 'local_epochs': 1, 'batch_size': 2},
                              optimizer={'amp': False, 'weight_decay': .1}, loss={'rho': 0. if mode == 'rho_zero' else .1, 'gamma': 0.})
                if mode == 'rgb_only':
                    config['model']['detector_rgb_only'] = True
                model = RoutedYOLO(config)
                count = 1 if mode == 'one_candidate' else 2
                items = []
                for index in range(count):
                    sample = {'rgb': np.full((64, 64, 3), 80 + index, np.uint8),
                              'depth': np.ones((64, 64), np.float32), 'nir': np.ones((64, 64), np.uint8),
                              'enabled_modalities': ['RGB'] if mode == 'missing_auxiliary' else ['RGB', 'D', 'NIR'],
                              'targets': [{'box': [.2, .2, .8, .8], 'class_id': index % 2}]}
                    items.append({'kind': 'labeled', 'sample': sample, 'route_weights': {0: 1.}})
                before = deepcopy(model.contrastive_projection.state_dict())
                ledger = train_local(model, items, config, method='fedavg')
                self.assertEqual(ledger['contrastive_batches'], 0)
                self.assertTrue(all(p.grad is None for p in model.contrastive_projection.parameters()))
                for name, value in model.contrastive_projection.state_dict().items():
                    torch.testing.assert_close(value, before[name])


if __name__ == '__main__':
    unittest.main()
