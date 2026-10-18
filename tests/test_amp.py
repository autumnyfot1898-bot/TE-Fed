"""AMP overflow recovery and nonfinite objective guards."""
from contextlib import ExitStack
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

try:
    import torch
    from tefed import engine
    from tefed.baselines import update_ema_teacher
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

CUDA_AVAILABLE = TRAINING_AVAILABLE and torch.cuda.is_available()


if TRAINING_AVAILABLE:
    class ScalarModel(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, device=device))
            self.heads = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, x, active_heads):
            return {head: self.weight * x for head in active_heads}


class ScalarCriterion:
    def __init__(self, objective):
        self.objective = objective

    def __call__(self, prediction, targets, weights, device):
        return self.objective(prediction), {}

    def epoch_update(self):
        pass


@unittest.skipUnless(TRAINING_AVAILABLE, "Requires training dependencies")
class AMPTrainingTests(unittest.TestCase):
    def test_initial_amp_scale_requires_a_finite_positive_number(self):
        for scale in [float("nan"), float("inf"), 0, -1, True, False]:
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "optimizer.amp_init_scale"):
                    engine.validate_run_config({"optimizer": {"amp_init_scale": scale}})
        engine.validate_run_config({"optimizer": {"amp_init_scale": 128.}})

    def run_local(self, model, *, count=1, amp=False, objective=None,
                  method="fedavg", teacher=None, reference=None, scaler_factory=None):
        objective = objective or (lambda value: value)
        config = {
            "experiment": {"rounds": 1, "local_epochs": 1, "batch_size": 1},
            "model": {},
            "optimizer": {"learning_rate": .1, "final_learning_rate": .1,
                          "warmup_rounds": 0, "momentum": 0., "weight_decay": 0.,
                          "gradient_clip": 10., "amp": amp},
            "loss": {"rho": 0., "gamma": 0.},
        }
        items = [{"kind": "labeled", "sample": {"targets": []},
                  "route_weights": {0: 1.}} for _ in range(count)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                engine, "WeightedNativeLoss",
                side_effect=lambda model, head: ScalarCriterion(objective)))
            stack.enter_context(patch.object(
                engine, "sample_tensor",
                side_effect=lambda sample, device, **kwargs: torch.ones((), device=device)))
            if scaler_factory is not None:
                stack.enter_context(patch.object(
                    engine.torch.amp, "GradScaler", side_effect=scaler_factory))
            return engine.train_local(model, items, config, method=method,
                                      teacher=teacher, reference=reference)

    def test_cpu_nonfinite_gradient_aborts_without_updating_parameters(self):
        model = ScalarModel("cpu")
        before = model.weight.detach().clone()
        hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite gradients"):
                self.run_local(model)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(model.weight, before))
        self.assertTrue(torch.isinf(model.weight.grad).all())

    def test_cpu_nan_loss_with_finite_derivative_aborts_before_backward(self):
        model = ScalarModel("cpu")
        before = model.weight.detach().clone()
        backward_calls = []
        hook = model.weight.register_hook(lambda grad: backward_calls.append(grad))
        try:
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite loss"):
                self.run_local(model, objective=lambda value: value + value.new_tensor(float("nan")))
        finally:
            hook.remove()
        self.assertEqual(backward_calls, [])
        self.assertIsNone(model.weight.grad)
        self.assertTrue(torch.equal(model.weight, before))

    def test_cpu_nan_penalty_aborts_before_penalty_backward(self):
        model = ScalarModel("cpu")
        before = model.weight.detach().clone()
        penalty = model.weight + model.weight.new_tensor(float("nan"))
        backward_calls = []
        hook = penalty.register_hook(lambda grad: backward_calls.append(grad))
        try:
            with patch("tefed.baselines.proximal_penalty", return_value=penalty):
                with self.assertRaisesRegex(FloatingPointError, "Nonfinite penalty"):
                    self.run_local(model, method="fedprox", reference=deepcopy(model))
        finally:
            hook.remove()
        self.assertEqual(backward_calls, [])
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertTrue(torch.equal(model.weight, before))

    @unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA")
    def test_cuda_overflow_skips_once_then_updates_parameters_and_teacher(self):
        model = ScalarModel("cuda")
        teacher = deepcopy(model)
        backward_calls = []

        def overflow_once(grad):
            backward_calls.append(True)
            return torch.full_like(grad, float("inf")) if len(backward_calls) == 1 else grad

        scaler_class = torch.amp.GradScaler
        scalers = []

        def make_scaler(device, **kwargs):
            kwargs["init_scale"] = 1.0
            scaler = scaler_class(device, **kwargs)
            scalers.append(scaler)
            return scaler

        hook = model.weight.register_hook(overflow_once)
        try:
            with patch("tefed.baselines.update_ema_teacher", wraps=update_ema_teacher) as ema:
                ledger = self.run_local(
                    model, count=2, amp=True, method="consistent_teacher_fedawa",
                    teacher=teacher, scaler_factory=make_scaler)
        finally:
            hook.remove()
        self.assertEqual(ledger["attempted_optimizer_steps"], 2)
        self.assertEqual(ledger["amp_skipped_steps"], 1)
        self.assertEqual(ledger["optimizer_steps"], 1)
        self.assertTrue(math.isfinite(ledger["mean_step_loss"]))
        self.assertEqual(len(scalers), 1)
        self.assertEqual(scalers[0].get_scale(), .5)
        self.assertEqual(ema.call_count, 1)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertAlmostEqual(model.weight.item(), .9, places=6)
        self.assertAlmostEqual(teacher.weight.item(), .9999, places=6)


if __name__ == "__main__":
    unittest.main()
