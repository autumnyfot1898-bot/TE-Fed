"""CPU integration with synthetic data and rendering disabled."""
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from dataclasses import replace
import json
import math
import unittest

from tests.fixtures import create_fixture

try:
    import torch
    import ultralytics
    from tefed import engine
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    engine = None

RUNNER_AVAILABLE = TRAINING_AVAILABLE and callable(getattr(engine, "run_experiment", None))
PRETRAIN_AVAILABLE = TRAINING_AVAILABLE and callable(getattr(engine, "pretrain", None))


class FixtureTests(unittest.TestCase):
    def test_float32_router_probabilities_are_canonicalized_before_storage(self):
        import numpy as np
        from tefed.core import select_route
        probabilities = [float(np.float32(1 / 6 + 2e-8))] * 6
        self.assertGreater(abs(sum(probabilities) - 1), 1e-7)
        route = select_route(probabilities)
        self.assertAlmostEqual(sum(route["probabilities"]), 1, places=12)
        self.assertAlmostEqual(sum(route["weights"]), 1, places=12)
        self.assertEqual(route["route"], [0, 1, 2])

    def test_manifest_has_explicit_partitions_backgrounds_and_calibration_flag(self):
        from tefed.data import load_manifest
        with TemporaryDirectory() as directory:
            config = create_fixture(Path(directory))
            manifest = load_manifest(config["experiment"]["manifest"], strict_files=True)
            self.assertEqual(len(manifest.records), 14)
            self.assertEqual(len(manifest.calibration_ids), 4)
            for client in ("client0", "client1"):
                self.assertEqual(len(manifest.by_split("labeled", client)), 2)
                self.assertTrue(any(not r.labels for r in manifest.by_split("labeled", client)))
                self.assertEqual(len(manifest.by_split("unlabeled", client)), 1)
                self.assertEqual(len(manifest.by_split("validation", client)), 2)
                self.assertEqual(len(manifest.by_split("test", client)), 1)
                self.assertEqual(len(manifest.by_split("hard", client)), 1)
            calibration = json.loads(Path(config["experiment"]["calibration"]).read_text(encoding="utf-8"))
            self.assertEqual(calibration["data_origin"], "generated")
            self.assertTrue(config["experiment"]["test_mode"])
            self.assertFalse(config["renderer"]["enabled"])


@unittest.skipUnless(RUNNER_AVAILABLE, "Native CPU integration requires training dependencies and completed engine.run_experiment API")
class NativeEngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = create_fixture(cls.root / "fixture")
        cls.config['evaluation']['monitor_every'] = 1
        cls.result = engine.run_experiment(cls.config, seed=0, output_dir=cls.root / "full_run")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_two_rounds_log_actual_optimization_and_background_presentations(self):
        ledgers = self.result["rounds"]
        self.assertEqual([row["round_index"] for row in ledgers], [0, 1])
        saved = [json.loads(line) for line in (Path(self.result["output_dir"]) / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(saved), 2)
        for row in ledgers:
            self.assertEqual(set(row["clients"]), {"client0", "client1"})
            self.assertEqual(row["new_cases"], 0)
            self.assertEqual(row["retained_pool_size"], 0)
            for client, local in row["clients"].items():
                self.assertGreater(local["optimizer_steps"], 0)
                self.assertEqual(local["presentations"]["labeled"], 2)
                self.assertEqual(local["presentations"]["correction"], 0)
                self.assertTrue(math.isfinite(local["mean_step_loss"]))
                self.assertEqual(row["reuse_selected_ids"][client], [])
                self.assertTrue(all(t < row["round_index"] for t in row["selected_creation_rounds"][client]))

    def test_round_monitoring_is_read_only_and_matches_final_evaluation(self):
        output=Path(self.result['output_dir'])
        rows=[json.loads(line) for line in (output/'monitoring.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['round_index'] for row in rows],[0,1])
        self.assertEqual(rows[-1]['metrics'],self.result['metrics'])
        self.assertTrue(all(row['monitoring_seconds']>0 for row in self.result['rounds']))
        environment=json.loads((output/'environment.json').read_text(encoding='utf-8'))
        self.assertEqual(len(environment['calibration_sha256']),64)

    def test_checkpoint_model_bn_and_all_four_evaluation_files(self):
        checkpoint = Path(self.result["checkpoint"])
        self.assertTrue(checkpoint.is_file())
        model, router, payload = engine.load_checkpoint(checkpoint, device="cpu")
        self.assertEqual(payload["round_index"], 1)
        self.assertEqual(len(model.heads), 6)
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.assertTrue(torch.isfinite(value).all(), name)
        views = {"participating", "unseen", "new-client", "hard"}
        self.assertEqual(set(self.result["metrics"]), views)
        output_dir = Path(self.result["output_dir"])
        self.assertTrue((output_dir / "metrics.json").is_file())
        for view in views:
            path = output_dir / "predictions" / f"{view}.jsonl"
            self.assertTrue(path.is_file(), view)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["id"] for row in rows}), 2)
            self.assertIn("mAP50", self.result["metrics"][view])

    def test_cold_start_assessment_has_no_gradients_state_or_bn_update(self):
        from tefed.data import load_manifest, load_sample
        model, router, payload = engine.load_checkpoint(self.result["checkpoint"], device="cpu")
        model.train()
        router.train()
        model.zero_grad(set_to_none=True)
        router.zero_grad(set_to_none=True)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        router_before = {k: v.detach().clone() for k, v in router.state_dict().items()}
        manifest = load_manifest(self.config["experiment"]["manifest"])
        sample = load_sample(manifest.by_split("test")[0], size=64)
        calibration = json.loads(Path(self.config["experiment"]["calibration"]).read_text(encoding="utf-8"))
        state = deepcopy(calibration["state_prior"])
        original_state = deepcopy(state)
        assessment = engine.assess_frame(model, router, sample, state, calibration, self.config, round_id=2)
        self.assertIn("candidates", assessment)
        self.assertEqual(state, original_state)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in router.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for name, value in router.state_dict().items():
            self.assertTrue(torch.equal(value, router_before[name]), name)
        self.assertTrue(model.training)

    def test_explicit_view_identity_subsets_are_respected(self):
        from tefed.data import load_manifest
        model, router, payload = engine.load_checkpoint(self.result["checkpoint"], device="cpu")
        manifest = load_manifest(self.config["experiment"]["manifest"])
        standard_ids = [row.id for row in manifest.by_split("test")]
        hard_ids = [row.id for row in manifest.by_split("hard")]
        metadata = {**manifest.metadata, "require_equal_standard_views": False,
                    "views": {"Participating": standard_ids[:1], "Unseen": standard_ids[:1],
                              "New-client": standard_ids[1:], "Hard": hard_ids}}
        subset_manifest = replace(manifest, metadata=metadata)
        calibration = json.loads(Path(self.config["experiment"]["calibration"]).read_text(encoding="utf-8"))
        metrics = engine.evaluate_model(model, router, subset_manifest, payload["states"], calibration,
                                        self.config, self.root / "explicit_view_subsets")
        for view in ("participating", "unseen", "new-client"):
            self.assertEqual(metrics[view]["num_images"], 1, view)
        self.assertEqual(metrics["hard"]["num_images"], 2)

    @unittest.skipUnless(PRETRAIN_AVAILABLE, "engine.pretrain API is not available yet")
    def test_shared_initial_checkpoint_pretraining_records_actual_updates(self):
        config = deepcopy(self.config)
        config["experiment"]["rounds"] = 1
        result = engine.pretrain(config, seed=0, output_dir=self.root / "pretrain")
        checkpoint = Path(result["checkpoint"])
        self.assertTrue(checkpoint.is_file())
        model, _, _ = engine.load_checkpoint(checkpoint, device="cpu")
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertGreater(sum(local["optimizer_steps"] for row in result["rounds"] for local in row["clients"].values()), 0)


if __name__ == "__main__":
    unittest.main()
