"""Calibration fitting, data boundaries and missing-cell validation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import math
import unittest

import numpy as np

from tefed.calibration import (FAMILIES, build_calibration, condition_probe,
                               fit_temperature)
from tefed.data import Manifest, load_sample, validate_manifest


class CalibrationFixture(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = []
        for index in range(2):
            uid = f"v{index}"
            Image.fromarray(np.full((16, 16, 3), [80, 120, 160], dtype=np.uint8)).save(self.root / f"{uid}.png")
            np.save(self.root / f"{uid}.npy", np.full((16, 16), 2 + index * 2, dtype=np.float32), allow_pickle=False)
            rows.append({"id": uid, "client_id": "c", "split": "val", "rgb": f"{uid}.png",
                         "depth": f"{uid}.npy", "labels": [{"box": [.25, .25, .75, .75], "class_id": 0}]})
        # Nonexistent paths detect accidental test-stream access.
        rows.extend([{"id": "test", "client_id": "c", "split": "test", "rgb": "never_open_test.png", "labels": []},
                     {"id": "unlabeled", "client_id": "c", "split": "unlabeled", "rgb": "never_open_unlabeled.png"}])
        self.manifest = validate_manifest(rows, calibration_ids=["v0", "v1"], base_dir=self.root,
                                          metadata={"class_names": ["ripe", "non-ripe"]})
        self.seen = []

    def tearDown(self):
        self.temp.cleanup()

    def posterior(self, sample):
        self.seen.append(("posterior", sample["id"]))
        # Match geometry independently of predicted class.
        return [{"box": target["box"], "posterior": [.8, .2], "class_id": 1, "score": .9}
                for target in sample["targets"]]

    def loss(self, sample):
        self.seen.append(("loss", sample["id"]))
        probe = sample["meta"].get("calibration_probe")
        if probe is None:
            return 2.
        delta = (FAMILIES.index(probe["family"]) + 1) * (probe["level"] + 1) / 100
        return 2 * (1 + delta)

    def build(self, **kwargs):
        return build_calibration(self.manifest, self.loss, self.posterior,
                                 image_size=16, strict_count=None, **kwargs)

    def test_all_twenty_measured_cells_and_320_lookup_expansion_inputs(self):
        artifact = self.build(seed=5)
        self.assertEqual(set(artifact["loss_lookup"]["base"]), set(FAMILIES))
        self.assertEqual(artifact["protocol"]["perturbed_loss_evaluations"], 120)
        self.assertEqual(len(artifact["observations"]), 120)
        self.assertEqual(artifact["protocol"]["perturbation_seeds"], [5, 6, 7])
        values = [i * j for i in range(1, 6) for j in range(1, 5)]
        p95 = float(np.quantile(values, .95))
        self.assertAlmostEqual(artifact["loss_lookup"]["base"]["illum"][0], 1 / p95)
        self.assertEqual(artifact["loss_lookup"]["base"]["sens"][3], 1)
        for family in FAMILIES:
            self.assertEqual(len(artifact["cells"][family]), 4)
            self.assertTrue(all(cell["n"] == 6 for cell in artifact["cells"][family]))
        self.assertEqual({uid for _, uid in self.seen}, {"v0", "v1"})
        self.assertEqual(len([x for x in self.seen if x[0] == "loss"]), 122)

    def test_measured_medians_and_identifiable_height_depth_prior(self):
        artifact = self.build()
        self.assertAlmostEqual(artifact["luminance_median"], (.299 * 80 + .587 * 120 + .114 * 160) / 255)
        self.assertEqual(artifact["depth_validation_median"], 3.)
        prior = artifact["class_priors"]["0"]
        self.assertAlmostEqual(prior["a_c"], math.log(.5))
        self.assertAlmostEqual(prior["b_c"], 0, places=10)
        self.assertAlmostEqual(prior["r_c_med"], 1)
        self.assertAlmostEqual(prior["area_depth"]["intercept"], math.log(.25))
        absent = artifact["class_priors"]["1"]
        self.assertIsNone(absent["a_c"])
        self.assertIsNone(absent["r_c_med"])

    def test_missing_depth_is_none_not_historical_coefficient_fallback(self):
        rows = []
        for row in self.manifest.records:
            value = row.as_dict()
            value.pop("depth", None)
            rows.append(value)
        self.manifest = validate_manifest(rows, calibration_ids=["v0", "v1"], metadata={"class_names": ["ripe", "non-ripe"]})
        artifact = self.build()
        self.assertIsNone(artifact["depth_validation_median"])
        self.assertIsNone(artifact["class_priors"]["0"]["a_c"])
        self.assertEqual(artifact["class_priors"]["0"]["height_depth_status"], "insufficient_depth_boxes")

    def test_strict_fifteen_frames_and_explicit_validation_ids(self):
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            build_calibration(self.manifest, self.loss, self.posterior, image_size=16)
        invalid = Manifest(self.manifest.records, ("test",), self.manifest.metadata)
        with self.assertRaisesRegex(ValueError, "validation record"):
            build_calibration(invalid, self.loss, self.posterior, strict_count=None)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            build_calibration(Manifest(self.manifest.records), self.loss, self.posterior, strict_count=None)

    def test_no_geometry_match_does_not_fabricate_temperature(self):
        with self.assertRaisesRegex(ValueError, "No validation detections"):
            build_calibration(self.manifest, self.loss, lambda sample: [], image_size=16, strict_count=None)

    def test_zero_nominal_loss_and_all_zero_cells_are_explicit(self):
        def zero_loss(sample):
            return 0.
        artifact = build_calibration(self.manifest, zero_loss, self.posterior, image_size=16, strict_count=None)
        self.assertTrue(artifact["protocol"]["all_cells_zero"])
        self.assertEqual(artifact["protocol"]["normalization_denominator"], 1e-6)
        self.assertEqual(artifact["protocol"]["zero_nominal_loss_ids"], ["v0", "v1"])
        self.assertTrue(all(value == 0 for row in artifact["loss_lookup"]["base"].values() for value in row))
        def positive_perturbation(sample):
            return .1 if "calibration_probe" in sample["meta"] else 0.
        artifact = build_calibration(self.manifest, positive_perturbation, self.posterior, image_size=16, strict_count=None)
        self.assertTrue(all(math.isfinite(row["relative_increase"]) for row in artifact["observations"]))
        self.assertAlmostEqual(artifact["observations"][0]["relative_increase"], 100000.)

    def test_invalid_cell_loss_aborts_without_imputation(self):
        def invalid(sample):
            probe = sample["meta"].get("calibration_probe")
            return math.nan if probe and probe["family"] == "ctx" else 1.
        with self.assertRaisesRegex(ValueError, "finite numeric scalar"):
            build_calibration(self.manifest, invalid, self.posterior, image_size=16, strict_count=None)

    def test_condition_probes_are_deterministic_and_preserve_geometry_and_targets(self):
        sample = load_sample(self.manifest.records[0], size=16)
        targets = copy.deepcopy(sample["targets"])
        for family in FAMILIES:
            a = condition_probe(sample, family, 3, 5)
            b = condition_probe(sample, family, 3, 5)
            self.assertEqual(a["targets"], targets)
            self.assertEqual(a["rgb"].shape, sample["rgb"].shape)
            np.testing.assert_array_equal(a["rgb"], b["rgb"])
            self.assertFalse(np.array_equal(a["rgb"], sample["rgb"]), family)
        self.assertEqual(sample["targets"], targets)
        self.assertNotIn("calibration_probe", sample["meta"])

    def test_progress_contains_measured_completion_counts(self):
        events = []
        self.build(progress=events.append)
        self.assertEqual(events[-1]["stage"], "condition_loss")
        self.assertEqual(events[-1]["completed"], 120)
        self.assertEqual(events[-1]["total"], 120)


class TemperatureTests(unittest.TestCase):
    def test_nll_temperature_matches_hand_computable_calibration(self):
        # With 80% class 0, calibration should shift p from .9 to .8.
        logits = [[math.log(.9), math.log(.1)]] * 10
        fit = fit_temperature(logits, [0] * 8 + [1] * 2)
        self.assertAlmostEqual(fit["T_C"], math.log(9) / math.log(4), places=6)
        self.assertLess(fit["nll_after"], fit["nll_before"])

    def test_uniform_or_absent_logits_rejected_as_unidentifiable(self):
        with self.assertRaisesRegex(ValueError, "unidentifiable"):
            fit_temperature([[0., 0.], [0., 0.]], [0, 1])
        with self.assertRaises(ValueError):
            fit_temperature([], [])


if __name__ == "__main__":
    unittest.main()
