"""Data boundaries, synchronized geometry and detection metrics."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import subprocess
import sys
import unittest

import numpy as np

from tefed.data import (apply_unseen, letterbox_parameters, load_manifest, load_sample,
                        map_boxes, oriented_to_horizontal, validate_manifest,
                        validate_view_identities)
from tefed.metrics import aggregate_seed_metrics, box_iou, evaluate_detection, evaluate_jsonl


def target(box=(.1, .2, .5, .8), cls=0):
    return {"box": list(box), "class_id": cls}


def prediction(box=(.1, .2, .5, .8), cls=0, score=.9):
    return {**target(box, cls), "score": score}


def record(uid="a", split="labeled", client="c1", **kwargs):
    return {"id": uid, "client_id": client, "split": split, "rgb": f"{uid}.png",
            "labels": [target()], **kwargs}


class ManifestTests(unittest.TestCase):
    def test_duplicate_uid_and_rgb_cannot_cross_splits_or_clients(self):
        with self.assertRaisesRegex(ValueError, "Duplicate identity"):
            validate_manifest([record(), record(split="test", client="c2")])
        with self.assertRaisesRegex(ValueError, "RGB asset"):
            validate_manifest([record(), record(uid="other", split="hard", rgb="a.png")])

    def test_validation_only_calibration_and_test_view_reuse(self):
        m = validate_manifest([record("v", "val"), record("t", "test"), record("h", "hard")],
                              calibration_ids=["v"])
        self.assertEqual(m.by_split("validation")[0].id, "v")
        validate_view_identities(m, {"Participating": ["t"], "Unseen": ["t"],
                                    "New-client": ["t"], "Hard": ["h"]}, require_equal_standard_views=True)
        with self.assertRaisesRegex(ValueError, "Calibration"):
            validate_manifest([record("t", "test")], calibration_ids=["t"])
        with self.assertRaisesRegex(ValueError, "must be in split test"):
            validate_view_identities(m, {"Unseen": ["v"]})
        with self.assertRaisesRegex(ValueError, "must be in split hard"):
            validate_view_identities(m, {"Hard": ["t"]})

    def test_equal_standard_views_is_explicit_and_never_randomized(self):
        m = validate_manifest([record("t1", "test"), record("t2", "test")])
        views = {"Participating": ["t1"], "New-client": ["t2"]}
        validate_view_identities(m, views)
        with self.assertRaisesRegex(ValueError, "identical test identities"):
            validate_view_identities(m, views, require_equal_standard_views=True)

    def test_required_labels_and_valid_coordinates(self):
        row = record()
        row.pop("labels")
        with self.assertRaisesRegex(ValueError, "explicit labels"):
            validate_manifest([row])
        row["split"] = "unlabel"
        self.assertEqual(validate_manifest([row]).records[0].labels, ())
        for bad in ([0, 0, 0, 1], [0, 0, 2, 1], [0, 0, math.nan, 1]):
            with self.assertRaises(ValueError):
                validate_manifest([record(labels=[target(bad)])])
        with self.assertRaises(ValueError):
            validate_manifest([record(labels=[target(cls=True)])])
        with self.assertRaisesRegex(ValueError, "outside class_names"):
            validate_manifest([record(labels=[target(cls=1)])], metadata={"class_names": ["class0"]})

    def test_relative_paths_and_missing_file(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps({"records": [record()]}), encoding="utf-8")
            m = load_manifest(p)
            self.assertEqual(m.records[0].rgb, Path(d) / "a.png")
            with self.assertRaises(FileNotFoundError):
                load_manifest(p, strict_files=True)

    def test_unlabeled_serialization_drops_annotations(self):
        m = validate_manifest([record(split="unlabeled")])
        self.assertNotIn("labels", m.as_dict()["records"][0])

    def test_manifest_cli_roundtrip_preserves_explicit_partition_and_drops_unlabeled_labels(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_manifest.py"
        with TemporaryDirectory() as d:
            src, dst = Path(d) / "raw.json", Path(d) / "out.json"
            src.write_text(json.dumps({"records": [record("v", "validation"), record("u", "unlabeled")],
                                       "calibration_ids": ["v"]}), encoding="utf-8")
            run = subprocess.run([sys.executable, str(script), "--input", str(src), "--output", str(dst)],
                                 capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            normalized = json.loads(dst.read_text(encoding="utf-8"))
            self.assertEqual(normalized["calibration_ids"], ["v"])
            self.assertNotIn("labels", normalized["records"][1])
            self.assertEqual(load_manifest(dst).by_split("unlabeled")[0].labels, ())


class GeometryTests(unittest.TestCase):
    def test_letterbox_forward_inverse_respects_rounding(self):
        tr = letterbox_parameters((101, 51), (64, 64))
        boxes = [[0, 0, 1, 1], [.1, .2, .4, .8]]
        forward = map_boxes(boxes, tr)
        inverse = map_boxes(forward, tr.as_dict(), inverse=True)
        np.testing.assert_allclose(inverse, boxes, atol=1e-6)
        self.assertEqual(map_boxes([], tr).shape, (0, 4))

    def test_synchronized_pixels_and_targets_and_unlabeled_firewall(self):
        from PIL import Image
        with TemporaryDirectory() as d:
            root = Path(d)
            Image.fromarray(np.full((50, 100, 3), [10, 20, 30], dtype=np.uint8)).save(root / "rgb.png")
            Image.fromarray(np.full((50, 100), 1500, dtype=np.uint16)).save(root / "depth.png")
            Image.fromarray(np.full((50, 100), 50, dtype=np.uint8)).save(root / "nir.png")
            m = validate_manifest([record(rgb="rgb.png", depth="depth.png", nir="nir.png")], base_dir=root)
            s = load_sample(m.records[0], size=(100, 100), require_modalities=("depth", "nir"))
            self.assertEqual(s["rgb"].dtype, np.uint8)
            self.assertEqual(s["depth"].dtype, np.float32)
            self.assertEqual(s["rgb"][25, 0].tolist(), [10, 20, 30])
            self.assertEqual(s["rgb"][0, 0].tolist(), [114, 114, 114])
            np.testing.assert_array_equal(s["depth"][25:75], 1500)
            np.testing.assert_array_equal(s["nir"][25:75], 50)
            self.assertEqual(float(s["depth"][:25].sum()), 0)
            np.testing.assert_allclose(s["targets"][0]["box"], [.1, .35, .5, .65], atol=1e-6)
            unlabeled = validate_manifest([record(split="unlabeled", rgb="rgb.png")], base_dir=root)
            un = load_sample(unlabeled.records[0], include_labels=True)
            self.assertTrue(un["ignore_labels"])
            self.assertEqual(un["targets"], [])

    def test_unregistered_modality_fails(self):
        from PIL import Image
        with TemporaryDirectory() as d:
            root = Path(d)
            Image.new("RGB", (100, 50)).save(root / "rgb.png")
            Image.new("L", (50, 50)).save(root / "nir.png")
            r = validate_manifest([record(rgb="rgb.png", nir="nir.png")], base_dir=root).records[0]
            with self.assertRaisesRegex(ValueError, "not registered"):
                load_sample(r)

    def test_renderer_npy_depth_and_unit_float_nir(self):
        from PIL import Image
        with TemporaryDirectory() as d:
            root = Path(d)
            Image.new("RGB", (10, 8)).save(root / "rgb.png")
            np.save(root / "depth.npy", np.full((8, 10), 1.25, dtype=np.float32), allow_pickle=False)
            np.save(root / "nir.npy", np.full((8, 10), .5, dtype=np.float32), allow_pickle=False)
            r = validate_manifest([record(rgb="rgb.png", depth="depth.npy", nir="nir.npy")], base_dir=root).records[0]
            sample = load_sample(r, size=(10, 8))
            np.testing.assert_array_equal(sample["depth"], 1.25)
            np.testing.assert_array_equal(sample["nir"], 128)
            np.save(root / "nir.npy", np.full((8, 10), 2., dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "NIR must"):
                load_sample(r)

    def test_oriented_conversion_is_explicit_xy_corners(self):
        quad = [10, 20, 30, 10, 50, 40, 20, 60]
        self.assertEqual(oriented_to_horizontal(quad), [10, 10, 50, 60])
        np.testing.assert_allclose(oriented_to_horizontal(quad, image_size=(100, 100), normalize=True), [.1, .1, .5, .6])
        with self.assertRaises(ValueError):
            oriented_to_horizontal([10, 10, 5, 5, .2])

    def test_fixed_unseen_noise_and_shared_geometry(self):
        sample = {"rgb": np.full((10, 10, 3), 100, dtype=np.uint8),
                  "depth": np.full((10, 10), 1000, dtype=np.float32),
                  "nir": np.full((10, 10), 40, dtype=np.uint8),
                  "targets": [target((.1, .2, .3, .4))], "meta": {}}
        cfg = {"depth_noise_std": 4., "nir_noise_std": 2., "rgb_brightness": .5,
               "translate_x": .1, "horizontal_flip": True}
        a = apply_unseen(sample, identity="a", perturbation=cfg, seed=7)
        b = apply_unseen(sample, identity="a", perturbation=cfg, seed=7)
        c = apply_unseen(sample, identity="b", perturbation=cfg, seed=7)
        np.testing.assert_array_equal(a["depth"], b["depth"])
        self.assertFalse(np.array_equal(a["depth"], c["depth"]))
        np.testing.assert_allclose(a["targets"][0]["box"], [.8, .2, 1., .4])
        self.assertTrue(np.all(a["depth"][:, 0] == 0))
        np.testing.assert_array_equal(sample["rgb"], 100)  # Input was not mutated.
        with self.assertRaises(ValueError):
            apply_unseen(sample, identity="a", perturbation={"unknown": 1})


class DetectionMetricTests(unittest.TestCase):
    def test_perfect_and_absent_predictions(self):
        gt = {"a": [target()], "b": [target(cls=1)]}
        r = evaluate_detection(gt, {"a": [prediction()], "b": [prediction(cls=1)]})
        self.assertAlmostEqual(r["mAP50_95"], 1)
        self.assertEqual(r["classes_with_ground_truth"], 2)
        self.assertEqual(evaluate_detection(gt, {"a": [], "b": []})["mAP50"], 0)

    def test_recall_grid_is_101_points_and_image_matches_are_independent(self):
        gt = {"a": [target()], "b": [target()]}
        r = evaluate_detection(gt, {"a": [prediction()], "b": []})
        self.assertAlmostEqual(r["AP50"], 51 / 101)
        self.assertEqual(r["per_class"]["0"]["counts_by_iou"]["0.50"], {"tp": 1, "fp": 0, "fn": 1})

    def test_false_positive_before_true_positive_reduces_ap(self):
        p = [prediction((.6, .6, .9, .9), score=.99), prediction(score=.5)]
        self.assertAlmostEqual(evaluate_detection({"a": [target()]}, {"a": p})["AP50"], .5)

    def test_duplicate_detection_does_not_match_twice(self):
        gt = {"a": [target(), target((.6, .6, .9, .9))]}
        pred = {"a": [prediction(score=.9), prediction(score=.8),
                       prediction((.6, .6, .9, .9), score=.7)]}
        r = evaluate_detection(gt, pred)
        self.assertAlmostEqual(r["AP50"], (51 + 50 * 2 / 3) / 101)
        self.assertEqual(r["per_class"]["0"]["counts_by_iou"]["0.50"]["fp"], 1)

    def test_equal_iou_gt_tie_uses_later_gt_like_coco_greedy_loop(self):
        # Equal-IoU first prediction; second prediction matches only GT0.
        gt = {"a": [target((0, 0, .6, 1)), target((.4, 0, 1, 1))]}
        pred = {"a": [prediction((0, 0, 1, 1), score=.9), prediction((0, 0, .6, 1), score=.8)]}
        self.assertEqual(evaluate_detection(gt, pred)["AP50"], 1)

    def test_class_matching_not_shared_across_labels(self):
        r = evaluate_detection({"a": [target(cls=0)]}, {"a": [prediction(cls=1)]})
        self.assertEqual(r["AP50"], 0)
        self.assertIsNone(r["per_class"]["1"]["AP50"])
        self.assertEqual(r["per_class"]["1"]["counts_by_iou"]["0.50"]["fp"], 1)

    def test_iou_thresholds_and_tiny_boxes(self):
        r = evaluate_detection({"a": [target((0, 0, 1, 1))]},
                               {"a": [prediction((0, 0, .6, 1))]})
        self.assertEqual(r["AP50"], 1)
        self.assertEqual(r["AP75"], 0)
        self.assertAlmostEqual(r["mAP50_95"], .3)
        np.testing.assert_allclose(box_iou([0, 0, 1e-10, 1e-10], [[0, 0, 1e-10, 1e-10]]), [1])

    def test_zero_gt_is_undefined_never_perfect(self):
        r = evaluate_detection({"a": []}, {"a": [prediction()]})
        self.assertIsNone(r["AP50"])
        self.assertIsNone(r["mAP50_95"])
        self.assertEqual(r["num_ground_truth"], 0)
        self.assertEqual(r["per_class"]["0"]["counts_by_iou"]["0.50"]["fp"], 1)
        empty = evaluate_detection({}, {})
        self.assertIsNone(empty["AP50"])

    def test_per_class_max_detections_and_strict_ids(self):
        gt = {"a": [target()]}
        pred = {"a": [prediction((.6, .6, .9, .9), score=.9), prediction(score=.8)]}
        self.assertEqual(evaluate_detection(gt, pred, max_detections=1)["AP50"], 0)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            evaluate_detection(gt, {})
        self.assertEqual(evaluate_detection(gt, {}, strict_ids=False)["AP50"], 0)
        with self.assertRaises(ValueError):
            evaluate_detection(gt, {"a": [], "outside": []})

    def test_jsonl_requires_all_identity_rows_and_rejects_duplicates(self):
        m = validate_manifest([record("a", "test"), record("b", "test")])
        with TemporaryDirectory() as d:
            p = Path(d) / "predictions.jsonl"
            rows = [{"id": "a", "predictions": [prediction()]}, {"id": "b", "predictions": []}]
            p.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            self.assertAlmostEqual(evaluate_jsonl(m, p)["mAP50"], 51 / 101)
            p.write_text(json.dumps(rows[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                evaluate_jsonl(m, p)
            p.write_text("\n".join(json.dumps(rows[0]) for _ in range(2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                evaluate_jsonl(m, p)


class SeedSummaryTests(unittest.TestCase):
    def test_sample_sd_and_exact_seed_set(self):
        summary = aggregate_seed_metrics({0: {"AP50": .5}, 1: {"AP50": .7}}, expected_seeds=[0, 1])
        self.assertAlmostEqual(summary["metrics"]["AP50"]["mean"], .6)
        self.assertAlmostEqual(summary["metrics"]["AP50"]["sample_sd"], math.sqrt(.02))
        with self.assertRaisesRegex(ValueError, "missing"):
            aggregate_seed_metrics({0: {"AP50": .5}}, expected_seeds=[0, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            aggregate_seed_metrics([{"seed": 0, "metrics": {"AP50": .5}}, {"seed": "0", "metrics": {"AP50": .6}}], expected_seeds=[0, 1])

    def test_missing_metrics_and_invalid_sd_inputs_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_seed_metrics({0: {"a": 1}, 1: {"b": 2}}, expected_seeds=[0, 1])
        with self.assertRaises(ValueError):
            aggregate_seed_metrics({0: {"a": 1}, 1: {"a": None}}, expected_seeds=[0, 1])
        with self.assertRaises(ValueError):
            aggregate_seed_metrics({0: {"a": 1}}, expected_seeds=[0])


if __name__ == "__main__":
    unittest.main()
