"""Inputs for software smoke tests."""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np


def create_fixture(root: Path) -> dict:
    """Create two 64px client fixtures with empty-image cases and absolute paths."""
    from PIL import Image

    root = Path(root).resolve()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records, calibration_ids, standard_ids, hard_ids = [], [], [], []
    width = height = 64
    yy, xx = np.mgrid[:height, :width]
    for client in range(2):
        cid = f"client{client}"
        entries = [("labeled", 0, False), ("labeled", 1, True), ("unlabeled", 0, False),
                   ("validation", 0, False), ("validation", 1, True),
                   ("test", 0, client == 1), ("hard", 0, False)]
        for split, index, negative in entries:
            uid = f"{cid}:{split}:{index}"
            stem = uid.replace(":", "_")
            # Keep background variance in empty images.
            rgb = np.stack([35 + xx // 4, 45 + yy // 4, 55 + (xx + yy) // 8], axis=-1).astype(np.uint8)
            depth = (2.0 + client * .25 + xx / 512).astype(np.float32)
            nir = (40 + (xx + yy) // 3).astype(np.uint8)
            box = [.25, .25, .625, .75]
            labels = []
            if not negative:
                left, top, right, bottom = (16, 16, 40, 48)
                rgb[top:bottom, left:right] = [200, 45, 35] if client == 0 else [45, 175, 50]
                depth[top:bottom, left:right] = 1.25 + client * .1
                nir[top:bottom, left:right] = 180
                labels = [{"box": box, "class_id": client}]
            rgb_path, depth_path, nir_path = (image_dir / f"{stem}{suffix}" for suffix in ("_rgb.png", "_depth.npy", "_nir.png"))
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            Image.fromarray(nir).save(nir_path)
            row = {"id": uid, "client_id": cid, "split": split, "rgb": str(rgb_path),
                   "depth": str(depth_path), "nir": str(nir_path), "fixture_only": True}
            if split != "unlabeled":
                row["labels"] = labels
            records.append(row)
            if split == "validation":
                calibration_ids.append(uid)
            elif split == "test":
                standard_ids.append(uid)
            elif split == "hard":
                hard_ids.append(uid)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"data_origin": "software_fixture", "class_names": ["ripe", "non-ripe"],
        "calibration_ids": calibration_ids, "require_equal_standard_views": True,
        "views": {"Participating": standard_ids, "Unseen": standard_ids,
                  "New-client": standard_ids, "Hard": hard_ids}, "records": records}, indent=2), encoding="utf-8")
    calibration = {"data_origin": "software_fixture", "artifact_version": "fixture-v1",
        "implementation": "tefed-software-check", "calibration_ids": calibration_ids,
        "T_C": 1., "luminance_median": .35, "depth_validation_median": 2.,
        "class_priors": {str(cls): {"a_c": math.log(.5), "b_c": -.5, "r_c_med": .75,
                                   "n_boxes": 1, "n_depth_boxes": 1, "height_depth_status": "test_fixture_only"}
                         for cls in range(2)},
        "loss_lookup": {"base": {family: [0., .2, .5, .8] for family in ("illum", "occ", "ctx", "mat", "sens")}},
        "state_prior": {"q": .5, "p": .5, "h": [.2] * 5}}
    calibration_path = root / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return {
        "test_mode": True,
        "experiment": {"method": "tefed", "seeds": [0], "rounds": 2, "local_epochs": 1,
                       "batch_size": 4, "manifest": str(manifest_path), "calibration": str(calibration_path),
                       "output": str(root / "runs"), "device": "cpu", "test_mode": True,
                       "task": "tomato"},
        "model": {"architecture": "yolo26n.yaml", "classes": 2, "specialists": 5,
                  "specialist_width": .5, "image_size": 64, "weights": "",
                  "head_loss_policy": "route_weighted"},
        "optimizer": {"learning_rate": .001, "final_learning_rate": .0001, "warmup_rounds": 0,
                      "momentum": .9, "weight_decay": .0005, "gradient_clip": 10., "amp": False},
        "loss": {"alpha": 1., "beta": .5, "rho": 0., "gamma": 0., "temperature": .1},
        "decision": {"reliability_coefficients": [.22, .24, .16, .21, .24, .08],
                     "delta_d": .35, "delta_c": .5, "delta_g": .5, "tau_cov": .8, "tau_cls": .05,
                     "tau_e": .65, "verification_geometry_threshold": .6, "tau_r": .65,
                     "epsilon": 1e-8, "reuse_cap": 6, "reuse_gate": .55, "pattern_cap": 2,
                     "source_cap": 1, "tau_state": .5, "tau_s": 1., "lambda_s": .2,
                     "allocation_tau": .5, "allocation_chi": .2, "routing_temperature": .5,
                     "priority_coefficients": [.15, .15, .25, .30, .25], "render_budget": 0},
        "router": {"learning_rate": .001, "steps": 0, "load_balance": .01},
        "renderer": {"enabled": False, "blender": "blender", "scene": "", "timeout": 30, "samples": 1},
        "evaluation": {"views": ["participating", "unseen", "new-client", "hard"],
                       "prediction_confidence": 0., "max_detections": 5, "unseen_brightness": .65},
    }
