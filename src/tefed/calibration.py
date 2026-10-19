"""Measure validation priors and loss lookups with frozen model callbacks."""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence
import json
import math
import numbers

import numpy as np

from .data import Manifest, load_sample, parse_annotation
from .metrics import box_iou


FAMILIES = ("illum", "occ", "ctx", "mat", "sens")
SEVERITIES = (.125, .375, .625, .875)
RELATIVE_LOSS_EPSILON = 1e-6


def _copy_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    for key in ("rgb", "depth", "nir"):
        value = sample.get(key)
        out[key] = None if value is None else np.asarray(value).copy()
    out["targets"] = [parse_annotation(x).as_dict() for x in sample["targets"]]
    out["meta"] = dict(sample.get("meta", {}))
    return out


def _scalar(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    value = float(value)
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _content_slice(sample: Mapping[str, Any]):
    meta = sample["meta"]
    px, py = meta["pad"]
    width, height = meta["resized_size"]
    return slice(py, py + height), slice(px, px + width)


def _object_mask(sample: Mapping[str, Any]) -> np.ndarray:
    height, width = sample["rgb"].shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for target in sample["targets"]:
        x1, y1, x2, y2 = target["box"]
        left, top = max(0, int(math.floor(x1 * width))), max(0, int(math.floor(y1 * height)))
        right, bottom = min(width, int(math.ceil(x2 * width))), min(height, int(math.ceil(y2 * height)))
        mask[top:bottom, left:right] = True
    return mask


def condition_probe(sample: Mapping[str, Any], family: str, level: int, seed: int, *, task: str = "tomato") -> dict[str, Any]:
    """Apply a fixed image-space probe without changing labels or box geometry.

    Levels 0..3 use bin midpoints .125/.375/.625/.875. The mat family changes
    appearance only; occlusion retains complete-object boxes."""
    if family not in FAMILIES or isinstance(level, bool) or level not in range(4):
        raise ValueError("A condition probe requires a known family and integer level 0..3")
    if task not in {"tomato", "dronevehicle"}:
        raise ValueError("task must be tomato or dronevehicle")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("perturbation seed must be a uint32 integer")
    out = _copy_sample(sample)
    uid = str(sample["id"])
    token = f"tefed-calibration-v1|{uid}|{family}|{level}|{seed}".encode()
    rng = np.random.default_rng(int.from_bytes(sha256(token).digest()[:8], "little"))
    severity = SEVERITIES[level]
    rgb = out["rgb"].astype(np.float32)
    valid = np.zeros(rgb.shape[:2], dtype=bool)
    valid[_content_slice(sample)] = True
    foreground = _object_mask(sample) & valid
    if family == "illum":
        factor = (.90, .75, .60, .45)[level]
        rgb[valid] *= factor
        params = {"rgb_brightness_factor": factor, "nir_unchanged": True}
    elif family == "occ":
        fraction = (.08, .16, .24, .32)[level]
        height, width = rgb.shape[:2]
        covered = np.zeros((height, width), dtype=bool)
        for target in sample["targets"]:
            x1, y1, x2, y2 = target["box"]
            left, top = int(math.floor(x1 * width)), int(math.floor(y1 * height))
            right, bottom = int(math.ceil(x2 * width)), int(math.ceil(y2 * height))
            bw, bh = max(1, right - left), max(1, bottom - top)
            rw = min(bw, max(1, int(round(bw * math.sqrt(fraction)))))
            rh = min(bh, max(1, int(round(bh * math.sqrt(fraction)))))
            ox = left + int(rng.integers(0, max(1, bw - rw + 1)))
            oy = top + int(rng.integers(0, max(1, bh - rh + 1)))
            covered[max(0, oy):min(height, oy + rh), max(0, ox):min(width, ox + rw)] = True
        covered &= valid
        rgb[covered] = 114
        for modality in ("depth", "nir"):
            if out[modality] is not None:
                out[modality][covered] = 0
        params = {"requested_box_area_fraction": fraction, "actual_image_covered_fraction": float(covered.mean()),
                  "fill_rgb": 114, "masked_depth_nir": True, "complete_object_targets_preserved": True}
    elif family == "ctx":
        alpha = (.10, .20, .30, .40)[level]
        background = valid & ~foreground
        color = rng.uniform(32, 224, size=3)
        rgb[background] = (1 - alpha) * rgb[background] + alpha * color
        params = {"background_color_mix": alpha, "background_color": color.tolist(),
                  "object_boxes_preserved": True, "background_available_pixels": int(background.sum())}
    elif family == "mat":
        alpha = (.10, .20, .30, .40)[level]
        gray = rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114
        rgb[foreground] = (1 - alpha) * rgb[foreground] + alpha * gray[foreground, None]
        params = {"object_desaturation": alpha, "semantic_labels_preserved": True,
                  "appearance_attribute": "maturity_appearance" if task == "tomato" else "class_boundary_appearance",
                  "class_transition": "none"}
    else:
        sigma = (.02, .04, .06, .08)[level]
        dropout = (.02, .04, .06, .08)[level]
        rgb[valid] += rng.normal(0, 255 * sigma, size=(int(valid.sum()), 3))
        if out["nir"] is not None:
            nir = out["nir"].astype(np.float32)
            nir[valid] += rng.normal(0, 255 * sigma, size=int(valid.sum()))
            out["nir"] = np.clip(nir, 0, 255).astype(np.uint8)
        if out["depth"] is not None:
            depth = out["depth"].copy()
            dvalid = valid & (depth > 0)
            if dvalid.any():
                unit = float(np.median(depth[dvalid]))
                depth[dvalid] += rng.normal(0, sigma * unit, size=int(dvalid.sum()))
                depth[(rng.random(depth.shape) < dropout) & dvalid] = 0
                depth = np.maximum(depth, 0).astype(np.float32)
            out["depth"] = depth
        params = {"rgb_nir_noise_std_fraction": sigma, "depth_noise_std_fraction_of_frame_median": sigma,
                  "depth_dropout_probability": dropout}
    out["rgb"] = np.clip(rgb, 0, 255).astype(np.uint8)
    out["meta"]["calibration_probe"] = {"family": family, "level": int(level), "severity": severity,
                                           "seed": seed, "task": task, "parameters": params}
    return out


def _matched_logits(sample: Mapping[str, Any], detections: Any, class_count: int | None):
    if not isinstance(detections, (list, tuple)):
        raise ValueError("posterior_evaluator must return a list of detection objects")
    parsed = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping) or "box" not in detection or "posterior" not in detection:
            raise ValueError("Each calibration detection needs box and posterior")
        box = parse_annotation({"box": detection["box"], "class_id": 0}).box
        p = np.asarray(detection["posterior"], dtype=np.float64)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or abs(float(p.sum()) - 1) > 1e-5:
            raise ValueError("Calibration posterior must be a finite normalized class probability vector")
        if class_count is not None and len(p) != class_count:
            raise ValueError("Class posterior dimension changed during calibration")
        class_count = len(p)
        if "native_logits" in detection:
            logits = np.asarray(detection["native_logits"], dtype=np.float64)
            if logits.shape != p.shape or not np.isfinite(logits).all():
                raise ValueError("native_logits must match the finite posterior dimension")
            kind = "native_logits"
        else:
            logits = np.log(np.maximum(p, 1e-12))
            kind = "log_posterior"
        score = _scalar(detection.get("score", float(p.max())), "detection score", nonnegative=True)
        parsed.append((score, index, box, logits, kind))
    parsed.sort(key=lambda row: (-row[0], row[1]))
    targets = [parse_annotation(x) for x in sample["targets"]]
    if class_count is not None and any(t.class_id >= class_count for t in targets):
        raise ValueError("GT class ID exceeds the calibration posterior dimension")
    matched = np.zeros(len(targets), dtype=bool)
    rows = []
    for _, index, box, logits, kind in parsed:
        overlaps = box_iou(box, [t.box for t in targets])
        if not len(overlaps):
            break
        overlaps[matched] = -1
        best = len(overlaps) - 1 - int(np.argmax(overlaps[::-1]))
        if overlaps[best] >= .5 - 1e-12:
            matched[best] = True
            rows.append((logits, targets[best].class_id, kind))
    return rows, class_count


def fit_temperature(logits: Sequence[Sequence[float]], classes: Sequence[int], *,
                    bounds: tuple[float, float] = (.05, 10.)) -> dict[str, Any]:
    """Fit positive temperature by matched-box NLL using grid and golden search.

    Uniform logits are unidentifiable and raise an error."""
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(classes)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("Temperature fitting needs at least one finite matched logit vector")
    if labels.ndim != 1 or len(labels) != len(scores) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("Temperature classes must be valid integer indices")
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] or not all(math.isfinite(x) for x in bounds):
        raise ValueError("Temperature bounds must be finite, positive and increasing")
    def nll(log_t):
        scaled = scores / math.exp(float(log_t))
        max_z = scaled.max(axis=1)
        return float(np.mean(max_z + np.log(np.exp(scaled - max_z[:, None]).sum(axis=1)) - scaled[np.arange(len(scores)), labels]))
    grid = np.linspace(math.log(bounds[0]), math.log(bounds[1]), 257)
    values = np.array([nll(x) for x in grid])
    if float(values.max() - values.min()) < 1e-12:
        raise ValueError("Temperature is unidentifiable from uniform/flat matched posteriors; use a trained checkpoint")
    index = int(values.argmin())
    left, right = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
    golden = (math.sqrt(5) - 1) / 2
    c, d = right - golden * (right - left), left + golden * (right - left)
    fc, fd = nll(c), nll(d)
    for _ in range(100):
        if right - left < 1e-10:
            break
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - golden * (right - left)
            fc = nll(c)
        else:
            left, c, fc = c, d, fd
            d = left + golden * (right - left)
            fd = nll(d)
    choices = [(float(grid[index]), float(values[index])), ((left + right) / 2, nll((left + right) / 2))]
    log_t, optimum = min(choices, key=lambda x: x[1])
    temperature = math.exp(log_t)
    return {"T_C": temperature, "nll_before": nll(0.), "nll_after": optimum,
            "matched_boxes": len(scores), "class_count": scores.shape[1],
            "temperature_bounds": list(bounds), "at_lower_bound": abs(temperature - bounds[0]) < 1e-6,
            "at_upper_bound": abs(temperature - bounds[1]) < 1e-6,
            "optimizer": "257-point log-temperature grid plus golden-section refinement"}


def _line_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) < 2:
        return {"intercept": None, "slope": None, "status": "insufficient_depth_boxes"}
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=np.float64)])
    if np.linalg.matrix_rank(design) < 2:
        return {"intercept": None, "slope": None, "status": "constant_depth_not_identifiable"}
    coefficient, _, _, _ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    if not np.isfinite(coefficient).all():
        raise ValueError("Nonfinite class-depth regression")
    return {"intercept": float(coefficient[0]), "slope": float(coefficient[1]), "status": "fitted"}


def build_calibration(
    manifest: Manifest, loss_evaluator: Callable[[Mapping[str, Any]], float],
    posterior_evaluator: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]], *,
    image_size: int | tuple[int, int] = 640, seed: int = 0, task: str = "tomato",
    strict_count: int | None = 15, perturbation_seeds: Sequence[int] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build calibration from designated validation identities and a frozen model.

    Default: 15 frames, five families, four levels, three perturbation seeds.
    strict_count=None records a smaller study. Callbacks receive data.load_sample
    output; the caller fixes the checkpoint, weights, gradients, and BN state."""
    if not isinstance(manifest, Manifest) or not manifest.calibration_ids:
        raise ValueError("A manifest with explicit nonempty calibration_ids is required")
    if len(set(manifest.calibration_ids)) != len(manifest.calibration_ids):
        raise ValueError("Calibration identities must be unique")
    if strict_count is not None and (isinstance(strict_count, bool) or not isinstance(strict_count, int) or strict_count < 1):
        raise ValueError("strict_count must be positive integer or None")
    if strict_count is not None and len(manifest.calibration_ids) != strict_count:
        raise ValueError(f"Calibration protocol requires exactly {strict_count} identities")
    if task not in {"tomato", "dronevehicle"}:
        raise ValueError("task must be tomato or dronevehicle")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    seeds = list(perturbation_seeds if perturbation_seeds is not None else [(seed + n) % 2**32 for n in range(3)])
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Exactly three distinct uint32 perturbation seeds are required")
    ids = list(manifest.calibration_ids)
    records = manifest.by_id
    if len(records) != len(manifest.records):
        raise ValueError("Manifest contains duplicate identities")
    if any(uid not in records or records[uid].split != "validation" for uid in ids):
        raise ValueError("Every calibration ID must refer to a validation record")
    samples = []
    nominal_losses = {}
    matched = []
    posterior_count = None
    gray_pixels, depth_pixels = [], []
    for index, uid in enumerate(ids):
        sample = load_sample(records[uid], size=image_size)
        samples.append(sample)
        rgb = sample["rgb"][_content_slice(sample)].astype(np.float64) / 255
        gray_pixels.append((.299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]).ravel())
        if sample["depth"] is not None:
            depth = sample["depth"][_content_slice(sample)]
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if len(valid):
                depth_pixels.append(valid.ravel())
        nominal_losses[uid] = _scalar(loss_evaluator(_copy_sample(sample)), f"nominal loss {uid}", nonnegative=True)
        rows, posterior_count = _matched_logits(sample, posterior_evaluator(_copy_sample(sample)), posterior_count)
        matched.extend(rows)
        if progress:
            progress({"stage": "nominal", "completed": index + 1, "total": len(ids), "id": uid})
    if not matched:
        raise ValueError("No validation detections match GT at IoU>=0.50; train/select a checkpoint before calibration")
    if manifest.metadata.get("class_names") and posterior_count != len(manifest.metadata["class_names"]):
        raise ValueError("Calibration posterior dimension must equal manifest class_names")
    temperature = fit_temperature([x[0] for x in matched], [x[1] for x in matched])
    luminance_median = float(np.median(np.concatenate(gray_pixels)))
    depth_median = float(np.median(np.concatenate(depth_pixels))) if depth_pixels else None
    observed_classes = {target["class_id"] for sample in samples for target in sample["targets"]}
    if manifest.metadata.get("class_names"):
        observed_classes |= set(range(len(manifest.metadata["class_names"])))
    class_priors = {}
    for cls in sorted(observed_classes):
        aspects, depths, heights, areas = [], [], [], []
        n_boxes = 0
        for sample in samples:
            height, width = sample["rgb"].shape[:2]
            for target in sample["targets"]:
                if target["class_id"] != cls:
                    continue
                n_boxes += 1
                x1, y1, x2, y2 = target["box"]
                aspects.append((x2 - x1) * width / ((y2 - y1) * height))
                if depth_median is None or sample["depth"] is None:
                    continue
                crop = sample["depth"][max(0, int(math.floor(y1 * height))):min(height, int(math.ceil(y2 * height))),
                                       max(0, int(math.floor(x1 * width))):min(width, int(math.ceil(x2 * width)))]
                valid = crop[np.isfinite(crop) & (crop > 0)]
                if not crop.size or len(valid) / crop.size < .35:
                    continue
                depths.append(math.log(max(float(np.median(valid)) / depth_median, 1e-3)))
                heights.append(math.log(max(y2 - y1, 1e-12)))
                areas.append(math.log(max((x2 - x1) * (y2 - y1), 1e-12)))
        height_fit, area_fit = _line_fit(depths, heights), _line_fit(depths, areas)
        class_priors[str(cls)] = {"a_c": height_fit["intercept"], "b_c": height_fit["slope"],
                                  "r_c_med": float(np.median(aspects)) if aspects else None,
                                  "n_boxes": n_boxes, "n_depth_boxes": len(depths),
                                  "height_depth_status": height_fit["status"], "area_depth": area_fit}
    cell_values = {family: [[] for _ in range(4)] for family in FAMILIES}
    raw_observations = []
    total = len(samples) * len(FAMILIES) * 4 * len(seeds)
    completed = 0
    for sample in samples:
        for family in FAMILIES:
            for level in range(4):
                for perturb_seed in seeds:
                    probe = condition_probe(sample, family, level, perturb_seed, task=task)
                    loss = _scalar(loss_evaluator(probe), f"perturbed loss {sample['id']}/{family}/{level}/{perturb_seed}", nonnegative=True)
                    nominal = nominal_losses[sample["id"]]
                    delta = max(0., (loss - nominal) / (nominal + RELATIVE_LOSS_EPSILON))
                    if not math.isfinite(delta):
                        raise ValueError("Relative validation-loss increase is nonfinite")
                    cell_values[family][level].append(delta)
                    raw_observations.append({"id": sample["id"], "family": family, "level": level,
                                             "severity": SEVERITIES[level], "seed": perturb_seed,
                                             "nominal_loss": nominal, "perturbed_loss": loss, "relative_increase": delta,
                                             "parameters": probe["meta"]["calibration_probe"]["parameters"]})
                    completed += 1
                    if progress:
                        progress({"stage": "condition_loss", "completed": completed, "total": total,
                                  "id": sample["id"], "family": family, "level": level, "seed": perturb_seed})
    cells = {}
    medians = {}
    for family in FAMILIES:
        cells[family], medians[family] = [], []
        for level, values in enumerate(cell_values[family]):
            if len(values) != len(samples) * len(seeds):
                raise ValueError(f"Incomplete calibration cell {family}/{level}; imputation is prohibited")
            low, high = np.quantile(values, [.05, .95])
            median = float(np.median(np.clip(values, low, high)))
            medians[family].append(median)
            cells[family].append({"level": level, "severity": SEVERITIES[level], "n": len(values),
                                  "winsor_lower": float(low), "winsor_upper": float(high), "median_increase": median})
    p95 = float(np.quantile([value for family in FAMILIES for value in medians[family]], .95))
    normalizer = p95 if p95 > 0 else RELATIVE_LOSS_EPSILON
    loss_lookup = {"base": {family: [float(np.clip(v / normalizer, 0, 1)) for v in medians[family]] for family in FAMILIES}}
    artifact = {"artifact_version": "tefed-validation-v1", "implementation": "tefed",
                "task": task, "calibration_ids": ids,
                "T_C": temperature["T_C"], "temperature_fit": {**temperature, "source_kinds": sorted({x[2] for x in matched}),
                    "posterior_floor": 1e-12, "association_iou": .5, "class_filter_used": False},
                "luminance_median": luminance_median, "depth_validation_median": depth_median,
                "class_priors": class_priors, "loss_lookup": loss_lookup,
                "nominal_losses": nominal_losses, "cells": cells, "observations": raw_observations,
                "protocol": {"image_size": list(samples[0]["meta"]["size"]), "strict_count": strict_count,
                    "seed": seed, "perturbation_seeds": seeds, "families": list(FAMILIES),
                    "severity_bin_midpoints": list(SEVERITIES), "nominal_frames": len(samples),
                    "perturbed_loss_evaluations": total, "cell_count": 20, "entries_after_fixed_offsets": 320,
                    "loss_epsilon": RELATIVE_LOSS_EPSILON, "relative_increase": "max(0,(perturbed-nominal)/(nominal+epsilon))",
                    "winsorization": "within each family-level cell, 5th/95th percentiles, numpy linear quantiles",
                    "aggregation": "median after winsorization", "normalization_p95": p95,
                    "normalization_denominator": normalizer, "all_cells_zero": p95 == 0,
                    "geometry_offsets": [0., .0271, .0594, .1087], "cue_offsets": [0., .0136, .0324, .0511],
                    "geometry_offset_origin": "manuscript SI fixed settings, not fitted by this routine",
                    "zero_nominal_loss_ids": [uid for uid, loss in nominal_losses.items() if loss == 0],
                    "luminance_definition": "pooled content-pixel BT.601-weighted RGB in [0,1], excluding letterbox pad",
                    "depth_definition": "median of positive finite content-pixel sensor values over calibration frames",
                    "class_prior_coordinates": "normalized letterboxed detector input; aspect uses pixel width/height",
                    "depth_valid_fraction_minimum": .35, "height_depth_regression": "log(h/H)=a_c+b_c*log(max(z/z_val,1e-3))",
                    "area_depth_regression": "log(normalized_box_area)=intercept+slope*log(max(z/z_val,1e-3))",
                    "geometry_preserved": True, "ground_truth_preserved": True,
                    "perturbation_space": "image", "class_transition": "none",
                    "frozen_model_responsibility": "caller freezes weights, gradients and BN and supplies one selected checkpoint"}}
    # Reject nonfinite JSON values.
    json.dumps(artifact, allow_nan=False)
    return artifact
