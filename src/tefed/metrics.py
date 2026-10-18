"""Compute COCO-style 101-point box AP and cross-seed summaries.

Crowd/ignore labels and area strata are unsupported. Boxes use normalized xyxy.
Classes without GT are excluded from AP; all-GT-empty AP is None."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import numbers

import numpy as np

from .data import Annotation, Manifest, _identifier, canonical_split, parse_annotation


def box_iou(box: Sequence[float], boxes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    reference = np.asarray(box, dtype=np.float64)
    other = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(other):
        return np.empty(0, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(reference[2:], other[:, 2:]) - np.maximum(reference[:2], other[:, :2]))
    intersection = intersection[:, 0] * intersection[:, 1]
    area = (reference[2] - reference[0]) * (reference[3] - reference[1])
    other_area = (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1])
    union = area + other_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _normalize_rows(rows: Mapping[str | int, Sequence[Mapping[str, Any] | Annotation]], *, predictions: bool) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(rows, Mapping):
        raise ValueError("Targets and predictions must be mappings from image ID to lists")
    out = {}
    for raw_uid, values in rows.items():
        uid = _identifier(raw_uid, "image ID")
        if uid in out:
            raise ValueError(f"Duplicate canonical image ID {uid!r}")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Image {uid!r} requires a list of boxes")
        boxes = []
        for value in values:
            try:
                anno = parse_annotation(value)
            except ValueError as exc:
                box = value.get("box") if isinstance(value, Mapping) else getattr(value, "box", None)
                kind = "prediction" if predictions else "target"
                raise ValueError(f"Invalid {kind} for image {uid!r}, box={box!r}: {exc}") from exc
            item = anno.as_dict()
            if predictions:
                score = value.get("score") if isinstance(value, Mapping) else None
                if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Prediction for {uid!r} needs a finite score in [0, 1]")
                item["score"] = float(score)
            boxes.append(item)
        out[uid] = boxes
    return out


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if not len(recall):
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    sampled = np.zeros(101, dtype=float)
    valid = index < len(envelope)
    sampled[valid] = envelope[index[valid]]
    return float(sampled.mean())


def evaluate_detection(
    targets: Mapping[str | int, Sequence[Mapping[str, Any] | Annotation]],
    predictions: Mapping[str | int, Sequence[Mapping[str, Any]]], *,
    iou_thresholds: Sequence[float] | None = None, max_detections: int = 100,
    strict_ids: bool = True,
) -> dict[str, Any]:
    """Compute AP50, AP75, and mAP50_95 for image-keyed targets/predictions.

    Entries contain box/class_id and prediction score; use [] for no detections.
    ID sets must match unless strict_ids=False allows missing predictions.
    Extra prediction IDs are rejected. max_detections applies per image/class."""
    gt = _normalize_rows(targets, predictions=False)
    pred = _normalize_rows(predictions, predictions=True)
    extra = set(pred) - set(gt)
    missing = set(gt) - set(pred)
    if extra or (strict_ids and missing):
        raise ValueError(f"Prediction identity mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections <= 0:
        raise ValueError("max_detections must be a positive integer")
    thresholds = list(np.linspace(.5, .95, 10) if iou_thresholds is None else iou_thresholds)
    if not thresholds or any(isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("iou_thresholds must contain finite values in (0, 1]")
    if len(set(round(float(t), 10) for t in thresholds)) != len(thresholds):
        raise ValueError("iou_thresholds must be unique")
    if any(abs(float(t) - round(float(t), 2)) > 1e-10 for t in thresholds):
        raise ValueError("iou_thresholds must use at most two decimal places")
    thresholds = sorted(float(t) for t in thresholds)
    classes = sorted({x["class_id"] for rows in gt.values() for x in rows} |
                     {x["class_id"] for rows in pred.values() for x in rows})
    per_class: dict[str, dict[str, Any]] = {}
    counts_raw = sum(len(rows) for rows in pred.values())
    total_used = 0
    for cls in classes:
        gt_by_image = {uid: [x["box"] for x in rows if x["class_id"] == cls] for uid, rows in gt.items()}
        n_gt = sum(len(rows) for rows in gt_by_image.values())
        detection_rows = []
        for uid in sorted(gt):
            candidates = [(i, x) for i, x in enumerate(pred.get(uid, [])) if x["class_id"] == cls]
            candidates.sort(key=lambda pair: (-pair[1]["score"], pair[0]))
            detection_rows.extend((x["score"], uid, i, x["box"]) for i, x in candidates[:max_detections])
        detection_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
        total_used += len(detection_rows)
        aps = {}
        counts = {}
        for threshold in thresholds:
            matched = {uid: np.zeros(len(boxes), dtype=bool) for uid, boxes in gt_by_image.items()}
            tp = np.zeros(len(detection_rows), dtype=np.float64)
            for index, (_, uid, _, box) in enumerate(detection_rows):
                overlaps = box_iou(box, gt_by_image[uid])
                if len(overlaps):
                    overlaps[matched[uid]] = -1
                    # Match COCO ties: the later GT index wins equal IoU.
                    best = len(overlaps) - 1 - int(np.argmax(overlaps[::-1]))
                    if overlaps[best] >= threshold - 1e-12:
                        matched[uid][best] = True
                        tp[index] = 1
            fp = 1 - tp
            key = f"{threshold:.2f}"
            if n_gt:
                recall = np.cumsum(tp) / n_gt
                precision = np.cumsum(tp) / np.maximum(1, np.cumsum(tp + fp))
                aps[key] = _ap(recall, precision)
            else:
                aps[key] = None
            counts[key] = {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(n_gt - tp.sum())}
        defined = [v for v in aps.values() if v is not None]
        per_class[str(cls)] = {"num_ground_truth": n_gt, "num_predictions": len(detection_rows),
                               "AP50": aps.get("0.50"), "AP75": aps.get("0.75"),
                               "mAP": float(np.mean(defined)) if defined else None,
                               "ap_by_iou": aps, "counts_by_iou": counts}
    mean_ap = {}
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        valid = [r["ap_by_iou"][key] for r in per_class.values() if r["ap_by_iou"][key] is not None]
        mean_ap[key] = float(np.mean(valid)) if valid else None
    valid = [v for v in mean_ap.values() if v is not None]
    standard = np.linspace(.5, .95, 10)
    standard_grid = len(thresholds) == 10 and np.allclose(thresholds, standard, rtol=0, atol=1e-10)
    result = {"AP50": mean_ap.get("0.50"), "mAP50": mean_ap.get("0.50"),
              "AP75": mean_ap.get("0.75"), "mAP": float(np.mean(valid)) if valid else None,
              "mAP50_95": float(np.mean(valid)) if valid and standard_grid else None,
              "ap_by_iou": mean_ap, "per_class": per_class, "num_images": len(gt),
              "num_ground_truth": sum(len(v) for v in gt.values()), "num_predictions": counts_raw,
              "num_predictions_evaluated": total_used, "classes_with_ground_truth":
              sum(r["num_ground_truth"] > 0 for r in per_class.values()),
              "iou_thresholds": thresholds, "max_detections_per_image_class": max_detections}
    return result


def evaluate_jsonl(manifest: Manifest, prediction_path: str | Path, *, split: str = "test",
                   client_id: str | int | None = None, view: str | None = None,
                   max_detections: int = 100) -> dict[str, Any]:
    """Read one {id, predictions:[{box, class_id, score}]} record per identity.

    IDs must match the split/client or view. Boxes use original normalized
    image coordinates; the unlabeled adaptation stream is not evaluated."""
    split = canonical_split(split)
    if split not in {"test", "hard", "validation"}:
        raise ValueError("Evaluation accepts validation, test or hard splits only")
    if view is not None:
        views = manifest.metadata.get("views", {})
        if view not in views:
            raise ValueError(f"Manifest has no explicit view {view!r}")
        ids = set(str(x) for x in views[view])
        records = [r for r in manifest.records if r.id in ids]
        if client_id is not None:
            raise ValueError("A named view already defines its IDs; do not filter it by client")
    else:
        records = list(manifest.by_split(split, client_id))
    if not records:
        raise ValueError("The evaluation selection is empty")
    targets = {r.id: [a.as_dict() for a in r.labels] for r in records}
    predictions = {}
    with Path(prediction_path).open("r", encoding="utf-8-sig") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSON on line {n}") from exc
            if not isinstance(row, Mapping) or "id" not in row or "predictions" not in row:
                raise ValueError(f"Prediction line {n} needs id and predictions")
            uid = _identifier(row["id"], "prediction id")
            if uid in predictions:
                raise ValueError(f"Duplicate prediction identity {uid!r}")
            predictions[uid] = row["predictions"]
    return evaluate_detection(targets, predictions, max_detections=max_detections)


def aggregate_seed_metrics(seed_results: Mapping[str | int, Mapping[str, float]] | Sequence[Mapping[str, Any]], *,
                           expected_seeds: Sequence[str | int]) -> dict[str, Any]:
    """Return mean and sample SD over the exact declared seeds.

    Accept seed-keyed metric dicts or records containing seed and metrics.
    Require at least two seeds, equal metric sets, and finite numeric values."""
    expected = [_identifier(s, "expected seed") for s in expected_seeds]
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ValueError("At least two unique expected seeds are required for sample SD")
    if isinstance(seed_results, Mapping):
        rows = [(key, value) for key, value in seed_results.items()]
    elif isinstance(seed_results, (list, tuple)):
        rows = []
        for row in seed_results:
            if not isinstance(row, Mapping) or "seed" not in row or "metrics" not in row:
                raise ValueError("Seed records need seed and metrics")
            rows.append((row["seed"], row["metrics"]))
    else:
        raise ValueError("seed_results must be a mapping or a list of seed records")
    normalized = {}
    for raw_seed, metrics in rows:
        seed = _identifier(raw_seed, "seed")
        if seed in normalized:
            raise ValueError(f"Duplicate seed {seed!r}")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"Seed {seed!r} requires a nonempty metric mapping")
        parsed = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"Seed {seed!r}: {key!r} must be a finite numeric metric")
            parsed[key] = float(value)
        normalized[seed] = parsed
    if set(normalized) != set(expected):
        raise ValueError(f"Seed mismatch: missing={sorted(set(expected)-set(normalized))}, extra={sorted(set(normalized)-set(expected))}")
    keys = set(normalized[expected[0]])
    if any(set(row) != keys for row in normalized.values()):
        raise ValueError("All seeds must report the same metric names")
    summary = {}
    for key in sorted(keys):
        values = np.array([normalized[seed][key] for seed in expected], dtype=np.float64)
        summary[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "n": len(values)}
    return {"seed_ids": expected, "n": len(expected), "metrics": summary}
