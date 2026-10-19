"""Validate manifests and transform aligned TE-Fed observations.

Boxes use normalized xyxy and sizes use (width, height). Unlabeled records
never expose labels as supervision."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os

import numpy as np


SPLIT_ALIASES = {
    "label": "labeled", "labeled": "labeled", "labelled": "labeled",
    "unlabel": "unlabeled", "unlabeled": "unlabeled", "unlabelled": "unlabeled",
    "val": "validation", "validation": "validation", "test": "test", "hard": "hard",
}


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty string or integer")
    return str(value)


def canonical_split(value: str) -> str:
    try:
        return SPLIT_ALIASES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown split {value!r}; expected {sorted(SPLIT_ALIASES)}") from exc


@dataclass(frozen=True)
class Annotation:
    box: tuple[float, float, float, float]
    class_id: int

    def as_dict(self) -> dict[str, Any]:
        return {"box": list(self.box), "class_id": self.class_id}


def parse_annotation(value: Mapping[str, Any] | Annotation) -> Annotation:
    if isinstance(value, Annotation):
        value = value.as_dict()
    if not isinstance(value, Mapping) or "box" not in value or "class_id" not in value:
        raise ValueError("Each annotation needs box and class_id")
    cls = value["class_id"]
    if isinstance(cls, bool) or not isinstance(cls, (int, np.integer)) or cls < 0:
        raise ValueError("class_id must be a nonnegative integer")
    raw = value["box"]
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 4:
        raise ValueError("box must contain four normalized xyxy coordinates")
    if any(isinstance(x, (bool, np.bool_)) for x in raw):
        raise ValueError("Box coordinates must be numeric, not boolean")
    try:
        box = tuple(float(x) for x in raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Box coordinates must be numeric") from exc
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
        raise ValueError("Box coordinates must be finite and in [0, 1]")
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("Box must have positive width and height")
    return Annotation(box, int(cls))


@dataclass(frozen=True)
class SampleRecord:
    id: str
    client_id: str
    split: str
    rgb: Path
    depth: Path | None = None
    nir: Path | None = None
    labels: tuple[Annotation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.metadata)
        out.update(id=self.id, client_id=self.client_id, split=self.split, rgb=str(self.rgb))
        if self.depth is not None:
            out["depth"] = str(self.depth)
        if self.nir is not None:
            out["nir"] = str(self.nir)
        # Exclude optional evaluation labels from unlabeled records.
        if self.split != "unlabeled":
            out["labels"] = [x.as_dict() for x in self.labels]
        return out


@dataclass(frozen=True)
class Manifest:
    records: tuple[SampleRecord, ...]
    calibration_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def by_split(self, split: str, client_id: str | int | None = None) -> tuple[SampleRecord, ...]:
        name = canonical_split(split)
        client = None if client_id is None else _identifier(client_id, "client_id")
        return tuple(r for r in self.records if r.split == name and (client is None or r.client_id == client))

    @property
    def by_id(self) -> dict[str, SampleRecord]:
        return {r.id: r for r in self.records}

    def as_dict(self) -> dict[str, Any]:
        return {**self.metadata, "records": [r.as_dict() for r in self.records],
                "calibration_ids": list(self.calibration_ids)}


def _path(value: Any, base_dir: Path, name: str, required: bool = False) -> Path | None:
    if value is None or value == "":
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{name} must be a local path")
    if "://" in str(value):
        raise ValueError(f"{name} must be a local path, not a URL")
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base_dir / p).resolve()


def validate_manifest(
    records: Iterable[Mapping[str, Any] | SampleRecord], *,
    calibration_ids: Sequence[str | int] = (), strict_files: bool = False,
    base_dir: str | Path | None = None, metadata: Mapping[str, Any] | None = None,
) -> Manifest:
    """Validate identities, assets, splits, and calibration membership.

    Each identity has one record. Evaluation views reuse test IDs;
    calibration IDs must belong to validation."""
    base = Path(base_dir or ".").resolve()
    out: list[SampleRecord] = []
    ids: set[str] = set()
    assets: dict[str, str] = {}
    metadata = dict(metadata or {})
    class_names = metadata.get("class_names")
    if class_names is not None and (not isinstance(class_names, list) or not class_names
                                    or not all(isinstance(x, str) and x.strip() for x in class_names)):
        raise ValueError("class_names must be a nonempty list of names")
    for index, value in enumerate(records):
        if isinstance(value, SampleRecord):
            # Inspect raw labels; as_dict() hides unlabeled annotations.
            value = {**value.metadata, "id": value.id, "client_id": value.client_id,
                     "split": value.split, "rgb": value.rgb, "depth": value.depth,
                     "nir": value.nir, "labels": [x.as_dict() for x in value.labels]}
        if not isinstance(value, Mapping):
            raise ValueError(f"Record {index} is not an object")
        missing = {"id", "client_id", "split", "rgb"} - value.keys()
        if missing:
            raise ValueError(f"Record {index} is missing {sorted(missing)}")
        uid = _identifier(value["id"], "id")
        if uid in ids:
            raise ValueError(f"Duplicate identity {uid!r}: identities cannot cross clients or splits")
        ids.add(uid)
        split = canonical_split(value["split"])
        raw_labels = value.get("labels")
        if raw_labels is None:
            if split != "unlabeled":
                raise ValueError(f"Record {uid!r}: labeled/evaluation records require explicit labels, including []")
            raw_labels = []
        if not isinstance(raw_labels, (list, tuple)):
            raise ValueError(f"Record {uid!r}: labels must be a list")
        labels = tuple(parse_annotation(x) for x in raw_labels)
        if class_names is not None and any(a.class_id >= len(class_names) for a in labels):
            raise ValueError(f"Record {uid!r}: class_id outside class_names")
        rgb = _path(value["rgb"], base, "rgb", required=True)
        depth = _path(value.get("depth"), base, "depth")
        nir = _path(value.get("nir"), base, "nir")
        key = os.path.normcase(str(rgb))
        if key in assets:
            raise ValueError(f"RGB asset is duplicated under {assets[key]!r} and {uid!r}; possible split leakage")
        assets[key] = uid
        if strict_files:
            for name, path in (("rgb", rgb), ("depth", depth), ("nir", nir)):
                if path is not None and not path.is_file():
                    raise FileNotFoundError(f"Record {uid!r} {name}: {path}")
        extra = {k: v for k, v in value.items()
                 if k not in {"id", "client_id", "split", "rgb", "depth", "nir", "labels"}}
        out.append(SampleRecord(uid, _identifier(value["client_id"], "client_id"), split,
                                rgb, depth, nir, labels, extra))
    if not out:
        raise ValueError("Manifest must contain at least one record")
    if isinstance(calibration_ids, (str, bytes)) or not isinstance(calibration_ids, Sequence):
        raise ValueError("calibration_ids must be an explicit sequence of IDs")
    calibration = tuple(_identifier(x, "calibration ID") for x in calibration_ids)
    if len(set(calibration)) != len(calibration):
        raise ValueError("calibration_ids contains duplicates")
    by_id = {r.id: r for r in out}
    for uid in calibration:
        if uid not in by_id or by_id[uid].split != "validation":
            raise ValueError(f"Calibration identity {uid!r} must belong to validation")
    manifest = Manifest(tuple(out), calibration, metadata)
    if "views" in metadata:
        validate_view_identities(manifest, metadata["views"],
                                 require_equal_standard_views=bool(metadata.get("require_equal_standard_views", False)))
    return manifest


def load_manifest(path: str | Path, *, strict_files: bool = False) -> Manifest:
    """Load JSON records or JSONL; resolve images relative to the manifest.

    JSON objects may include records, calibration_ids, class_names, and views.
    JSONL contains records only."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".jsonl":
            data: Any = []
            for n, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {n}") from exc
        else:
            data = json.load(stream)
    if isinstance(data, list):
        return validate_manifest(data, strict_files=strict_files, base_dir=path.parent)
    if not isinstance(data, Mapping) or not isinstance(data.get("records"), list):
        raise ValueError("Manifest JSON must be a list or an object with a records list")
    metadata = {k: v for k, v in data.items() if k not in {"records", "calibration_ids"}}
    return validate_manifest(data["records"], calibration_ids=data.get("calibration_ids", ()),
                             strict_files=strict_files, base_dir=path.parent, metadata=metadata)


def validate_view_identities(manifest: Manifest, views: Mapping[str, Sequence[str | int]], *,
                             require_equal_standard_views: bool = False) -> None:
    """Validate that evaluation views contain only test IDs.

    require_equal_standard_views enforces identical Participating, Unseen,
    and New-client IDs; otherwise held-out subsets may differ."""
    if not isinstance(views, Mapping):
        raise ValueError("views must map view names to lists of IDs")
    normalized: dict[str, set[str]] = {}
    known = manifest.by_id
    aliases = {"participating": "participating", "unseen": "unseen", "new-client": "new-client",
               "new_client": "new-client", "hard": "hard", "test": "participating"}
    for name, values in views.items():
        key = aliases.get(str(name).lower())
        if key is None:
            raise ValueError(f"Unknown evaluation view {name!r}")
        if key in normalized:
            raise ValueError(f"Duplicate evaluation view alias {name!r}")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"View {name!r} needs a nonempty ID list")
        seq = [_identifier(v, "view ID") for v in values]
        if len(set(seq)) != len(seq):
            raise ValueError(f"View {name!r} contains duplicate identities")
        expected_split = "hard" if key == "hard" else "test"
        for uid in seq:
            if uid not in known or known[uid].split != expected_split:
                raise ValueError(f"View {name!r} ID {uid!r} must be in split {expected_split}")
        normalized[key] = set(seq)
    if require_equal_standard_views:
        standard = [normalized[k] for k in ("participating", "unseen", "new-client") if k in normalized]
        if any(s != standard[0] for s in standard[1:]):
            raise ValueError("Standard evaluation views must reuse identical test identities")


@dataclass(frozen=True)
class LetterboxTransform:
    original_size: tuple[int, int]
    size: tuple[int, int]
    resized_size: tuple[int, int]
    scale: float
    scale_xy: tuple[float, float]
    pad: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {"original_size": self.original_size, "size": self.size,
                "resized_size": self.resized_size, "scale": self.scale,
                "scale_xy": self.scale_xy, "pad": self.pad}


def letterbox_parameters(original_size: tuple[int, int], size: int | tuple[int, int] = (640, 640)) -> LetterboxTransform:
    if isinstance(size, int) and not isinstance(size, bool):
        size = (size, size)
    if len(original_size) != 2 or len(size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in (*original_size, *size)
    ):
        raise ValueError("Image sizes must contain positive integer width and height")
    width, height = map(int, original_size)
    dst_w, dst_h = map(int, size)
    scale = min(dst_w / width, dst_h / height)
    new_w = min(dst_w, max(1, int(round(width * scale))))
    new_h = min(dst_h, max(1, int(round(height * scale))))
    return LetterboxTransform((width, height), (dst_w, dst_h), (new_w, new_h), scale,
                              (new_w / width, new_h / height), ((dst_w - new_w) // 2, (dst_h - new_h) // 2))


def map_boxes(boxes: Sequence[Sequence[float]] | np.ndarray, transform: LetterboxTransform | Mapping[str, Any], *, inverse: bool = False) -> np.ndarray:
    """Map normalized xyxy boxes through the rounded resize geometry."""
    if isinstance(transform, Mapping):
        transform = LetterboxTransform(**{key: transform[key] for key in
            ("original_size", "size", "resized_size", "scale", "scale_xy", "pad")})
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4 or not np.isfinite(arr).all():
        raise ValueError("boxes must be a finite Nx4 array")
    src = np.array([*transform.original_size, *transform.original_size], dtype=float)
    dst = np.array([*transform.size, *transform.size], dtype=float)
    scales = np.array([*transform.scale_xy, *transform.scale_xy])
    pad = np.array([*transform.pad, *transform.pad])
    mapped = ((arr * dst - pad) / scales) / src if inverse else (arr * src * scales + pad) / dst
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def load_sample(record: SampleRecord, *, size: int | tuple[int, int] = (640, 640),
                require_modalities: Sequence[str] = (), include_labels: bool | None = None) -> dict[str, Any]:
    """Letterbox aligned RGB, depth, NIR, and targets with one transform.

    Return RGB uint8 HWC3, depth float32 HW in sensor units, NIR uint8 HW,
    normalized targets, and transform metadata. Depth uses nearest interpolation
    and zero padding; RGB/NIR use bilinear. NIR accepts uint8 or [0,1] float .npy;
    higher bit depths require upstream preprocessing. Unlabeled targets stay []."""
    from PIL import Image

    if any(m not in {"rgb", "depth", "nir"} for m in require_modalities):
        raise ValueError("require_modalities accepts rgb, depth and nir")
    with Image.open(record.rgb) as image:
        rgb = image.convert("RGB")
        transform = letterbox_parameters(rgb.size, size)
        resized = rgb.resize(transform.resized_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", transform.size, (114, 114, 114))
        canvas.paste(resized, transform.pad)
        rgb_array = np.array(canvas, dtype=np.uint8)
    modalities: dict[str, np.ndarray | None] = {"depth": None, "nir": None}
    for name, path in (("depth", record.depth), ("nir", record.nir)):
        if path is None:
            if name in require_modalities:
                raise ValueError(f"Sample {record.id!r} lacks required modality {name}")
            continue
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                arr = np.array(image)
        if arr.ndim != 2:
            raise ValueError(f"Sample {record.id!r}: {name} must be one channel")
        if (arr.shape[1], arr.shape[0]) != transform.original_size:
            raise ValueError(f"Sample {record.id!r}: {name} is not registered at RGB resolution")
        if name == "nir" and arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and np.isfinite(arr).all() and np.all((arr >= 0) & (arr <= 1)):
                arr = np.rint(arr * 255).astype(np.uint8)
            else:
                raise ValueError("NIR must be uint8 or floating-point in [0,1]; normalize high-bit-depth inputs explicitly")
        if name == "depth":
            arr = arr.astype(np.float32)
            if not np.isfinite(arr).all() or np.any(arr < 0):
                raise ValueError("Depth must contain finite nonnegative sensor values")
            layer = Image.fromarray(arr).resize(transform.resized_size, Image.Resampling.NEAREST)
            canvas = Image.new("F", transform.size, 0.0)
        else:
            layer = Image.fromarray(arr).resize(transform.resized_size, Image.Resampling.BILINEAR)
            canvas = Image.new("L", transform.size, 0)
        canvas.paste(layer, transform.pad)
        modalities[name] = np.array(canvas, dtype=np.float32 if name == "depth" else np.uint8)
    use_labels = record.split != "unlabeled" and include_labels is not False
    labels = record.labels if use_labels else ()
    mapped = map_boxes([a.box for a in labels], transform)
    targets = [{"box": b.tolist(), "class_id": a.class_id} for b, a in zip(mapped, labels)]
    return {"id": record.id, "client_id": record.client_id, "split": record.split,
            "rgb": rgb_array, **modalities, "targets": targets,
            "meta": transform.as_dict(), "ignore_labels": record.split == "unlabeled"}


def oriented_to_horizontal(box: Sequence[float] | Sequence[Sequence[float]], *,
                           image_size: tuple[int, int] | None = None, normalize: bool = False) -> list[float]:
    """Enclose four corners in a horizontal xyxy box.

    Accept eight pixel coordinates or a 4x2 array; normalize=True requires size
    (width, height). Angle conventions are not inferred."""
    pts = np.asarray(box, dtype=np.float64)
    if pts.size != 8 or not np.isfinite(pts).all():
        raise ValueError("Oriented box must contain four finite xy corners")
    pts = pts.reshape(4, 2)
    out = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError("Oriented box has no positive-area horizontal enclosure")
    if normalize:
        if image_size is None:
            raise ValueError("image_size is required for normalized boxes")
        letterbox_parameters(image_size, image_size)  # Validate the dimensions.
        out /= np.array([*image_size, *image_size], dtype=float)
        out = np.clip(out, 0, 1)
        if out[0] >= out[2] or out[1] >= out[3]:
            raise ValueError("Oriented box lies outside the image")
    return out.tolist()


def apply_unseen(sample: Mapping[str, Any], *, identity: str | int,
                 perturbation: Mapping[str, float | bool], seed: int = 0) -> dict[str, Any]:
    """Apply aligned image-space perturbations; translation uses image fractions."""
    allowed = {"rgb_brightness", "rgb_contrast", "depth_scale", "depth_noise_std",
               "nir_brightness", "nir_noise_std", "translate_x", "translate_y", "horizontal_flip"}
    unknown = set(perturbation) - allowed
    if unknown:
        raise ValueError(f"Unknown unseen perturbations: {sorted(unknown)}")
    values: dict[str, Any] = {}
    for key, value in perturbation.items():
        if key == "horizontal_flip":
            if not isinstance(value, bool):
                raise ValueError("horizontal_flip must be boolean")
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        elif key not in {"translate_x", "translate_y"} and value < 0:
            raise ValueError(f"{key} must be nonnegative")
        values[key] = value
    token = json.dumps({"id": _identifier(identity, "identity"), "seed": int(seed),
                        "perturbation": values}, sort_keys=True, separators=(",", ":"))
    rng = np.random.default_rng(int.from_bytes(sha256(token.encode()).digest()[:8], "little"))
    out = dict(sample)
    out["meta"] = dict(sample.get("meta", {}))
    out["targets"] = [parse_annotation(t).as_dict() for t in sample.get("targets", [])]
    rgb = np.asarray(sample["rgb"], dtype=np.float32)
    rgb = (rgb - 127.5) * values.get("rgb_contrast", 1.0) + 127.5
    out["rgb"] = np.clip(rgb * values.get("rgb_brightness", 1.0), 0, 255).astype(np.uint8)
    for name in ("depth", "nir"):
        arr = sample.get(name)
        if arr is None:
            if any(key in values for key in (f"{name}_scale", f"{name}_brightness", f"{name}_noise_std")):
                raise ValueError(f"Cannot perturb absent modality {name}")
            out[name] = None
            continue
        arr = np.asarray(arr, dtype=np.float32).copy()
        valid = arr > 0 if name == "depth" else np.ones_like(arr, dtype=bool)
        arr *= values.get("depth_scale" if name == "depth" else "nir_brightness", 1.0)
        if values.get(f"{name}_noise_std", 0):
            arr[valid] += rng.normal(0, values[f"{name}_noise_std"], size=int(valid.sum()))
        if name == "depth":
            arr[~valid] = 0
            out[name] = np.maximum(arr, 0).astype(np.float32)
        else:
            out[name] = np.clip(arr, 0, 255).astype(np.uint8)
    height, width = out["rgb"].shape[:2]
    dx = int(round(values.get("translate_x", 0.0) * width))
    dy = int(round(values.get("translate_y", 0.0) * height))
    flip = values.get("horizontal_flip", False)
    if flip or dx or dy:
        for name in ("rgb", "depth", "nir"):
            if out[name] is None:
                continue
            arr = out[name][:, ::-1].copy() if flip else out[name]
            shifted = np.full_like(arr, 114 if name == "rgb" else 0)
            sx0, sy0 = max(0, -dx), max(0, -dy)
            sx1, sy1 = min(width, width - dx), min(height, height - dy)
            if sx1 > sx0 and sy1 > sy0:
                shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = arr[sy0:sy1, sx0:sx1]
            out[name] = shifted
        targets = []
        for target in out["targets"]:
            x1, y1, x2, y2 = target["box"]
            if flip:
                x1, x2 = 1 - x2, 1 - x1
            box = np.clip([x1 + dx / width, y1 + dy / height,
                           x2 + dx / width, y2 + dy / height], 0, 1).tolist()
            if box[0] < box[2] and box[1] < box[3]:
                targets.append({"box": box, "class_id": target["class_id"]})
        out["targets"] = targets
    out["meta"]["unseen"] = {"identity": str(identity), "seed": int(seed), "perturbation": values}
    return out
