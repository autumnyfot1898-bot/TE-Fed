#!/usr/bin/env python3
"""Normalize explicit data partitions."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tefed.data import load_manifest, oriented_to_horizontal, validate_manifest


def coco_records(path: Path, image_root: Path, client_id: str, split: str):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
        raise ValueError("COCO input requires images, annotations and categories lists")
    categories = sorted(data["categories"], key=lambda x: int(x["id"]))
    category_map = {int(c["id"]): i for i, c in enumerate(categories)}
    if len(category_map) != len(categories):
        raise ValueError("COCO categories contain duplicate IDs")
    images = {i["id"]: i for i in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("COCO images contain duplicate IDs")
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            raise ValueError("Crowd/ignore annotations need an explicit supported evaluation protocol; not silently discarded")
        if ann.get("image_id") not in images or ann.get("category_id") not in category_map:
            raise ValueError("COCO annotation refers to an unknown image/category")
        image = images[ann["image_id"]]
        width, height = image["width"], image["height"]
        if width <= 0 or height <= 0:
            raise ValueError("COCO image width/height must be positive")
        if "quadrilateral" in ann:
            box = oriented_to_horizontal(ann["quadrilateral"], image_size=(width, height), normalize=True)
        else:
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("COCO bbox must be pixel xywh; use quadrilateral for four explicit corners")
            x, y, w, h = map(float, bbox)
            box = [x / width, y / height, (x + w) / width, (y + h) / height]
        annotations[ann["image_id"]].append({"box": box, "class_id": category_map[ann["category_id"]]})
    records = [{"id": f"{client_id}:{image['id']}", "client_id": client_id, "split": split,
                "rgb": str((image_root / image["file_name"]).resolve()),
                "labels": annotations[image["id"]]} for image in data["images"]]
    return records, {"class_names": [str(c["name"]) for c in categories],
                     "source_category_id_to_class_id": {str(k): v for k, v in category_map.items()}}


def dronevehicle_records(annotation_dir: Path, image_root: Path, client_id: str,
                         split: str, class_map_path: Path, image_suffix: str,
                         difficult_policy: str):
    from PIL import Image
    class_map = json.loads(class_map_path.read_text(encoding="utf-8-sig"))
    if not isinstance(class_map, dict) or not class_map or sorted(class_map.values()) != list(range(len(class_map))):
        raise ValueError("class-map must map category names to contiguous integer class IDs starting at zero")
    files = sorted(annotation_dir.glob("*.txt"))
    if not files:
        raise ValueError("No *.txt DroneVehicle annotation files found")
    records = []
    for file in files:
        image_path = image_root / (file.stem + image_suffix)
        with Image.open(image_path) as image:
            size = image.size
        labels = []
        for line_no, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip() or line.strip().lower().startswith(("imagesource:", "gsd:")):
                continue
            fields = line.split()
            if len(fields) not in (9, 10):
                raise ValueError(f"{file}:{line_no}: expected 8 corner coordinates, class and optional difficulty")
            name = fields[8]
            if name not in class_map:
                raise ValueError(f"{file}:{line_no}: unknown category {name!r}")
            difficult = int(fields[9]) if len(fields) == 10 else 0
            if difficult and difficult_policy == "error":
                raise ValueError(f"{file}:{line_no}: choose an explicit difficult-policy")
            if difficult and difficult_policy == "drop":
                continue
            box = oriented_to_horizontal([float(x) for x in fields[:8]], image_size=size, normalize=True)
            labels.append({"box": box, "class_id": class_map[name]})
        records.append({"id": f"{client_id}:{file.stem}", "client_id": client_id,
                        "split": split, "rgb": str(image_path.resolve()), "labels": labels})
    return records, {"class_names": [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])],
                     "oriented_conversion": "minimum enclosing horizontal box",
                     "difficult_policy": difficult_policy}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-format", choices=("auto", "json", "jsonl", "csv", "coco", "dronevehicle"), default="auto")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--client-id")
    parser.add_argument("--split", choices=("labeled", "unlabeled", "validation", "test", "hard"))
    parser.add_argument("--class-map", type=Path, help="DroneVehicle category-name to zero-based integer mapping JSON")
    parser.add_argument("--image-suffix", default=".jpg")
    parser.add_argument("--difficult-policy", choices=("error", "include", "drop"), default="error")
    parser.add_argument("--calibration-ids", type=Path, help="JSON list of existing validation IDs")
    parser.add_argument("--views", type=Path, help="JSON object mapping view names to held-out ID lists")
    parser.add_argument("--require-equal-standard-views", action="store_true")
    args = parser.parse_args(argv)
    path = args.input.resolve()
    fmt = args.input_format
    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()
    metadata = {}
    calibration_ids = []
    if fmt in ("json", "jsonl"):
        manifest = load_manifest(path, strict_files=args.check_files)
        records = manifest.records
        metadata = dict(manifest.metadata)
        calibration_ids = list(manifest.calibration_ids)
    elif fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = []
            for row in csv.DictReader(stream):
                row = {key: value for key, value in row.items() if value != ""}
                if "labels" in row:
                    row["labels"] = json.loads(row["labels"])
                records.append(row)
    elif fmt in ("coco", "dronevehicle"):
        if args.image_root is None or args.client_id is None or args.split is None:
            parser.error("COCO/DroneVehicle conversion requires --image-root, --client-id and --split")
        if fmt == "coco":
            records, metadata = coco_records(path, args.image_root.resolve(), args.client_id, args.split)
        else:
            if args.class_map is None:
                parser.error("DroneVehicle conversion requires --class-map")
            records, metadata = dronevehicle_records(path, args.image_root.resolve(), args.client_id,
                args.split, args.class_map.resolve(), args.image_suffix, args.difficult_policy)
    else:
        parser.error("Cannot infer input format; provide --input-format")
    if args.calibration_ids:
        calibration_ids = json.loads(args.calibration_ids.read_text(encoding="utf-8-sig"))
        if not isinstance(calibration_ids, list):
            raise ValueError("calibration-ids file must contain a JSON list")
    if args.views:
        metadata["views"] = json.loads(args.views.read_text(encoding="utf-8-sig"))
    if args.require_equal_standard_views:
        metadata["require_equal_standard_views"] = True
    manifest = validate_manifest(records, calibration_ids=calibration_ids, strict_files=args.check_files,
                                 base_dir=path.parent, metadata=metadata)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Validate before replacing the output atomically.
    handle, temp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest.as_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, out)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    counts = defaultdict(int)
    for row in manifest.records:
        counts[f"{row.client_id}/{row.split}"] += 1
    print(json.dumps({"output": str(out), "records": len(manifest.records),
                      "calibration_id_count": len(manifest.calibration_ids), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Manifest preparation failed: {exc}")
