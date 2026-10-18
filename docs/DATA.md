# Data, manifests and evaluation coordinates

TE-Fed reads synchronized observations, fixed partitions and evaluation identities from a local manifest. Experiment configurations default to seeds `0..9`. Dataset files and renderer assets are supplied separately.

## Manifest contract

A JSON manifest has one record per original synchronized observation identity. Paths are local paths, resolved relative to the manifest file. Example:

```json
{
  "class_names": ["ripe", "non-ripe"],
  "calibration_ids": ["client1:val:001"],
  "require_equal_standard_views": true,
  "views": {
    "Participating": ["client1:test:001"],
    "Unseen": ["client1:test:001"],
    "New-client": ["client1:test:001"],
    "Hard": ["client1:hard:001"]
  },
  "records": [
    {"id": "client1:train:001", "client_id": "client1", "split": "labeled", "rgb": "images/001_rgb.png", "depth": "images/001_depth.png", "nir": "images/001_nir.png", "labels": [{"box": [0.1, 0.2, 0.4, 0.6], "class_id": 0}]},
    {"id": "client1:unlabeled:001", "client_id": "client1", "split": "unlabeled", "rgb": "images/002_rgb.png"},
    {"id": "client1:val:001", "client_id": "client1", "split": "validation", "rgb": "images/003_rgb.png", "labels": []},
    {"id": "client1:test:001", "client_id": "client1", "split": "test", "rgb": "images/004_rgb.png", "labels": []},
    {"id": "client1:hard:001", "client_id": "client1", "split": "hard", "rgb": "images/005_rgb.png", "labels": []}
  ]
}
```

Replace the schema paths and IDs with your dataset records before execution. `depth` and `nir` are optional fields. A trimodal run must explicitly require both modalities. Classes are zero-based nonnegative integers, and `class_names`, when given, bounds the valid class IDs.

Valid splits are `labeled`, `unlabeled`, `validation`, `test` and `hard`. Input aliases `label`, `unlabel` and `val` are accepted and normalized. Labeled/validation/test/hard rows require an explicit `labels` list; `[]` means a genuinely empty annotated image, not a missing annotation. Every box has finite, normalized `xyxy` coordinates in `[0,1]` and positive width/height. No automatic split, annotation inference or fallback ground truth is provided.

### Leakage boundaries

- IDs are unique across all clients and splits. Reusing an RGB path under another ID also fails validation.
- The ID must refer to the original synchronized acquisition, not a new ID per modality or augmentation. If duplicate files have different paths and IDs, the validator cannot infer their common content: data owners must retain the original acquisition ID or perform a content-level duplicate audit upstream.
- Server calibration IDs belong only to `validation`. They may remain in client validation pools as specified in the manuscript; they cannot belong to train/test/hard.
- Evaluation views refer to existing held-out IDs. Standard test IDs may be reused across Participating, Unseen and New-client without duplicating records. Hard IDs belong to `hard` and are disjoint from `test` by construction.
- `require_equal_standard_views=true` checks that the three provided standard views use identical identity sets, matching the primary protocol. Other benchmarks may use explicitly distinct held-out test subsets; the default does not silently force their identities to match.
- `load_sample()` always returns `targets=[]` for an `unlabeled` record, including when `include_labels=True`. Serialization removes optional labels from unlabeled records. Evaluation accepts `validation`, `test` or `hard` only.

### Private source-identifier key

Actual-data pretraining/training requires `TEFED_SOURCE_SECRET` in the process
environment, with at least 32 UTF-8 bytes. The runner validates it before creating
an output directory. It is used only at runtime for HMAC source/state audit keys
and is never added to configuration, logs or checkpoints. Generate it locally:

```powershell
$env:TEFED_SOURCE_SECRET = python -c "import secrets; print(secrets.token_hex(32))"
```

```bash
export TEFED_SOURCE_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Keep it outside the source repository and preserve it privately if exact source
identifier replay is needed. `experiment.test_mode=true` instead uses a fixed,
public test key and does not consume the private environment value.
This explicit test setting must not describe an actual-data experiment.

## Image and coordinate API

```python
from tefed.data import load_manifest, load_sample, map_boxes

manifest = load_manifest("manifests/tomato.json", strict_files=True)
row = manifest.by_split("labeled", client_id="client1")[0]
sample = load_sample(row, size=(640, 640), require_modalities=("depth", "nir"))
# sample["rgb"]: uint8 HWC3, channel order RGB
# sample["depth"]: float32 HW, original nonnegative sensor units; zero means invalid/padding
# sample["nir"]: uint8 HW
# sample["targets"]: list of {"box": normalized_xyxy_in_letterbox, "class_id": int}
# sample["meta"]: original_size, size, resized_size, scale, scale_xy, pad
original_boxes = map_boxes([[.1, .2, .5, .7]], sample["meta"], inverse=True)
```

Sizes use `(width,height)`. One resize/padding geometry applies to all modalities and boxes. RGB and NIR use bilinear interpolation; depth uses nearest-neighbor to preserve measured values. Rounded resize dimensions determine actual horizontal/vertical coordinate scales. The image library is loaded only when needed. Inputs must already be registered at RGB resolution; mismatched dimensions fail instead of being independently resized and treated as registered. Timestamps and geometric calibration cannot be established from filenames alone.

Depth scaling/valid ranges and NIR normalization require explicit dataset-specific preprocessing. The reader accepts depth as a single-channel image or a float `.npy` loaded with `allow_pickle=False` and preserves its sensor units. NIR accepts uint8 or a floating-point `.npy` already bounded to `[0,1]`, which is mapped to uint8 using the fixed factor 255. It does not estimate normalization from test images. Higher-bit-depth integer NIR requires documented preprocessing. Do not convert depth to 8-bit appearance imagery for training as if it retained metric values. Renderer-proxy NIR retains its proxy status after conversion.

### DroneVehicle boxes

`oriented_to_horizontal(corners, image_size=(width,height), normalize=True)` accepts four explicit `(x,y)` corners, or eight flattened pixel coordinates, and returns their minimum enclosing horizontal normalized box. It does not guess five-parameter angle conventions. Use identical conversion and class mapping for every comparator.

### Fixed Unseen perturbations

`apply_unseen(sample, identity=sample["id"], perturbation=config, seed=0)` uses a SHA-256-derived identity seed. Brightness/contrast, depth scale/noise, NIR brightness/noise and optional image-plane translation/flip are deterministic under the same ID/configuration/seed. Geometry changes apply to all modalities and targets together; fully out-of-frame targets are removed. This function never updates a client state, model weights or thresholds.

Perturbation strengths are set in the run configuration. Image-plane translation is a surrogate transformation, not evidence of physical camera-pose robustness. For a transformed view, compare predictions with that view's transformed `sample["targets"]` in the same coordinate system using `evaluate_detection()`. `evaluate_jsonl()` compares original-image coordinates with original manifest annotations; it must not be used unchanged for geometrically transformed targets. Photometric-only perturbations preserve original geometry after letterbox inversion.

## Detection metrics

```python
from tefed.metrics import evaluate_detection, evaluate_jsonl, aggregate_seed_metrics

targets = {"image1": [{"box": [.1, .2, .4, .6], "class_id": 0}]}
predictions = {"image1": [{"box": [.1, .2, .4, .6], "class_id": 0, "score": .9}]}
scores = evaluate_detection(targets, predictions)
# scores["AP50"] == scores["mAP50"], scores["AP75"], scores["mAP50_95"]
```

AP uses the monotone precision envelope sampled at 101 recall levels. Greedy IoU matching is independent for each image/class/threshold, and a GT box matches at most once. Predictions are ranked by descending score with deterministic identity/index tie order. At most 100 predictions per image/class are evaluated by default. The standard IoU grid is `0.50,0.55,...,0.95`; `mAP50_95` is returned only for that grid. This is COCO-style ordinary-box AP, not the complete COCO crowd/ignore, area-stratified or AR evaluation suite.

GT classes with no predictions receive AP=0. Classes with no GT have AP=null/None and are excluded from mAP; their false positives remain visible in `per_class`. All-GT-empty evaluation returns undefined AP, never a perfect score. JSONL prediction files contain one row per evaluated identity:

```json
{"id": "image1", "predictions": [{"box": [0.1, 0.2, 0.4, 0.6], "class_id": 0, "score": 0.9}]}
```

An image with no detections requires an explicit `"predictions": []` row. Missing, duplicate and extra IDs fail, preventing accidental evaluation on a favorable subset. Predictions for ordinary evaluation are in original normalized image coordinates.

Cross-seed summaries require an exact declared seed set and identical finite scalar metric fields. `aggregate_seed_metrics(results, expected_seeds=list(range(10)))` computes mean and sample SD with denominator `n-1`. It rejects missing/duplicate/extra seeds, mismatched metric names, null values and fewer than two seeds. It never fills missing seeds or turns within-round observations into independent repetitions.

## Prepare explicit manifests

```text
python scripts/prepare_manifest.py --input raw_manifest.jsonl --output manifests/validated.json --check-files
python scripts/prepare_manifest.py --input partitions.csv --output manifests/validated.json --input-format csv --check-files
python scripts/prepare_manifest.py --input instances.json --input-format coco --image-root images --client-id client1 --split test --output manifests/client1_test.json --check-files
python scripts/prepare_manifest.py --input labels --input-format dronevehicle --image-root images --class-map vehicle_classes.json --client-id client1 --split test --output manifests/client1_test.json --check-files
```

CSV columns are `id,client_id,split,rgb,depth,nir,labels`; the labels cell is a JSON list. COCO conversion maps sorted category IDs to zero-based class IDs and records that mapping. Crowd/ignore annotations raise an error because the ordinary-box metric does not implement their semantics. DroneVehicle text conversion accepts eight corner coordinates, category and an optional difficulty field per line; category mapping must be supplied explicitly. A nonzero difficulty flag requires an explicit `--difficult-policy include` or `drop`; neither is silently assumed. Record this policy consistently for all methods.

`--calibration-ids ids.json` and `--views views.json` provide explicit identity lists. `--require-equal-standard-views` enables the primary-protocol identity equality check. The script validates before atomically writing output, converts paths to absolute paths, and never randomly repartitions or fabricates annotations. For multiple clients/splits, combine the explicit record lists and validate the complete manifest together so that cross-partition leakage can be checked.

## Tests

Run `python -m unittest discover -s tests -p test_data_metrics.py -v`. Tests cover leakage rejection, calibration/test-view boundaries, unlabeled annotation isolation, synchronized letterbox/box transforms, deterministic Unseen noise, oriented boxes, 101-point AP, duplicate/false-positive/class matching, empty GT handling, prediction identity completeness and sample SD across declared seeds.
