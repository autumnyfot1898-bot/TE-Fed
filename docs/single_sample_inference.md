# Single-sample inference

The `infer` command runs the existing TE-Fed candidate assessment path on one RGB, depth, and NIR observation. It uses no ground-truth labels, performs no training or calibration, and leaves the model, router, and selected client state unchanged. It exports predictions and a display overlay rather than computing AP.

## Required files

- A compatible TE-Fed checkpoint produced by the reference implementation, including the detector, router, configuration, and client states. A plain YOLO weights file is not sufficient.
- The frozen calibration JSON associated with that training setup. Calibration is not embedded in the current checkpoint format. The file must contain an explicit positive `T_C`; other calibration fields follow the existing candidate-assessment implementation.
- Registered RGB, depth, and NIR inputs of the same original image dimensions, prepared with the preprocessing used for training.
- The sample ID and a client ID present in the checkpoint. The sample ID contributes to the deterministic reference-augmentation seed.

This entry point supports the tomato TE-Fed configuration with five specialist heads and auxiliary modalities enabled. General-only and RGB-only controls use different input policies and are not presented as trimodal inference here.

## Original data and model inputs

The files bundled in `examples/tomato_trimodal/inputs/` document the original input modalities. They are preserved for inspection, including 16-bit depth and NIR. The display images in `images/` are for viewing.

The current model reader accepts:

| Input | Accepted representation |
|---|---|
| RGB | Image converted to 8-bit RGB by the existing reader |
| Depth | Single-channel finite nonnegative sensor values; the checkpoint's `model.depth_scale` is applied later |
| NIR | Single-channel uint8, or a floating-point `.npy` array with values in `[0,1]` |

Use the fixed NIR preprocessing recipe from training when starting with a higher-bit-depth sensor image. This command rejects raw 16-bit NIR instead of guessing a scaling rule. Per-image display contrast limits are not a training recipe.

## Run

Install the CPU dependencies, or use your existing compatible training environment:

```bash
uv sync --locked --extra cpu
```

From the repository root:

```bash
uv run --locked --extra cpu python -m tefed infer \
  --checkpoint /path/to/final.pt \
  --calibration /path/to/calibration.json \
  --rgb /path/to/prepared/rgb.png \
  --depth /path/to/prepared/depth.png \
  --nir /path/to/prepared/nir.png \
  --sample-id YOUR_SAMPLE_ID \
  --client-id YOUR_CLIENT_ID \
  --device cpu \
  --display-confidence 0.25 \
  --output runs/single_sample
```

The output directory must not exist. No renderer, retained pool, or optimizer is needed for this entry point. The detector and decision settings come from the checkpoint configuration; only the device and display options are supplied here. Missing client states cause an error rather than a silent prior-state fallback.

Use `--class-names /path/to/classes.json` to supply your training label mapping:

```json
{"0": "your_first_class", "1": "your_second_class"}
```

Without that mapping, labels are shown as `class_0`, `class_1`, and so on. Class names are not guessed from the task name.

## Outputs

| File | Contents |
|---|---|
| `predictions.json` | All candidates returned by the existing assessment path, including class IDs, scores, routes, and both letterbox and original-image coordinates |
| `rgb_with_detections.png` | The original RGB image with predictions above `--display-confidence` drawn on it |
| `provenance.json` | File hashes, checkpoint configuration source, selected frozen state, display settings, coordinate transform, and whether the checkpoint is a software test |

The original `box` field remains normalized xyxy in letterbox coordinates. Explicit `box_original_normalized_xyxy` and `box_original_xyxy` fields provide the inverse-mapped normalized and pixel coordinates. The display threshold affects the overlay only; it does not remove candidates from the JSON or change the checkpoint's candidate-generation threshold.

`--round-id` defaults to `0`, matching the reference-augmentation context used by the current `evaluate` entry point. It is not the number of completed training rounds. If you override it, the value is recorded. The checkpoint's training-round metadata is recorded separately.

The command hashes the calibration file but does not independently establish its historical association with the checkpoint. Supply the calibration used for that setup. It also verifies that model/router parameters and buffers, the selected state, and the input files remain unchanged during inference.

Software-fixture checkpoints are marked `software_test_only` in the outputs. Their generated examples are used to test the interface and are not included as detector results in the public input gallery. This addition does not reproduce the paper's full-test-set metrics or include trained experiment weights.
