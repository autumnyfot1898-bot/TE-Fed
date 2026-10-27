# TE-Fed V2.6

[![uv](https://img.shields.io/badge/uv-environment-6E56CF)](https://docs.astral.sh/uv/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

Reference implementation of **TE-Fed: Correction with Source Evidence and Reuse by State Matching for Federated Semi-Supervised Multisensor Object Detection**.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="TE-Fed overview">
</p>

# Getting Started

## uv

```bash
git clone https://github.com/autumnyfot1898-bot/TE-Fed.git
cd TE-Fed
uv sync --locked --extra cpu

uv run --locked --extra cpu python -m tefed --help
```

## Docker

```bash
git clone https://github.com/autumnyfot1898-bot/TE-Fed.git
cd TE-Fed
docker build -t tefed .
docker run --rm tefed --help
```

# Examples

## Example inputs

RGB, depth, and near-infrared (NIR) inputs from one example scene are shown below. The [example folder](examples/tomato_trimodal/README.md) includes the original files; depth and NIR remain 16-bit PNGs.

![RGB, depth and NIR inputs](examples/tomato_trimodal/images/rgb_depth_nir.png)

## Preview command

Generate the input preview without a checkpoint or the training dependencies:

```bash
uv sync --locked
uv run --locked python -m tefed preview --example examples/tomato_trimodal --output runs/tomato_preview
```

The command saves the three display images, `rgb_depth_nir.png`, and `preview.json`. Use a new output directory for each run. Display mappings are separate from model input preprocessing.

## Single-sample inference

Inference uses a compatible tomato TE-Fed checkpoint, its frozen client state, and the corresponding calibration file. Supply your own checkpoint; trained weights are not included.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python -m tefed infer \
  --checkpoint /path/to/final.pt \
  --calibration /path/to/calibration.json \
  --rgb /path/to/prepared/rgb.png \
  --depth /path/to/prepared/depth.png \
  --nir /path/to/prepared/nir.png \
  --sample-id YOUR_SAMPLE_ID --client-id YOUR_CLIENT_ID \
  --device cpu --output runs/tomato_inference
```

Replace the paths and IDs with your files and the client ID stored in the checkpoint. This saves `predictions.json`, `rgb_with_detections.png`, and `provenance.json`. See [single-sample inference](docs/single_sample_inference.md) for the input contract, class names, and frozen-state behavior. In particular, raw 16-bit NIR requires the fixed preprocessing used for training; the contrast-stretched preview image is not a model input substitute.

# Configuration

Settings and paths are defined in [configs/tomato.toml](configs/tomato.toml) and [configs/dronevehicle.toml](configs/dronevehicle.toml). Set the local paths before running.

```bash
uv run --locked --extra cpu python -m tefed --help
```

# Citation

Software citation metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@software{tefed_2026,
  author  = {Sun, Pinghan and Jin, Wenquan},
  title   = {{TE-Fed}: Correction with Source Evidence and Reuse by State Matching
             for Federated Semi-Supervised Multisensor Object Detection},
  year    = {2026},
  version = {2.6.3},
  url     = {https://github.com/autumnyfot1898-bot/TE-Fed}
}
```

License: [AGPL-3.0-only](LICENSE). [Third-party components](THIRD_PARTY.md).
