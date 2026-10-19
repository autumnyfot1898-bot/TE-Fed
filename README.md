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
  version = {2.6.2},
  url     = {https://github.com/autumnyfot1898-bot/TE-Fed}
}
```

License: [AGPL-3.0-only](LICENSE). [Third-party components](THIRD_PARTY.md).
