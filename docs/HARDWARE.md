# Hardware environments

Use separate environments for the server, Jetson clients and CPU tests. The CPU lockfile selects CPU wheels.

## Installation targets

| Role | System | Python | PyTorch / torchvision | Wheel index |
| --- | --- | --- | --- | --- |
| RTX 5080 server | Windows or Linux x86_64 with a compatible NVIDIA driver | 3.12 | 2.7.1 / 0.22.1 | `cu128` |
| Orin Nano Super 8 GB client | JetPack 7.2.1, L4T 39.2.1, Ubuntu 24.04, CUDA 13.2.2 | 3.12 | 2.12.0 / 0.27.0 | `cu132` |

The server pair is listed in the [official PyTorch installation table](https://pytorch.org/get-started/previous-versions/). PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels. [PyTorch release notes](https://pytorch.org/blog/pytorch-2-7/)

As checked on 2026-10-18, NVIDIA lists JetPack 7.2.1 for the Orin family. Its ISO uses the Orin Nano Super configuration by default. The client pair follows NVIDIA's demonstrated upstream CUDA 13.2 installation on Orin. [JetPack release](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1), [NVIDIA PyTorch guidance](https://forums.developer.nvidia.com/t/how-do-i-correctly-install-pytorch-on-jetpack-7-2/372773/5)

These are installation targets. A successful on-device check is required before reporting hardware compatibility or experiment results.

## RTX 5080 server

Create and activate a Python 3.12 virtual environment, then run:

```bash
python -m pip install -r requirements/rtx5080.txt
python -m pip install -e ".[train]"
python -m pip check
python scripts/check_environment.py --profile rtx5080 --device cuda:0 --require-cuda --smoke --output runs/rtx5080-environment.json
python -m tefed.main --help
python -m tefed.main smoke --device cuda:0 --amp --output runs/rtx5080-smoke
```

Use `cuda:0` in the experiment configuration. A locally installed CUDA toolkit alone does not make a CPU-only PyTorch wheel GPU-enabled.

## Orin Nano Super client

First inspect the existing device:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack
python3 --version
```

For this route, the release line must identify `R39` and `REVISION: 2.1`. The JetPack metapackage may be absent even when L4T is installed; retain the release line in the environment report. If the device uses an older BSP, follow [NVIDIA's Orin Nano setup guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/index.html) before preparing this environment. Repository scripts do not flash the device or change drivers.

On a device already running JetPack 7.2.1:

```bash
python3 -m venv .venv-jetson
source .venv-jetson/bin/activate
python -m pip install -r requirements/jetson-jp72.txt
python -m pip install -e ".[train]"
python -m pip check
python scripts/check_environment.py --profile jetson --device cuda:0 --require-cuda --smoke --output runs/jetson-environment.json
python -m tefed.main --help
python -m tefed.main smoke --device cuda:0 --amp --output runs/jetson-smoke
```

The project retains `ultralytics==8.4.0`. Install the matching PyTorch pair first and avoid package upgrade commands that replace it. Do not run `uv sync --extra cpu` in either GPU environment.

Use a fresh output directory for every fixture run. The short AMP fixture uses initial loss scale 128; regular training defaults to 65536. Overflowed steps are skipped and counted separately from applied optimizer updates.

JetPack 6 uses a different BSP, CUDA stack and Python ABI. The `cu132` commands above do not apply to that system. Older JetPack installations require a separately matched NVIDIA wheel or container and are outside this installation target. [NVIDIA Jetson PyTorch installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

## Environment checks

`check_environment.py` reports Python, machine architecture, L4T, the installed JetPack metapackage, package versions, PyTorch's CUDA build and visible GPUs. It performs no downloads or installations. `--smoke` checks NumPy conversion, convolution forward/backward and torchvision NMS on the selected device. A failed import, profile mismatch or kernel check returns a nonzero exit code.

The Jetson profile requires Linux aarch64, L4T 39.2.x, an Orin GPU with compute capability 8.7 and a CUDA 13.2+ PyTorch build. The RTX 5080 profile requires that GPU and a CUDA 12.8+ build. Metadata without `--smoke` establishes identification only.

For the locked CPU environment:

```bash
uv run --locked --extra cpu python scripts/check_environment.py --profile cpu --device cpu --smoke
```

Run the configured data and model workflow on each physical device and retain its environment report with the resulting artifacts. The Orin Nano has shared memory; begin device validation with a small local batch and record any configuration change separately from the manuscript protocol.

## Executed validation

See [TEST_REPORT.md](TEST_REPORT.md) for V2.6 CPU/RTX 5080 regression counts and source-bound logs. HTTP checks use loopback workers with independent local data roots.
