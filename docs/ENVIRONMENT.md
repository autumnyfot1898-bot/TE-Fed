# Environment and validation scope

## Locked CPU environment

Run commands from the repository root with Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --extra cpu
uv lock --check
uv run --locked --extra cpu python -m tefed.main --help
uv run --locked --extra cpu python -m tefed.main methods
uv run --locked --extra cpu python -m unittest discover -s tests -t . -v
uv run --locked --extra cpu python -m tefed.main smoke --output runs/software-smoke
```

Use a fresh output directory for each smoke run. `smoke` creates generated images, performs two CPU training rounds and evaluates four views.

`uv.lock` records the resolved packages and hashes. The `cpu` extra fixes PyTorch 2.5.1, torchvision 0.20.1 and Ultralytics 8.4.0; PyTorch and torchvision use the official CPU index on Windows and Linux. The adapter requires the exact Ultralytics version. The project metadata currently supports Python 3.12. Other Python versions are outside this release's supported range.

`uv run --locked` validates the lockfile rather than silently changing it. `--extra cpu` must remain present on the CPU commands. See the official [locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/) and [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Dataset training

The method-first entry point dispatches to the existing training command:

```bash
uv run --locked --extra cpu python -m tefed.main tefed --config configs/tomato.toml --seed 0 --output runs/tefed-seed0
```

This command requires locally supplied data, a local initialization checkpoint and a fixed calibration artifact. Set a private `TEFED_SOURCE_SECRET` as described in the [actual-data setup](GETTING_STARTED.md#actual-data-experiments) before training. Update the paths in your configuration and set `experiment.device = "cpu"` for the CPU environment. The example configuration uses `cuda:0` for a separately prepared GPU environment. External Blender data and its executable are configured separately.

`python -m tefed.main METHOD ...` is equivalent to `python -m tefed train --method METHOD ...`. Regular commands (`pretrain`, `calibrate`, `evaluate`, `validate`, `methods`, `smoke`) also pass through this entry point. A method-first invocation rejects a second `--method` argument.

## RTX 5080 and Jetson environments

Create and activate a separate Python 3.12 environment for each hardware target. The [hardware guide](HARDWARE.md) lists the matching systems and package pairs. For RTX 5080:

```bash
python -m pip install -r requirements/rtx5080.txt
python -m pip install -e ".[train]"
python scripts/check_environment.py --profile rtx5080 --device cuda:0 --require-cuda --smoke
python -m tefed.main --help
python -m tefed.main smoke --device cuda:0 --amp --output runs/rtx5080-smoke
```

Use ordinary `python -m tefed.main ...` in that environment. Do not run `uv sync --extra cpu` there: that command installs CPU packages. The `train` extra retains the installed matching pair and pins `ultralytics==8.4.0`.

For Jetson Orin Nano Super, the current installation target is JetPack 7.2.1 / L4T 39.2.1 with Python 3.12 and the pair in `requirements/jetson-jp72.txt`. Follow the device checks in [HARDWARE.md](HARDWARE.md) before installing. Those wheels do not apply to JetPack 6.

The CUDA fixture uses the same generated inputs as the CPU fixture. It exercises native detector training, checkpoint reload and four-view evaluation. Rendering and router optimization are disabled in this fixture.

The short AMP fixture sets `optimizer.amp_init_scale = 128`. Regular training defaults to 65536; a finite positive override is supported. GradScaler skips overflowed updates and reduces its scale, while nonfinite losses and non-AMP gradients raise an error. Round logs distinguish attempted, skipped and applied updates.

## Optional Docker image

The Dockerfile targets a CPU software-test image and pins uv 0.13.1:

```bash
docker build -t tefed:cpu .
docker run --rm tefed:cpu methods
docker run --rm -v "$PWD/runs:/app/runs" tefed:cpu smoke --output runs/software-smoke
```

The volume example uses Bash syntax. On PowerShell, use `-v "${PWD}/runs:/app/runs"`; create `runs` first. The container contains source and tests; supply datasets, weights, Blender and scene assets separately.

## Make and CI

The optional GNU Make targets expand to the same uv commands: `make install`, `make check`, `make test`, `make smoke`, `make methods`, and `make train CONFIG=... SEED=... OUTPUT=...`. The underlying uv commands can also be run directly.

The GitHub Actions workflow checks the lockfile and undefined Python names with Ruff 0.17.1, installs the locked CPU environment, checks the public help entry point and runs the software tests on Ubuntu. It runs on pushes and pull requests without repository secrets.

## Executed checks

V2.6 checks use the existing CPU and RTX 5080 environments. Counts, source identities and logs are in [TEST_REPORT.md](TEST_REPORT.md) and [project_validation.json](project_validation.json).
