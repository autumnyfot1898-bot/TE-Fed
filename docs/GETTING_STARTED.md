# TE-Fed experiments

TE-Fed trains a shared detector and specialist heads with source-evidence verification and state-matched correction reuse.

See [METHOD_CONTRACT.md](METHOD_CONTRACT.md) for the method, [TRAINING.md](TRAINING.md) for optimization, and [IMPLEMENTATION_CHOICES.md](IMPLEMENTATION_CHOICES.md) for implementation interfaces.

## What is included

- YOLO26 general detector, shared trunk/neck and five specialist heads.
- Candidate routing, fusion, measurable reliability cues and separate qualification/correction decisions.
- Descriptor priorities, deterministic rendering allocation, frozen source-route verification and five retention gates.
- State-matched reuse with next-round eligibility, recorded-route training and the original mean-reliability aggregation factor.
- Server router probes with complete-bank load balancing.
- Data manifests, calibration, four evaluation views, AP50/AP75/mAP50–95, seed summaries and work-exposure checks.
- Local-only, FedAvg, FedProx, FedBN, confidence self-training, and explicitly documented detector adaptations of Consistent-Teacher/FedAWA, FedSTO and FSSLOD, plus the two internal controls.
- External Blender scene rendering, synchronized RGB/depth and an explicitly labelled NIR/thermal intensity proxy. The tomato scene is not bundled. A class-map adapter supports other user-supplied scenes.
- Native CPU integration tests, unit tests and a CI workflow. See [TEST_REPORT.md](TEST_REPORT.md) for the executed checks.

The coordinator schedules clients sequentially through local tasks or authenticated HTTP workers. Workers load their own observations and return updates, descriptors and evaluation records. Resource fields describe the executed mode; deployment energy and latency require measurements on the actual devices. [WORKFLOWS.md](WORKFLOWS.md) covers workers, round recovery, frozen-pool replay and factorial execution.

## Install

This release targets Python 3.12. The locked CPU setup is documented in the [root README](../README.md) and [ENVIRONMENT.md](ENVIRONMENT.md). For a separate pip environment, install a matching PyTorch/torchvision pair first, then install this checkout:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train]"
python -m unittest discover -s tests -t . -v
```

The CPU pair above is the tested CPU environment. Use the RTX 5080 and Jetson setup in [HARDWARE.md](HARDWARE.md) for GPU execution and device validation status.

The detector adapter requires **Ultralytics 8.4.0** and fails explicitly on another version. Editable installation is used because the Blender entry scripts live in `scripts/`.

## Actual-data experiments

Set a private source-identifier key in the active shell before pretraining or
actual-data training. The runtime requires at least 32 UTF-8 bytes and never
writes this key to configuration, logs or checkpoints. Generate a random value
locally; keep the same private key when exact identifier replay is required.

PowerShell:

```powershell
$env:TEFED_SOURCE_SECRET = python -c "import secrets; print(secrets.token_hex(32))"
```

bash:

```bash
export TEFED_SOURCE_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Do not commit the value. Explicit `experiment.test_mode=true` uses a fixed
test key instead; dataset training requires the private key.

1. Prepare a manifest with the fixed labeled, unlabeled, validation, test and hard identities. Do not create a new random split when comparing methods. Include the fixed server calibration IDs. See [DATA.md](DATA.md).
2. Copy `configs/tomato.toml` or `configs/dronevehicle.toml` to `configs/local.toml`. Set real file paths, device, sensor units and assets; use `experiment.device = "cpu"` for the locked CPU environment. `model.depth_scale` is a fixed **network divisor**. `renderer.depth_scale` is an optional **scene-to-dataset unit conversion**; they have different meanings. Unknown rendered depth units are treated as missing.
3. Provide a model-schema-2 initial checkpoint generated with this version. The command below prepares one using supervised FedAvg on local labeled partitions only. Use the same initial checkpoint for every compared method. Earlier checkpoints require regenerated pretraining and calibration; see [TRAINING.md](TRAINING.md#checkpoint-schema).
4. Build and freeze a calibration artifact from the specified validation identities. The calibration perturbations are documented in [CALIBRATION.md](CALIBRATION.md).

```bash
python -m tefed validate data/tomato/manifest.json --check-files
python -m tefed pretrain --config configs/local.toml --rounds 10 --seed 0 --output runs/initial_detector
python -m tefed calibrate --config configs/local.toml --checkpoint runs/initial_detector/final.pt --output data/tomato/calibration.json
# Set model.initial_checkpoint and experiment.calibration in configs/local.toml.
python -m tefed train --config configs/local.toml --method tefed --seed 0 --output runs/tefed_seed0
python -m tefed evaluate --config configs/local.toml --checkpoint runs/tefed_seed0/final.pt --output runs/tefed_seed0_recheck
```

Calibration needs usable detection matches. It aborts when temperature cannot be identified, when a required cell is missing, or when measurements are nonfinite. It does not fill missing cells with favorable values.

The tomato renderer expects ripe/non-ripe classes. Other tasks require an explicit `scene_class_map` and suitable assets. A five-class geometry test checks that interface; vehicle assets must be supplied and validated separately.

## Comparisons and controls

```bash
python -m tefed methods
python scripts/run_suite.py --config configs/local.toml --plan comparisons --output runs/comparisons
python scripts/run_suite.py --config configs/local.toml --plan mechanisms --output runs/mechanisms --execute
```

Without `--execute`, the suite writes/prints its plan. The plans use explicit operations: stored/current/cyclic verification routes; matched/independent selection; rendering budgets; reuse caps; no alignment; RGB-only; and correction supervision set to zero while retaining the rendering path.

Compare actual correction presentations and optimizer steps for every client and round before describing selector arms as having equal training exposure. Reporting means do not establish that equality. See [EXPERIMENTS.md](EXPERIMENTS.md).

## Outputs

Each run writes its resolved configuration, calibration, environment, model capacity, candidate audit, verification decisions, round-by-round optimization counts, final checkpoint, frozen states, per-image predictions and metrics. `final.pt` for Local-only contains every client model; FedBN checkpoints include client BN states. Evaluation pools predictions before computing AP, rather than averaging client APs.

The task configurations set `evaluation.monitor_every=1` to write read-only metrics and predictions after each round. Set it to `0` to disable monitoring. Monitoring never updates model/state, chooses an early stopping round, or changes training. Its duration is recorded separately from the round's training/verification window. Final comparisons use the declared final round.

Renderer-target observations are logged for retained cases and once per selected `(client, round, case)` event before local epochs. The current round-start bank uses the recorded heads for selected-event observations. These internal reference measurements do not replace an independent label-quality audit.

Conditional class posterior, native detector confidence, reliability, fidelity and compatibility are separate quantities. AP ranking uses native detector confidence, fused across matched experts by their route weights. Fidelity/compatibility do not add an undocumented correction loss weight.

## Project layout

```text
TE-Fed/
├── assets/overview.png       # TE-Fed overview
├── configs/                  # Portable TOML templates
├── docs/                     # Preparation, protocol and implementation guides
├── scripts/                  # Experiment suites, rendering and analysis
├── src/tefed/                # Method implementation and command entry points
├── tests/                    # Software checks
├── .github/workflows/        # Source validation workflow
├── .python-version
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Makefile
├── CITATION.cff
└── README.md
```

## Software checks

```bash
python -m tefed smoke --output runs/software_smoke
```

This creates small generated images and calibration with `data_origin = "generated"`. It runs two native CPU rounds, saves a checkpoint, and evaluates four views. Generated artifacts require explicit `test_mode` and provide software-execution metrics.

## License and sources

This project is released under **AGPL-3.0-only**, consistent with its Ultralytics dependency. See [LICENSE](../LICENSE) and [THIRD_PARTY.md](../THIRD_PARTY.md). Raw data, trained weights, local scene assets and experiment run outputs are not included in the source archive. Software-test evidence is included in `docs/`. Data and asset availability and permissions are separate from the code license.
