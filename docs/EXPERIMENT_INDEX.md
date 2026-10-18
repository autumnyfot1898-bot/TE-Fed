# Experiment entry points

Use these entry points to train, compare and analyze TE-Fed. Prepare a shared initialization with the supervised command in [Getting Started](GETTING_STARTED.md), then train and evaluate local checkpoints. Dataset files, trained weights and run outputs are supplied or generated separately.

| Question | Configuration / entry point | Output |
| --- | --- | --- |
| Can the software execute? | `python -m tefed smoke --output runs/software_smoke` | Two native CPU rounds and four-view evaluation on generated test inputs |
| How do the registered methods compare? | `configs/tomato.toml` or `configs/dronevehicle.toml`; `scripts/run_suite.py --plan comparisons` | Resolved configuration, final metrics, local checkpoint |
| What changes when the verification route changes? | `scripts/run_suite.py --plan mechanisms`; stored/current/cyclic route settings | Per-variant verification audit and final metrics |
| What changes with matched/independent selection? | Mechanism plan plus `scripts/analyze_runs.py exposure` | Per-client, per-round selection and actual training counts |
| How do rendering budgets and reuse caps affect execution? | `scripts/run_suite.py --plan budgets` | Local run ledgers, retained/selected cases and metrics |
| What do the component controls change? | `scripts/run_suite.py --plan components` | Explicitly resolved component configuration and metrics |
| How are strict selector controls replayed? | `prepare_replay.py` and `run_suite.py --plan replay` | One immutable pool, bound protocol and checked actual per-client-round exposure |
| How are all eight cells executed? | `run_suite.py --plan factorial` | Operation/configuration/artifact bindings and completed-cell records |
| How is interrupted training continued? | `train --resume runs/seed0/resume.pt` | Verified next-round continuation without duplicate committed records |
| How do clients run through workers? | `python -m tefed.worker` and `communication.mode="http"` | Worker-local training and pooled evaluation records |
| How are multiple seeds summarized? | `scripts/analyze_runs.py summary --runs ... --seeds ...` | Means and sample SD across the exact declared seed set |
| How is renderer-reference agreement counted? | `scripts/analyze_runs.py process --run ...` | Separate case/event RTA, qualification and retention numerators and denominators |
| How are latency and power recorded? | `scripts/profile_inference.py`, `scripts/record_power.py` | Actual-device CSV records for the supplied hardware |

## Execution sequence

1. Prepare fixed data identities and sensor units: [DATA.md](DATA.md).
2. Train or supply a compatible local initialization; freeze the validation artifact: [CALIBRATION.md](CALIBRATION.md).
3. Configure external Blender assets and unit conversions: [RENDERING.md](RENDERING.md).
4. Execute a comparison plan or one selected method.
5. Read actual execution ledgers before describing training exposure as matched: [EXPERIMENTS.md](EXPERIMENTS.md).

The suite writes a plan unless `--execute` is supplied. Dynamic selector arms can develop different pools; an equal cap does not establish equal exposure or a fixed-pool experiment. Model weights are generated locally and are outside the public source package.

Executed software checks are recorded in [TEST_REPORT.md](TEST_REPORT.md), and device validation is recorded in [HARDWARE.md](HARDWARE.md). Report dataset performance, multiple-seed results and device resources from the corresponding completed runs.
