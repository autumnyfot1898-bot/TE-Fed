# Executable workflows

Use the existing project Python environment. These commands add no environment setup.

[Audit exports, review diagnostics, paired comparisons and figures](ANALYSIS_TOOLS.md) use the records produced by these workflows. [MANUSCRIPT_ALIGNMENT.md](MANUSCRIPT_ALIGNMENT.md) maps the revised manuscript contracts to their implementation.

## Software checks

```bash
python -m unittest discover -s tests -t . -v
python -m tefed.main smoke --output runs/software-smoke
```

The tests generate their own inputs. They cover native optimization, eight factorial cells, nonempty frozen-pool replay, round recovery and authenticated workers. Outputs are software evidence rather than dataset measurements. See [TEST_REPORT.md](TEST_REPORT.md).

## Continue an interrupted run

```bash
python -m tefed.main tefed --config configs/local.toml --seed 0 --output runs/seed0
python -m tefed.main tefed --config configs/local.toml --seed 0 --output runs/seed0 --resume runs/seed0/resume.pt
```

`resume.pt` commits completed rounds. Recovery checks the code, configuration, data, calibration, initialization and execution backend, validates committed output bytes and removes unfinished round outputs before continuing. Keep the same total round count and output directory. Completed runs return their existing result after integrity checks.

Snapshots preserve model/router weights, client states, persistent teachers, Local/FedBN state, pool/source states, previous aggregation direction and Python/NumPy/Torch RNG. Optimizers and GradScaler are recreated each round by the training protocol. Recovery starts at a round boundary, so an unfinished round can be recomputed. `final.pt` remains the initialization/evaluation checkpoint; it is not a resume snapshot.

## Frozen-pool exposure replay

Freeze a completed source run and record its actual per-client, per-round counts:

```bash
python scripts/prepare_replay.py freeze --run runs/pilot/seed0 --output private/frozen-pool
python scripts/prepare_replay.py protocol --pool private/frozen-pool/pool.json --runs runs/pilot/seed0 --output private/exposure.json
```

Add the local paths to the configuration:

```toml
[replay]
pool_manifest = "private/frozen-pool/pool.json"
exposure_protocol = "private/exposure.json"
arm = "matched"
```

```bash
python scripts/run_suite.py --config configs/local.toml --plan replay --output runs/replay --execute
```

Both arms read the same content-bound pool and retain its creation-round eligibility and diversity rules. The matched arm keeps the compatibility gate; the independent arm ranks by fidelity, priority and ID. Replay disables new case construction. It checks selection counts, correction presentations, attempted updates and applied updates against the same protocol. Insufficient eligible cases, changed assets, altered training settings or different AMP skips cause a failure. No below-gate filling, duplicate cases or reporting weights are used to force agreement.

## Eight factorial cells

```bash
python scripts/run_suite.py --config configs/local.toml --plan factorial --output runs/factorial --execute
```

Set `communication.mode = "local"` for this plan. The `tefed-factorial-v1` operations are:

| Factor | On | Off |
| --- | --- | --- |
| A | State-dependent reliability qualification | Posterior threshold 0.75, retaining validity checks and routed reliability weights |
| B | Source-evidence descriptors and stored-route verification | Generic HC priors and current-route verification |
| C | Complete correction path | Server-MoE without rendering, verification, insertion or reuse |

B has no training effect when C is off. The full `111` cell uses ordinary TE-Fed. The plan binds operation definitions, configuration, input artifacts and seeds; execution records every job's status. `factorial_<view>_<metric>.jsonl` can be read by the existing factorial analyzer.

## Continue a batch experiment

```bash
python scripts/run_suite.py --config configs/local.toml --plan factorial --output runs/factorial --execute --resume
```

Use `communication.mode = "local"` for batch recovery. Keep the original configuration, seeds, inputs and output directory. The runner validates the saved plan and every existing task before writing. Input binding covers every variant, including rendering enabled by a budget variant; identical input scopes share one plan-time scan. Completed tasks pass checkpoint and artifact checks and are skipped; unfinished tasks continue from their `resume.pt`. Attempt history is retained. Factorial metrics are rebuilt from committed task results, so retries do not duplicate rows. Use `--resume` when executing an existing dry-run plan as well.

## Client workers

Workers use local manifests containing their own client partitions and the same frozen calibration artifact. The coordinator manifest contains the logical identities and labels; client image paths need not exist there. Configure matching protocol/model settings on each node. Shared transport keys are read from the named environment variable and are never written into messages or checkpoints.

```bash
python -m tefed.worker --config configs/local-worker0.toml --client-id client0 --worker-id node0 --cache-dir runs/worker0 --port 8765
python -m tefed.worker --config configs/local-worker1.toml --client-id client1 --worker-id node1 --cache-dir runs/worker1 --port 8766
```

Coordinator configuration:

```toml
[communication]
mode = "http"

[communication.workers.client0]
url = "http://127.0.0.1:8765"
worker_id = "node0"
key_env = "TEFED_TRANSPORT_KEY"

[communication.workers.client1]
url = "http://127.0.0.1:8766"
worker_id = "node1"
key_env = "TEFED_TRANSPORT_KEY"
```

Run the normal training or saved-checkpoint evaluation command with this configuration. Client initialization, assessment, training, validation gains and held-out predictions execute at the worker. The coordinator retains construction, verification, aggregation and router updates. Server-generated correction images can be sent down to workers; private observation images are not returned.

Requests and responses bind the client, round, model bank, data, configuration and source code. Stable task identities support retries. Use matching CPU/CUDA backends and dependency versions across nodes, plus matching device counts for exact CUDA RNG transfer. An explicitly allowed build difference is recorded in the run.

Completed task results and round cursors are committed together to `tasks.sqlite3` under `--cache-dir`. Restart with the same cache directory, code, data, configuration and keys to reuse them. A repeated completed task returns its stored result without training again. A task interrupted before commit can be recomputed. Results remain on disk for later retries and include model updates and RNG state.

Non-loopback connections require TLS or a loopback tunnel. Router calibration and Local-only cold-start selection need independent server-held calibration observations. Prepare the shared checkpoint with local pretraining, then start the workers.
