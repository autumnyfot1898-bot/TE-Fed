# Executed verification: TE-Fed V2.6

Package version: `2.6.1`.

## 2.6.1 maintenance checks - 19 October 2026

The first GitHub Actions Linux CPU run for V2.6.0 executed 304 tests: 300 passed, one failed and three CUDA-only tests were skipped. The transport tamper-check fixture selected an image by filesystem enumeration order, which could select another client's input. It now selects a labeled client 0 image from worker 0's manifest and checks its partition digest before and after tampering.

The corrected transport suite passed all seven tests. Full CI records are available under [Software tests](https://github.com/autumnyfot1898-bot/TE-Fed/actions/workflows/tests.yml).

## 2.6.0 full verification - 18 October 2026

Checks used the existing Python 3.12.14 CPU and RTX 5080 environments. Source identities, commands and log hashes for V2.6.0 are in [project_validation.json](project_validation.json).

| Check | Observed result |
| --- | --- |
| Full CPU suite | 304 tests: 301 passed, 3 CUDA-only skips; 234.752 seconds |
| Full RTX 5080 environment suite | 304 passed, 0 skips; 277.836 seconds |
| Source checks | 74 Python files parsed and bound by SHA-256; state/cues unchanged; core audit equivalence checked |
| Static checks | Locked package metadata and Ruff F821/F822/F823 passed |

Both full suites ran on the final V2.6.0 source. Logs: [CPU](cpu_test_output.txt), [CUDA environment](cuda_test_output.txt). Temporary profile paths are normalized in public logs; raw-capture hashes are retained in the JSON.

## Coverage

- Native detection and auxiliary-loss gradients, missing and all-zero sensors, BN restoration and AMP update accounting.
- All eight factorial cells, 111 equivalence, nonempty frozen-pool replay and round recovery.
- Independent worker image roots, pooled evaluation and saved-checkpoint evaluation at seed 3.
- Worker restart reuse, synchronized process contention, process-crash rollback and cache integrity checks.
- Batch resume, completed-task skipping, attempt history and interrupted factorial metric publication.
- Rendering input binding across all batch variants, generated-data provenance and model/manifest class checks.
- Per-case presentations, supervised presentations and actual attempted/applied/AMP-skipped steps.
- Local and remote evaluation identity manifests, explicit metadata and transform/model/state bindings.
- Full fidelity audit fields, early-gate failures and exact equality of 512 fidelity plus 4096 verification decisions with the prior V2.6 package.
- Case-chain exports, actual-result plots, fixed-identity qualitative panels and independent-review reliability/routing/paired analyses.
- Native two-method, two-round reporting: 20 candidates, four client-round audit units, four loss curves, 24 metric groups and two fixed held-out identities with prediction-file/transform bindings.

## Inputs and execution

Native round checks use generated 64-pixel YOLO26n inputs marked `data_origin = "generated"`. Small round fixtures disable new rendering and router optimization. Renderer-input binding and server-case downlink have separate tests. HTTP checks use loopback workers with independent local data roots.

Recovery uses committed round snapshots with matching code, inputs, configuration and backend. Worker results and round cursors persist together; tasks interrupted before commit can be recomputed. Set `communication.mode = "local"` for factorial execution and batch recovery. Commands are in [WORKFLOWS.md](WORKFLOWS.md).

The CPU pair is torch 2.5.1+cpu / torchvision 0.20.1+cpu; the CUDA pair is torch 2.7.1+cu128 / torchvision 0.22.1+cu128. Both use Ultralytics 8.4.0.
