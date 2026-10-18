# Experiment, training exposure and resource analysis

TE-Fed analyzes completed runs, training ledgers and collected device records. Hand-calculated test inputs check the analysis routines; they are not detection-performance results or device measurements.

## 1. Ten-seed summaries of completed runs

Each run must contain:

- `environment.json`: the actual seed, environment and manifest hash.
- `configuration.json`: the parameters used for the run.
- `rounds.jsonl`: execution counts for each client and round.
- `metrics.json`: actual evaluation of the final checkpoint in each view.

```powershell
python scripts/analyze_runs.py summary --runs runs/tefed/seed_0 runs/tefed/seed_1 runs/tefed/seed_2 runs/tefed/seed_3 runs/tefed/seed_4 runs/tefed/seed_5 runs/tefed/seed_6 runs/tefed/seed_7 runs/tefed/seed_8 runs/tefed/seed_9 --out analysis/tefed_summary.json
```

By default, the runs must contain exactly seeds 0–9. Missing, duplicate or extra seeds, undefined metrics and missing rounds raise errors. Means and SDs use the recorded values, with sample standard deviation computed using `ddof=1`; a single result is not copied into ten seeds. All seeds must use the same recorded manifest. To use another seed set for debugging, pass `--seeds` explicitly. The output still records the actual n.

## 2. Actual training exposure under state-based selection

```powershell
python scripts/analyze_runs.py exposure --left runs/matched/seed_0 --right runs/independent/seed_0 --require-match --out analysis/exposure_seed0.json
```

Three quantities are compared for each `client_id × round_index`:

| Count | Source |
|---|---|
| correction presentations | `clients[client].presentations.correction`, the number of actual training presentations |
| optimizer steps | `clients[client].optimizer_steps`, the number of updates actually applied |
| selection count | The number of cases in `reuse_selected_ids[client]` |

The records are reported as matched only when all three counts agree for every unit. Global totals, reporting weights or an aggregate mean do not establish matching at each unit. With `--require-match`, a discrepancy returns a nonzero exit code and the output retains the specific differences. Missing execution counts cause failure; optimizer steps are not inferred from epoch counts or case counts.

When AMP is used, the trainer should record attempted steps, AMP-skipped steps and actually applied steps separately. If an older ledger lacks `optimizer_steps_semantics`, the tool compares only its recorded values; this does not establish that the older implementation correctly excluded updates skipped by AMP. Set `optimizer_steps_semantics="actual_applied_updates"` only when the implementation actually uses that counting rule. Matching counts alone also do not establish identical initialization, data partitions or other control conditions.

## 3. Mechanism studies and ablations

Record the following executed operations for each mechanism study:

1. `stored/current/cyclic` changes only the verification reference, preserves the original source route, and fixes the rendering budget and case-availability timing.
2. The current matched/independent scripts each perform dynamic training and case construction. Earlier selections can change subsequent models and pools. Identical caps and diversity constraints do not establish a shared-pool or equal-exposure control. The tool reports actual training presentations, updates and selections for every unit, including any mismatches. Use the separate `--plan replay` workflow for an immutable pool and bound per-client-round exposure counts. It rejects insufficient eligibility, changed inputs and unequal applied updates; see [WORKFLOWS.md](WORKFLOWS.md#frozen-pool-exposure-replay).
3. `beta=0` disables only the correction-supervision loss. Rendering, verification, the pool and selection remain active, with their associated costs.
4. Disabling the entire correction path must explicitly disable descriptors, rendering, verification, pool insertion and selection. This differs from the beta-zero control.
5. RGB-only explicitly disables auxiliary modalities and the cross-modal loss, and renormalizes the remaining reliability terms as specified. Feeding only RGB to the detector while retaining IR auxiliary evidence is a separate task setting.
6. No alignment explicitly sets rho to 0 and omits AM. Record the normalization actually used.

`scripts/run_suite.py --plan factorial --execute` runs all eight `tefed-factorial-v1` cells and writes their operation definitions, configuration/artifact hashes and completion records. The full 111 cell matches ordinary TE-Fed. B has no training effect when C is off. See [WORKFLOWS.md](WORKFLOWS.md#eight-factorial-cells) before computing effects.

```powershell
python scripts/analyze_runs.py factorial --records new_factorial.jsonl --definitions executed_operations.json --metric value --scale 100 --out analysis/factorial.json
```

Each input row contains `seed`, `factors:{A:0/1,B:0/1,C:0/1}` and the metric. Every seed must have all eight cells without duplicates. `executed_operations.json` must specify the actual `on` and `off` operations for A/B/C. Main effects and interactions are computed within each seed before aggregation across seeds, using coefficients of 1/4 for main effects, 1/2 for two-way interactions and 1 for the three-way interaction. Scaling by 100 only converts differences on a 0–1 scale into percentage points. The tool does not automatically reverse the sign of boundary error or interpret differences as causal effects.

## 4. Actual power sampling

Example for the server GPU:

```powershell
python scripts/record_power.py --backend nvidia-smi --node serverGPU --gpu 0 --interval 1 --duration 600 --out telemetry/server_gpu.csv
```

Run the sampler separately on each Jetson. Determine the sysfs path and source units from the actual device:

```text
python scripts/record_power.py --backend sysfs --node J1 --power-path /sys/ACTUAL_BOARD_POWER_PATH --unit mW --scope jetson_board --interval 1 --duration 600 --out telemetry/J1.csv
```

The sampler only reads `nvidia-smi` or the sysfs file explicitly supplied by the user; it does not change device configuration. An unavailable GPU, N/A telemetry, a missing sysfs path or unspecified units causes an error rather than a replacement power value of 0. `--unit` accepts only W/mW/uW and explicitly converts the readings to W. A CPU/GPU sub-rail must not be treated as Jetson whole-board power. Server `power.draw` covers only the GPU board, excluding the server CPU and wall-power losses.

The output CSV columns are `ts,power_w,memory_mib,node`. GPU memory is reported in MiB; the field is left empty when Jetson memory is not measured. Each CSV has a `.csv.manifest.json` recording the scope, units, node, sampling interval, data source and completion status. Interrupted or failed collection is not marked as complete. Sampling must extend across all round boundaries, and the devices' UTC clocks must share a consistent actual time reference.

## 5. Energy integration across five devices

```powershell
python scripts/analyze_runs.py energy --power telemetry/J1.csv telemetry/J2.csv telemetry/J3.csv telemetry/J4.csv telemetry/server_gpu.csv --rounds runs/tefed/seed_0/rounds.jsonl --interval 1 --out analysis/energy_seed0.json
```

Each round must have actual boundaries in UTC seconds, `start_ts/end_ts` or the accepted aliases `utc_start_s/utc_end_s`. A duration in `seconds` alone cannot align power samples. Each device's sample times must be strictly increasing and cover both ends of the round. The tool interpolates boundaries only within the observed sample span and integrates by the trapezoidal rule, W×s=J. It does not extrapolate or infer energy from peak power or mean durations.

By default, J1, J2, J3, J4 and serverGPU must all be present. A sampling gap greater than 1.5 times the specified interval for any node causes the affected round to be rejected, so a missing sample is not treated as a complete interval. Missing device data or actual round boundaries are reported as missing; whole-system energy is not set to 0. The integration scope is the four Jetson boards plus the server GPU, excluding the server CPU and wall-power losses.

Record J1–J4 telemetry on the corresponding Jetson boards and retain each node's identity in its sampling manifest.

## 6. Request-latency P95

The raw CSV must contain `node,latency_ms`, with 200 actual requests for each of the four nodes:

```powershell
python scripts/analyze_runs.py latency --csv telemetry/request_latencies.csv --out analysis/latency_seed0.json
```

Compute P95 separately for each node, then take the maximum across the four nodes. Do not pool all 800 requests into one P95 or average across seeds before taking the maximum. By default, exactly 200 records are required for each node. A sampling protocol with more than 200 records must be specified explicitly; 200 records must not be selected from a larger set. Percentiles use NumPy's linear method, which is declared in the output.

## 7. RTA and process proportions with their respective denominators

```powershell
python scripts/analyze_runs.py process --run runs/tefed/seed_0 --out analysis/process_seed0.json
```

`case_event_proportions` takes retained records, selected events and the verification audit, and returns reconstructible numerator and denominator counts for each proportion:

- Retained RTA: count each newly retained case ID once.
- Selected RTA: count each `client_id × round_index × case_id` selection event once. The same case on another client or in another round is a new event; the two local epochs for the same event do not add duplicate counts.
- Qualification: qualified candidates / assessed candidates. The assessed population includes suppressed and correction-eligible candidates; candidates suppressed because of invalid cues remain in the assessed denominator.
- Retention: retained variants / variants that actually entered verification. Proposals before rendering-budget allocation, valid descriptor counts and qualification rates cannot replace this denominator.

An RTA hit requires the model-predicted class to equal the renderer focal class and box IoU≥.50. The tool recomputes IoU from the actual boxes rather than accepting an existing `agreement` value; conflicting values raise an error. A missing model prediction is a miss. RTA is undefined without the renderer reference, which must be reported as missing rather than replaced with the source pseudo-class. A denominator of 0 yields None, not 0% or 100%. If qualification records are entirely absent, both their numerator and denominator are None rather than fabricated 0s.

The retained provider uses the frozen verification bank and the fused prediction boxes and posteriors from available experts on the source route. The selected provider uses predictions from the current round's bank along the recorded route, without rerouting. Scope and reference are saved with each observation. These proportions measure agreement with renderer-generated annotations, not accuracy against independent human annotations. Identical duplicate rows for the same ID/event are counted once; conflicting duplicate observations raise an error rather than allowing the more favorable observation to be chosen.

The process CLI reads each round's `verification_round_*.json`, `selected_events.jsonl` and candidate audit, and checks that selection events cover the actual selection ledger. The manuscript does not fully specify the instance population and denominator for boundary error, so the tool requires a separately defined boundary subset and denominator. The corresponding boundary metric can be calculated only after an explicit manual boundary subset, independent ground truth and definitions of matches, missed detections and class errors are supplied separately.

## 8. Executed verification

`tests/test_analysis.py` covers ten-seed sample SD; matching global means with mismatched client-round records; known factorial effects; five-device trapezoidal integration and boundary interpolation; missing-device/sample and time-order rejection; worst-node P95; telemetry units and scope; unavailable GPU data; actual run files; and RTA deduplication, misses and empty denominators.

Seed summaries also require one record per method/seed and a common protocol signature. The signature includes configuration, dataset/calibration/initial-checkpoint digests and package versions. Seed and output-directory differences are allowed; changed protocol settings or missing required digests reject the summary. A calibration snapshot is hashed when present. File paths are ignored only when the corresponding content digest binds them.

These tests validate the analysis definitions and their boundary conditions. They do not provide new benchmark-performance results or hardware-resource measurements.
