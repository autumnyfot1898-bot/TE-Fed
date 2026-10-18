# Audit and analysis tools

Use the existing project Python environment. Tools read completed run records or explicit review inputs. Each command requires a fresh output path.

## Independent-review reliability

Review JSONL records use a unique `id`, recorded reliability and an independent review reference:

```json
{"id":"audit-a","reliability":0.8,"reviewed_class_match":true,"reviewed_iou":0.7,"review_reference":{"kind":"human_review","id":"review-set-v1"}}
```

The binary outcome is class match AND reviewed IoU >= 0.50. `reviewed_correct` can supply that binary outcome directly under the same rule. Model agreement and renderer-target agreement are separate observations and do not provide the independent review label.

```bash
python scripts/analyze_diagnostics.py reliability --records private/review.jsonl --out reports/reliability.json
```

The default six bins are `[.4,.5),...,[.9,1]`, following S11.3. Values below `.4` require explicit `--bins full` (ten bins over `[0,1]`) or `--bin-edges`. Every event must fall inside the declared bins. The output includes bin counts, mean reliability, accuracy, ECE, Brier and Bernoulli NLL; NLL clips probabilities at `1e-6`. Empty inputs do not produce zero-valued performance metrics.

## Routing against explicit condition labels

Supply independently recorded condition metadata in JSONL. The default join key is `(client_id, round_index, local_identity, id)`; `id` identifies the candidate within its observation:

```json
{"client_id":"client0","round_index":0,"local_identity":"source-key","id":"0","conditions":["illumination","occlusion"],"condition_source":"manifest","reference_id":"condition-manifest-v1"}
```

```bash
python scripts/analyze_diagnostics.py routing --candidates runs/seed0/client_<client-hash>_candidate_audit.jsonl --conditions private/conditions.jsonl --out reports/routing.json
```

Choose the actual `client_*_candidate_audit.jsonl` file in the run directory; the filename uses a client-ID hash. `--join-on identity`, `client_round` or `event_id` selects a different explicit key. The first condition controls correspondence. Top-1 is selected among specialists 1--5, with lower-ID tie breaking. Nominal records are excluded from correspondence and retained in route distributions. Missing condition labels are reported separately. Cue rule flags are not used as reference labels. Outputs contain condition groups, client route distributions, mean weights and correspondence counts.

## Paired method comparison

```bash
python scripts/analyze_diagnostics.py paired --left runs/fedavg/seed0 runs/fedavg/seed1 --right runs/tefed/seed0 runs/tefed/seed1 --seeds 0 1 --metrics mAP50 AP75 mAP50_95 --out reports/paired.json
```

The tool validates completed artifacts, declared seeds and shared data, initialization, calibration, evaluation and training settings. Method-specific differences in decision, loss, router, renderer, baseline, factorial and replay settings are retained and disclosed. Each method group must use one fixed protocol across seeds. Each output view/metric includes the two scores and right-minus-left difference for each seed, followed by mean and sample SD. Scores retain their input scale (normally 0--1); multiply differences and SD by 100 for percentage points. Seed IDs provide arithmetic matching. No CI or p-value is inferred.

## Case exports, plots and qualitative panels

```bash
python scripts/report_runs.py audit --run runs/seed0 --out reports/case-audit
python scripts/report_runs.py plots --runs runs/seed0 runs/seed1 --out reports/run-plots
python scripts/report_runs.py qualitative --manifest private/manifest.json --view hard --ids heldout-a heldout-b --prediction FedAvg=runs/fedavg/seed0/predictions/hard.jsonl --prediction TE-Fed=runs/tefed/seed0/predictions/hard.jsonl --image-size 640 --threshold 0.25 --out reports/qualitative
```

`audit` writes `audit.json`, `cases.csv`, `selections.csv`, `fidelity.csv` and a source manifest. `plots` writes training curves and metric comparisons as SVG, their values as CSV and source manifests. Supply `--factorial-records` and `--seeds` to add factorial-effect plots. `qualitative` writes a PNG, the explicit selection manifest and source manifests; `--view-spec` supplies Unseen perturbations when needed.

Case auditing links original observation keys, descriptors, rendered variants, retained cases and selection events. It checks next-round eligibility, route weights, fixed-mask fidelity and actual per-case presentations/optimizer steps. Earlier runs with absent audit fields keep those fields marked missing.

Result plots bind their input files and calculations in a source manifest. Qualitative panels use the same explicitly supplied IDs, view, display threshold and normalized coordinates for all methods; GT is shown separately. Available RGB, depth and NIR panels retain their source identity. Missing inputs are not generated.

Qualitative inputs normally bind the neighboring `evaluation_manifest.json`, including prediction-file bytes and data-manifest identity. A legacy prediction file needs explicit per-row `image_size`, `transform_hash_kind = "sample-metadata-sha256-v1"` and the actual sample metadata SHA-256. A view name alone does not establish geometry. Factorial plots require the executor-produced `factorial_<view>_<metric>.jsonl` beside its plan, execution and completed-job index; each plotted score is checked against its job metrics.

## Verification references on fixed rendered variants

```bash
python scripts/analyze_route_controls.py --run runs/seed0 --out reports/verification-routes.json
```

New runs save frozen verification inputs. This analysis reuses the same rendered population, targets, descriptor attributes, nonclass measurements, geometry and expert predictions for all three references. Class severity is recalculated as `1-(top1-top2)` from each covered route posterior, matching the online cue definition. Stored uses source tuples, current uses the frozen rendered-variant route, and cyclic uses the complete normalized candidate tuple 11 positions later in manifest-image-ID/candidate-ID order, with circular wrap-around.

The output reports variant-level gates, retention fractions and retained renderer-target micro agreement. Missing measurements needed by a control prevent a complete comparison. `--allow-incomplete` exports those events with undefined summary fractions. No historical retained-count targets or percentages are inserted. The online `mechanisms` plan can evolve different pools; this command supplies the fixed-variant comparison.

## Evaluation protocol records

Every new evaluation writes `predictions/evaluation_manifest.json` alongside prediction JSONL. It records exact view membership, client and pairing metadata, explicit condition tags/dictionary versions, letterbox geometry, Unseen parameters, normalization, seed, frozen state identities, model/router identities and configuration/calibration/manifest digests.

Model identities are hashes of state dictionaries, identified as such; they are not checkpoint-file hashes. Local and FedBN views identify the selected client model or BN state. Worker evaluation returns these metadata records without returning private images.

[Manuscript contracts](MANUSCRIPT_ALIGNMENT.md) · [Executed checks](TEST_REPORT.md)
