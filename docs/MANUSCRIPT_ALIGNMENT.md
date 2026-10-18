# Manuscript alignment

V2.6 was checked against the revised main text and supplement dated 11 October 2026, together with the reuse, fidelity and evaluation audit workbooks. The source fingerprints identify the revision used:

| Source | SHA-256 |
| --- | --- |
| Revised `main.tex` | `73e5c565d02dc7ec96edc5da82e4a926b885e468431fc8587836febb1bdb046e` |
| Revised `supplementary.tex` | `9ef518c0bdd24dbd9faca05b714229522ca1173ea7ce6a9d703d938ca4de4e27` |

## Method and protocol

| Manuscript contract | Executable implementation |
| --- | --- |
| S1: fixed splits and evaluation identities | Manifests separate labeled, unlabeled, validation, standard test and Hard identities. Participating, Unseen and New-client use declared views; Hard remains disjoint. |
| S1: frozen final states | Participating, Unseen and Hard use the completed client states; New-client uses the validation prior. Evaluation does not adapt state or BN. |
| S2: detector and optimization | YOLO26s, a shared trunk, one general and five specialist heads; defaults are 30 rounds, two local epochs, batch size four and SGD. |
| S3: qualification | Original route mass, calibrated cues and state threshold `max(.70,.72+dq+dp+dh)` are preserved. |
| S4: descriptor priority and allocation | Coefficients `.15,.15,.25,.30,.25`, allocation chi `.20`, budget 1152, priority ties within `1e-6`, Hamilton integer allocation and deterministic variant seeds. |
| S5/S7.4: fidelity | Fixed eight-attribute order, availability/rule mask, bin intervals, manuscript weights and tolerances; missing active measurements reject the case. |
| S7: correction and aggregation | Round-start verification bank, stored source route and weights, next-round eligibility, state matching and original-stream mean reliability in aggregation. |
| S8: matched reuse | Frozen-pool replay checks the same pool, selection counts, actual correction presentations and attempted/applied updates per client and round. |
| S8.2: cyclic verification | Sort the round's candidates by manifest image ID and candidate ID, then transfer complete normalized route tuples from 11 positions later with wrap-around. Fixed-variant analysis preserves geometry and predictions; class severity follows each route posterior. |
| S9: paired differences | Join declared seed IDs, compute right-minus-left within each seed, then mean and sample SD. No CI or significance test is inferred. |
| S11.3: reliability | Independent review outcomes, six default bins from `.4` to `1`, ECE, Brier and Bernoulli NLL with clipping at `1e-6`. |
| S11.4: routing | First explicit condition label; nominal events excluded from correspondence and included in route distributions; Top-1 among specialists 1--5. |
| S11.5: qualitative comparison | Explicit held-out IDs, shared view, crop/coordinates and display threshold across methods; ground truth is a separate panel. |

The current parameter workbook supplies the priority coefficients above. An older checklist contains different coefficients and chi; those values are not mixed into the revised protocol.

## Audit identities

The export chain is `source_observation_key -> descriptor_id -> rendered_variant_id/case_id -> selection_event_id`. A case keeps one persistent ID across receiving clients and rounds. A selection event identifies one client-round-case decision. The full eight-dimensional `pattern_key` remains separate from both identities.

Per-case ledgers record actual presentations, detection-supervised presentations and local zero-based attempted/applied/AMP-skipped step IDs. A selected case can have zero detection supervision, including when beta is zero. Selection counts do not substitute for presentation counts.

Fidelity records include requested intervals, realized severities, the fixed mask, weights, normalizers, per-attribute fidelities, active weight mass and the first failed gate. An early verification failure leaves fidelity explicitly unevaluated. A measured severity of zero is a valid measurement.

Each evaluation output includes `evaluation_manifest.json`, binding identities, explicit condition metadata, transforms, frozen state, model/router state and protocol inputs. Missing optional condition or pairing metadata remains null; labels are not inferred from cue rules.

## Factorial record semantics

Supplement S8 identifies the table's Full series as `ABL-v1.3`, distinct from the primary TE-Fed series. It describes A/B/C as configuration flags rather than proof that a disabled path executed.

The executable `tefed-factorial-v1` profile defines its operations in [WORKFLOWS.md](WORKFLOWS.md): A controls qualification, B selects source evidence versus Generic HC, and C enables correction. When C is off, B has no training path. Its `111` cell is ordinary TE-Fed. The analyzer can also calculate contrasts from separately supplied eight-cell records; their provenance stays with those records.

Manuscript scores, fixed audit counts and source images are not embedded in training or regenerated by these tools. Analyses use the selected completed run artifacts and explicit review inputs. [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md) defines their schemas and commands.
