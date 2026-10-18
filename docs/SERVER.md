# Server construction, verification, and router updates

`src/tefed/server.py` constructs correction cases, verifies frozen source evidence and updates the router. Rendering uses `BlenderRenderer` and the configured local `.blend` scene.

## Entry points

```python
created = construct_cases(
    descriptors, model, router, source_states, participating_clients,
    calibration, config, round_index, run_seed, output_dir,
)
# created = {"cases": [...], "audit": [...], "scored_descriptors": [...]}

report = update_router(
    model, router, manifest, pool, source_states, client_states,
    calibration, config, round_index,
    descriptors=created["scored_descriptors"],
)
```

The driver calls construction before expert aggregation and router updates after aggregation. Construction deep-copies model/router, switches the copies to evaluation mode and disables parameter gradients. This protects the round-start bank even if the surrounding driver later mutates its bank. New retained records carry `round_index`, and core reuse excludes records created in the current round.

## Construction sequence

1. Validate source class, source-state resolution, route, attributes, mask and duplicate descriptor ids.
2. Compute the paper's priority, assignment probabilities and deterministic global budget. `scored_descriptors` remain available when the budget is zero, so uploaded descriptor probes need not depend on successful rendering.
3. Render assigned tuples with the deterministic variant seed and a fresh output folder. Rendering failures raise an actionable error; stale assets are not accepted as a successful render.
4. Use actual frozen expert outputs and target geometry associations. Coverage is checked against original route mass before posterior renormalization. Verification never uses a source-label one-hot as a replacement prediction.
5. Measure class ambiguity from the actual target-associated verification posterior. Use the renderer annotation box as the geometric reference when measuring candidate attributes in the rendered frame. This interface uses renderer targets, which are available on the server; client unlabeled target annotations are never accessed.
6. Recompute classification, illumination, viewpoint, geometry, context and sensing from `extract_cues`. Real renderer visible-mask size and visibility measurements are retained for size/occlusion, with their proxy status stored in renderer metadata. Requested descriptor severities are never copied into missing measurements.
7. Apply coverage, source-class margin, original-route pairwise agreement, geometric validity and fixed-mask fidelity. All five gates must pass. An active attribute that remains unmeasurable causes rejection; the mask is not changed to make it pass.

The output `audit` preserves all gate values, route-control reference, original renderer measurements, final fidelity measurements, cue missingness and depth-scale status. There is no keep-all option.

## Depth units and synthetic sensing

Blender depth is expressed in scene units. Physical dataset calibration cannot be transferred to it by assuming those units are metres. `renderer.depth_unit_conversion` (legacy alias `renderer.depth_scale`), if supplied, explicitly multiplies rendered depth into the calibration units **once** before measurement and must also be applied when retained cases are loaded for training. A per-sample applied-factor marker prevents a loaded sample being converted twice.

This conversion is distinct from `model.depth_scale`, the fixed divisor used by the network on both real dataset inputs and rendered inputs. Server calibration and retained samples always receive that same network normalizer. Case records contain `depth_unit_conversion` and its legacy `depth_scale` alias as scene-unit conversion metadata, plus `depth_network_normalizer`; the legacy case field must never overwrite the prepared sample's network divisor.

Without a known conversion, the raw world-depth array is omitted (`sample.depth=None`), the depth-size calibration is disabled for rendered frames, and geometry uses the stated area/aspect fallback. Enabled-but-unavailable depth retains the explicit dropout behavior. A source-active geometry fidelity coordinate that cannot be measured remains missing and fails; retained counts are allowed to be zero. Rendered NIR is an image proxy, not a spectral simulation, and mask-based occlusion is a visibility proxy. These limitations remain in `render_metadata`.

`renderer.scene_class_map`, when configured, is forwarded to the renderer request, and `renderer.script_path` is forwarded to the Blender adapter. Supply the scene assets and class mapping for the target task.

## Route controls and Generic HC

`decision.verification_route` supports:

- `stored` (default): source identities and weights.
- `current`: the current route of an actual general-anchor-associated rendered detection. Missing anchors are logged as exclusions.
- `cyclic`: sort all round candidates by manifest image-ID and candidate-ID strings in ascending order, then transfer the complete normalized route and weight tuple from 11 positions later, with circular wrap-around. The audit binds recipient, donor, ordering and population size; source descriptors retain their original route.

For these controls, `case.d` continues to record the source descriptor; `verification_reference` records the control copy. `original_source_descriptor` and `original_source_route` retain the original metadata, including in Generic HC.

`experiment.method="generic_hc"` uses `calibration.generic_priors[family]` with `attrs` and optional `availability`. The dominant active source family selects the coarse prior; only that family flag remains in the coarse descriptor. Its pattern and fixed fidelity mask are recomputed. Verification then uses the actual current route. Missing generic prior artifacts cause a logged exclusion, rather than replacement with arbitrary values.

## Router probe bank and gradients

- Calibration identities must belong to validation. Probes use three deterministic photometric factors `.95, 1, 1.05`. A frame's first renderer/validation target defines its observation probe; retained cases use their focal target.
- Calibration probe states prefer `calibration.probe_states[identity_or_client]`. If absent, the explicit `client_states` argument is used and the report marks its origin. Retained probes require the immutable source-state lookup.
- Prior retained observations are ranked by fidelity and case id, default cap32. Raising `router.retained_probe_cap` can exceed the manuscript's maximum of 101 combined probes.
- Positive-priority uploaded descriptors are ranked by priority/id, default cap24. Pass the current round's `scored_descriptors` explicitly. Raw unscored descriptors can be scored using the frozen validation lookup even if rendering is disabled by budget.
- Observation probe targets use actual class-compatible IoU≥.50 expert predictions and the lowest native detection loss, breaking ties by expert id. Full-frame native losses use the complete available target set.
- Descriptor-only representations are the fixed adapter outputs, and their targets are the largest stored specialist weights. Descriptor probabilities are detached constants. Their values enter both complete-bank averages, while their target term has no encoder gradient.
- `router_objective` includes target CE and complete-bank load balance. Observation CE is divided by the total bank size, preserving the paper's observation/complete-bank scaling. A bank without observation probes cannot update the encoder and is explicitly skipped.
- The router uses 20 Adam steps at learning rate 0.001 and load-balance coefficient 0.02. Each run records its resolved configuration.

The report logs observation targets, their state origins, probe counts, optimizer steps and losses. Detector parameters remain frozen during probe construction and are excluded from the router optimizer.

## Verification performed

Server and route-control tests check frozen snapshots, source routes, gate rejection, complete tuple permutation, missing anchors, calibrated priors, router gradients and depth conversion. [TEST_REPORT.md](TEST_REPORT.md) records the executed suite. Injected render/inference services verify protocol mechanics.

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -p test_server.py -v
```
