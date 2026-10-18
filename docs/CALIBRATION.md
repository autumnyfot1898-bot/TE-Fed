# Build a measured frozen validation artifact

TE-Fed freezes a calibration artifact measured from validation identities. It contains temperature, class priors and a loss lookup built from the image-space probes below. Priors and losses must be measured with the selected checkpoint.

## Caller contract

`build_calibration(manifest, loss_evaluator, posterior_evaluator, image_size=640, seed=0, task="tomato")` reads **only** the explicitly named `manifest.calibration_ids`. They must be nonempty, unique and refer to validation records. By default exactly 15 identities are required. `strict_count=None` permits an explicitly documented small smoke study and is stored in the artifact; it does not silently claim compliance with the 15-frame protocol.

The caller first selects a common trained checkpoint for all methods, freezes its weights and batch-normalization statistics, disables gradients, and supplies two callbacks:

- `loss_evaluator(sample) -> float`: one finite, nonnegative detection loss using `sample["targets"]`. Keep the model, criterion, preprocessing and loss reduction identical for nominal and perturbed images. A return value that is NaN, infinite, negative or absent aborts construction.
- `posterior_evaluator(sample) -> list[dict]`: each detection contains normalized letterbox-coordinate `box` and a normalized finite `posterior` list. Optional `native_logits` must have the same dimension. Optional `score` controls deterministic matching order. The prediction's class label is not used for association.

Callbacks receive `data.load_sample()` dictionaries with RGB/depth/NIR and normalized targets. They must not train or select a checkpoint during calibration. A newly initialized model with no geometrically matching boxes is unsuitable: the routine fails instead of inserting `T_C=1`. Record pretraining and checkpoint selection separately from the calibration artifact.

For long runs, `progress(event)` receives nominal/condition-loss completion counts. This does not change the evaluated cells.

## Temperature calibration

Nominal detections are greedily associated with validation GT at IoU >=0.50, without filtering by predicted class. A prediction and GT are each matched at most once. The objective is mean categorical NLL across matched boxes. If native logits are absent, the routine uses `log(max(p,1e-12))`; for a softmax posterior this differs from its original logits only by a class-independent constant before temperature scaling.

One temperature is fitted over `[0.05,10]`: a 257-point log-temperature grid brackets the minimum, followed by golden-section refinement. The artifact records NLL before/after, matched-box count, source kind, bounds and boundary flags. No geometric matches or a flat/unidentifiable objective raise an error. Unmatched boxes are not included in this conditional matched-box calibration, and this temperature fit alone does not establish end-to-end detector calibration.

## Measured image and class statistics

- `luminance_median`: median across pooled content pixels of `(.299R+.587G+.114B)/255`; letterbox padding is excluded.
- `depth_validation_median`: median of pooled positive finite content depth values. If no valid depth is available, the field is null/None, not a guessed distance.
- `class_priors[class_id].r_c_med`: median pixel width/height of the class's GT boxes.
- `a_c,b_c`: ordinary least-squares coefficients in `log(h/H)=a_c+b_c*log(max(z/z_val,1e-3))`. The height is normalized in the actual letterboxed detector-input coordinates, and `z` is median valid depth in the GT box. Boxes with less than 35% valid depth do not enter the depth fit. At least two observations with a rank-two design are required; insufficient/constant depth returns null coefficients with an explicit status.
- `area_depth`: separately requested log-normalized-area regression on the same log-depth predictor. It is additional reporting; it does not replace the SI's height-depth equation.

Unobserved classes retain null priors and zero counts. These fields must remain missing in downstream cues rather than silently falling back to published coefficients. Depth units remain the dataset's sensor units; renderer scene units must not be labeled calibrated metres.

## Complete five-family, four-level loss lookup

Default evaluation consists of 15 nominal frames plus `15 x 5 x 4 x 3 = 900` perturbed loss calls, using three explicit seeds `(seed,seed+1,seed+2)` modulo 2^32. Four levels represent severity-bin midpoints `.125,.375,.625,.875`. All probes preserve image geometry and the original GT boxes/classes. Their image-space definitions are:

| Family | Four explicit controls | Scope |
|---|---|---|
| illum | RGB brightness factors `.90,.75,.60,.45` | Content pixels only; NIR unchanged |
| occ | Occluding rectangle requested areas `.08,.16,.24,.32` of each GT box | Same rectangle masks RGB/depth/NIR; complete-object targets remain fixed |
| ctx | Background color mixture `.10,.20,.30,.40` | Pixels outside the union of GT boxes; object boxes preserved |
| mat | Object desaturation `.10,.20,.30,.40` | Maturity-appearance proxy for tomato, class-boundary-appearance proxy for DroneVehicle; no physical class transition |
| sens | RGB/NIR noise SD fractions `.02,.04,.06,.08`, matching depth SD fraction and dropout `.02,.04,.06,.08` | Depth SD uses frame median positive depth; no missing modality is fabricated |

Retaining labels under photometric degradation and synthetic occlusion is an augmentation assumption, not an independent annotation-quality measurement. Context cells can produce zero increases if a frame has no background. Maturity appearance probes do not change biological maturity or vehicle category. Illumination and desaturation factors are fixed at each level, so their three seed calls may contain identical transformed pixels. These are probe measurements, not independent training repetitions or a source of cross-seed uncertainty.

For every identity, family, level and seed, the measured relative increase is `max(0,(loss_pert-loss_nom)/(loss_nom+1e-6))`. Zero nominal loss is allowed only through this explicit denominator and its identity is recorded in `zero_nominal_loss_ids`; no denominator is silently replaced. The routine:

1. Requires all identity/seed measurements in each of the 20 cells.
2. Winsorizes each cell at its observed 5th and 95th percentiles.
3. Takes the median in that cell.
4. Divides by the observed P95 across the 20 non-nominal cell scores and clips to `[0,1]`.

If every cell score is zero, the artifact explicitly records `all_cells_zero=true` and a `1e-6` normalization denominator, returning zeros. This is a measured degenerate outcome, not missing-cell imputation. A missing/nonfinite cell value aborts construction.

The output `loss_lookup={"base":{"illum":[4 values],"occ":[4],"ctx":[4],"mat":[4],"sens":[4]}}` plugs into `core.validation_loss_score()`. The SI's fixed geometry offsets `0,.0271,.0594,.1087` and cue-count offsets `0,.0136,.0324,.0511` expand these measured 20 base cells to the 320 family/severity/geometry/cue combinations. Their manuscript origin is recorded; the routine does not claim to have newly fitted these offsets.

## Audit and use

The artifact contains calibration IDs, nominal losses, all observed condition losses and parameters, cell counts/quantiles/medians, P95 normalization, frozen `T_C`, measured priors and complete protocol metadata. It is validated as strict JSON without NaN/Infinity. Store it together with the common checkpoint hash, manifest hash and configuration used by the CLI. Do not recompute it from unlabeled client streams or evaluation images during training.

Actual-data training requires `initial_checkpoint_sha256` in the artifact to match the bytes of `model.initial_checkpoint`. A missing or mismatched binding aborts before training output is created. Version 0.2 requires a model-schema-2 checkpoint; regenerate pretraining and calibration together when upgrading an earlier checkpoint.

Tests in `tests/test_calibration.py` verify all 20 measured cells with hand-computable callbacks, exact callback counts, no reads of nonexistent test/unlabeled image files, geometry/target preservation, deterministic probes, null unsupported priors, zero-loss handling, missing/nonfinite-cell rejection and a temperature optimum with a closed-form answer.
