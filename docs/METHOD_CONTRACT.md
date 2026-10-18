# TE-Fed method and implementation

## 1. Execution scope

TE-Fed trains a shared YOLO26 detector, a general head and five specialist heads. The runner combines candidate routing, source-evidence verification, state-matched correction reuse and reliability-weighted aggregation. It schedules client tasks sequentially, locally or through authenticated HTTP workers.

The comparison registry includes `local`, `fedavg`, `fedprox`, `fedbn`, `confidence`, `consistent_teacher_fedawa`, `fedsto`, `fsslod`, `server_moe`, `generic_hc` and `tefed`. [BASELINES.md](BASELINES.md) defines their detector adaptations and controls.

Core invariants:

- `bar_R` averages original-stream candidate reliability for each routed head. An empty original-stream population gives `bar_R=0`, so correction-only updates cannot acquire aggregation mass through head activity.
- Qualified targets use their recorded candidate routes. Correction cases train the current student heads named by the stored source route. Route weights distribute supervision; there is no implicit equal three-head loss.
- Fidelity and compatibility control retention and selection. They do not multiply an individual correction case's loss.
- Construction and verification use the round-start expert bank. Aggregation precedes router updates. A case created in round `t` first becomes eligible in round `t+1`.
- Run metrics come from the declared evaluation identities and checkpoints. Generated inputs record `data_origin = "generated"`.

## 2. Modules and record contracts

### 2.1 Executable modules

| Module | Main interfaces |
| --- | --- |
| `data.py` | `load_manifest`, `validate_manifest`, `load_sample`, `map_boxes`, `apply_unseen`; typed `Annotation`, `SampleRecord`, `Manifest` and `LetterboxTransform`. |
| `state.py` | `initialize_state`, `update_state`, `state_demand`, `state_specialist_probabilities`, `state_relevance`. |
| `cues.py` | `extract_cues`, `quality_stats`, `routing_vector`; unavailable measurements retain explicit availability flags. |
| `core.py` | Routing/fusion, qualification, descriptor validation/scoring, allocation, five-gate verification, reuse and `aggregate_updates`. |
| `model.py` | `InputFusion`, `ExpertHead`, `RoutedYOLO`, `Router`, `sample_tensor`; detector outputs and region projections. |
| `losses.py` | `WeightedNativeLoss`, `contrastive_alignment`, `soft_consistency`. |
| `calibration.py` | `condition_probe`, `fit_temperature`, `build_calibration`. |
| `rendering.py`, `server.py` | `BlenderRenderer.render`, `construct_cases`, `router_objective`, `update_router`. |
| `engine.py` | `apply_input_policy`, `prepared_sample`, `assess_frame`, `train_local`, `run_experiment`, `pretrain`, checkpoint I/O and evaluation. |
| `client.py`, `transport.py`, `worker.py` | Shared client initialization/training/evaluation and authenticated worker execution. |
| `resume.py`, `replay.py`, `experiments.py` | Atomic round recovery, immutable-pool exposure control and explicit factorial operations. |
| `baselines.py` | `policy_dict`, aggregation, teachers and local penalties. |
| `metrics.py`, `analysis.py`, `diagnostics.py` | Detection AP, completed-run summaries, exposure/resource analysis and renderer-target observations. |
| `evaluation_manifest.py` | Evaluation identities, transforms, explicit metadata and model/state/prediction content bindings. |
| `offline.py`, `reporting.py`, `verification_controls.py` | Independent-review diagnostics, paired differences, linked audit exports, actual-result figures and frozen-variant route controls. |
| `cli.py` | The `python -m tefed` command interface. |

The paths above are under `src/tefed/`. [IMPLEMENTATION_CHOICES.md](IMPLEMENTATION_CHOICES.md) records model interfaces and initialization; [TRAINING.md](TRAINING.md) specifies optimization.

### 2.2 Records used by the runner

The core APIs use validated dictionaries; the record names below describe their contents, not additional Python classes.

| Record | Fields and invariants |
| --- | --- |
| Observation | Manifest identity/client/split, RGB and configured depth/NIR or registered IR, synchronized geometry and normalized targets. Unlabeled loading returns no targets. |
| Detection | Normalized `xyxy` box, finite calibrated posterior, class ID, native detector score and stable index. |
| Candidate | General-anchor identity, fixed `route`/`weights`, fused prediction, cues, `R`, `D_E`, threshold and disposition. |
| Client state | `q`, `p`, `h`; seven values in `[0,1]`, with the five `h` entries summing to one. The round-start copy remains fixed during the round. |
| Descriptor | `id`, `client_id`, `round`, eight `attrs`/`availability` entries, five `rule_flags`, fixed `mask`, `source_class`, three `route`/`weights` entries, `pattern`, HMAC `source`/`source_state_key` and `priority`. |
| Retained case | `id`, creation `round`, descriptor `d`, stored `fidelity`, image paths, full renderer targets, focal target, seed, measurements, gate audit and renderer metadata. |
| Client upload | `client_id`, shared/head `delta`, six `bar_R` and `gain` values. Next states, descriptors and training counts are retained by the driver in their respective records. |
| Router probe | Calibration/retained/descriptor type, representation or fixed probabilities, target expert and state origin. Observation vectors have gradients; descriptor probabilities remain fixed. |

Validation rejects duplicate IDs, invalid classes/routes, unresolved source-state keys and nonfinite values. Missing measurements follow the cue and fidelity rules. JSON output rejects NaN/Infinity. Source HMACs use a private runtime key; the key is not saved in logs or checkpoints. Runtime epsilon is `1e-8`, exact bin edges enter the upper interval, and Top2 ties favor the lower expert ID.

## 3. Configuration constants supplied in the manuscript

| Configuration | Manuscript value |
|---|---|
| detector/input | Ultralytics 8.4.0 YOLO26s, 640×640 |
| rounds/local epochs/batch | 30 / 2 / 4 |
| SGD | Momentum .937, weight decay .0005, LR .010→.001; cosine schedule after a 3-round linear warm-up |
| precision | FP16 AMP, FP32 loss/metric accumulation, global grad-norm clip 10 |
| experts | General + 5 specialist heads; active general + Top2; inactive local heads frozen |
| R coefficients | C .22, AE .24, AM .16, G .21, S .24, D penalty .08 |
| correction eligibility | D≥.35, C≥.50, G≥.50; checked only when R<eta |
| temperatures | Router .50, allocation .50, aggregation 1.00, state .50, contrastive .10 |
| verification | Coverage .80, source margin .05, AE .65, G .60, fidelity .65 |
| loss coefficients | alpha 1.00, beta .50, rho .10, gamma .50 |
| priority | N .15, U .15, L .25, Rel .30, risk penalty .25; chi .20 |
| aggregation | lambda_s .20; preserve the bar_R multiplier; epsilon>0, but its numerical value is not specified consistently |
| threshold | max(.70,.72+dq+dp+dh); lookup details below |
| rendering/reuse | B_round 1152; request `clip[3,8](3+round(5W+))`, which reaches at most 7 with these coefficients; reuse gate .55, cap 6, at most 2 per pattern and 1 per source |
| router probes | Calibration≤45, prior retained≤32, descriptor≤24, total≤101; load-balance coefficient .02 |

Lookup boundaries for `threshold(state)`: q `[0,.548,.697,.819,.917,1.01]` with offsets `[.007,.003,0,-.005,-.009]`; p `[0,.502,.619,.741,.847,1.01]` with offsets `[.006,.003,0,-.004,-.007]`; max(h) `[0,.079,.157,.242,.358,1.01]` with offsets `[-.004,-.001,.002,.005,.009]`. Since max(h)≥.2, the first two bins are not used by a valid state. The lowest valid eta is .706, so the floor is normally inactive. Exact bin edges belong to the upper interval.

The eight-attribute order is permanently fixed as `[size,cls,occ,illum,view,geo,ctx,sens]`. Their internal bin edges are `[.20,.45,.70]`, `[.25,.45,.65]`, `[.15,.35,.60]`, `[.18,.38,.62]`, `[.20,.42,.68]`, `[.18,.36,.58]`, `[.22,.45,.70]`, and `[.20,.40,.65]`. Fidelity tolerances are `[.18,.20,.18,.22,.20,.18,.22,.20]`; weights are `[.12,.15,.14,.12,.10,.16,.10,.11]`.

## 4. Function contracts for states, candidates, and routing

### 4.1 `state.update_state(previous, completed_round_stats)`

- q=clip(.30(1-qRGB)+.30 validD+.20(1-qNIR)+.20 mean AMpre).
- p_raw=.25 acceptance +.35 mean R+.20 mean S+.20 mean AE; when assessed>0, p=.6old+.4raw; when assessed=0, p is unchanged.
- The assessment denominator includes all 3 dispositions and candidates with invalid cues. Count each candidate once per client-round; repeated training presentations do not increase it.
- h_c=(n_c+1)/(sum n+5); each correction-eligible candidate contributes to at most one active condition family, selected by maximum severity. Tie order is illumination, occlusion, context, maturity, sensing. Candidates with no active family do not count; zero total count gives .2 for each family.
- State remains fixed throughout the round. Statistics from a completed round affect only the next round.
- If there are no assessed candidates or no valid candidate cues, p stays unchanged. Initialization uses validation p_raw, or `empty_p=.5` when unavailable. q renormalizes its defined sensing terms; it rejects a state with no defined terms. Enabled failed sensors keep their dropout/degradation fallback, rather than becoming disabled modalities.

### 4.2 `cues.extract_cues`, `cues.routing_vector` and `core.select_route`

- c0=max calibrated anchor posterior; calibration uses p=softmax(logits/T_C), with T_C frozen after validation fitting.
- AMpre=available-modality weighted class-consistent mapped IoU; a class mismatch contributes 0, but its weight remains in the denominator; all missing gives 0.
- Nominal modality weights are RGB .40, D .35, NIR .25; renormalize over available branches after missing branches are removed.
- sigma_box=clip(sum mapped-coordinate L1diff/(4×n×640)); references use same-class boxes only; no reference gives 0.
- Routing input `[c0,sigma_b,AMpre,Gpre,bview,z_h(5),state(7)]` has dimension 17. Fail immediately if the condition flags do not have dimension 5.
- qphi=Pphi(r)/(norm+epsilon), with dimension 5; score=qphi·e_m; softmax(score/.5) over all 6 experts.
- The general expert is always active; Top2 is selected only from specialists 1..5. The 3 active weights are strictly positive and normalized. Top2 ties favor the lower expert ID.

### 4.3 `core.fuse_candidate`

- Specialist matching requires the same class as the general anchor and IoU≥.5. Select the highest IoU, breaking ties by posterior score and then detection index.
- For fusion only, renormalize weights over matched experts and compute weighted-average boxes/posteriors; the candidate id remains the general-anchor id.
- Reliability C uses the original active-route weights, with unmatched outputs contributing 0; **do not replace C with confidence renormalized over matched outputs**.
- The AE numerator sums w_i w_j mu_i mu_j sameClass IoU over all active pairs; its denominator retains w_i w_j for all active pairs. With fewer than 2 available outputs, AE=0. Do not discard unmatched route mass.
- D=1-AE+entropy(activeweights)/log3, with range [0,2]; it is not simply expert disagreement.
- After fusion, recompute AMfused and G without rerouting. The candidate chain is AMpre→route→fusion→AMfused→R, avoiding circular dependencies.

### 4.4 `cues.extract_cues` and `cues.quality_stats`

Cue measurements follow these formulas:

- G=.30 area+.25(1-aspectResidual)+.45(1-depthResidual); unreliable depth switches to .55 area+.45(1-aspect). Area is clipped intersection/original box area; a degenerate box is invalid.
- For S, prefer the previous synchronized frame; otherwise use weak/strong mapped views. Require same-class IoU≥.5, with at most one match per view; an empty set gives 0.
- size=clip(1-sqrt(boxarea/640²)/.35), cls=1-(p1-p2), illum=clip(abs(log(mu/mu_cal))/log2.5), mu=0→1; invalid illumination gives availability 0.
- occlusion=1-clip(clippedarea/boxarea×(1-min(max_foreground_overlap,.85))); comparators must have confidence≥the candidate's and exclude the candidate itself.
- Measured calibration priors take precedence. Direct cue calls without a `class_priors` field use tomato depth/aspect fallbacks: ripe (-1.64,-.92,1.03) and nonripe (-1.71,-.89,.97). When depth valid fraction<.35, depthResidual=1 and G uses the area/aspect fallback.
- Plane depth is trimmed to the 10–90 percentile range, with normalized xy in [-1,1]; perspective=clip(sqrt(a²+b²)/(.35abs(c)+1e-6)); if valid pixels<64 or valid fraction<.35, perspective=.5; view=.6perspective+.4aspect.
- geo=.5depthResidual+.5(1-tempIoU); RGB fallback gives geo=1-tempIoU and view=aspect.
- context=.55 proposal_density+.45 texture_entropy. Proposal score floor .05, before classwise NMS, excluding the target; density=clip((N-40)/220). If ring 1.5 has fewer than 256 pixels, retry 1.25 (although smaller, preserve the manuscript's rule). If still insufficient, entropy .5 is an audit placeholder only, with availability 0 and flag 0.
- Grayscale intensities for the 3×3 Laplacian must be 0–255, with variance 120; a crop smaller than 16×16 gives blur=1; RGB=.7blur+.3clipping, with clipping severity clipping_fraction/.20.
- NIR residual is the difference between the [0,1] crop and its 3×3 median-filtered crop, with MAD×1.4826/.08; missing NIR gives severity 1; sensor=max(depthdropout,NIRnoise,RGBdegradation) under the availability rules.
- Distinguish a fully disabled modality from a temporarily unavailable one. DroneVehicle/RGB-only masking follows the separate task mapping.

### 4.5 `core.qualify_candidate(candidate, state, config=None)`

First verify that C/AE/AM/G/S are all finite and in [0,1]. Discard invalid candidates without reducing the assessed count. Then compute D/R: R=clip(.22C+.24AE+.16AM+.21G+.24S-.08D). The decision order is: R≥eta gives qualified; otherwise, D≥.35 and C≥.5 and G≥.5 gives correction-eligible; otherwise, suppressed. The three sets are disjoint and cover assessed candidates.

For RGB-only and No-alignment, do not simply set AM=0 in the original formula. As specified in the SI, divide by the active positive-weight sum .91 and retain the D penalty. RGB-only sets rho 0 and makes the sensor attribute inactive while retaining RGB weak/strong consistency.

## 5. Descriptors, priorities, and rendering schedules

### 5.1 `core.make_descriptor` / `core.validate_descriptor`

source_class=fused class; retain the original route ids/weights. The 8 availability values are independent of the 8 fidelity-mask entries. pattern=sum(4**q ×bin_q), with zero-based q. The first 32 hexadecimal SHA-256 HMAC characters encode 128-bit source/state keys. HMAC inputs are canonical JSON lists `[client_id, local_id]` and `[client_id, round_id, state]`, using sorted keys, compact separators and UTF-8. The descriptor ID is the first 32 SHA-256 hexadecimal characters of `[source, round_id, candidate_id]`. The server must resolve statekey to the immutable round-start state. An invalid source class, unresolvable statekey, or duplicate descriptor id must fail before rendering.

### 5.2 `core.score_descriptors`

- For recurrence, count each pattern at most once per participating client, then divide by the number of participating clients. Do not use the number of descriptors.
- U is entropy/lognonzero after renormalizing the stored specialist weights only; it is 0 with 1 specialist (the current route normally has 2).
- Use fixed Wd×(availability⊙attrs), normalize, then compute cosine; a zero projection gives cosine 0→relevance .5. Copy Wd exactly from SI Table S7.3 and test its numerical entries and shape.
- risk=mean of 8 attrs; retain the manuscript's denominator regardless of availability.
- L comes from a frozen validation lookup, not an online estimate. Select the maximum active family severity; 5 families×4 severity×4 geometry×4 cue-count cells=320; geo offsets [0,.0271,.0594,.1087] and cue offsets [0,.0136,.0324,.0511]; no active family gives L=0.
- `calibration.build_calibration` measures the 20 base cells from frozen validation identities and records perturbation parameters, seeds and losses. [CALIBRATION.md](CALIBRATION.md) defines the complete protocol. Training requires the measured artifact bound to the shared initial checkpoint.
- W+=max(0,.15N+.15U+.25L+.30Rel-.25risk). Only W>0 enters scheduling; this is not the candidate-qualification threshold.
- Specialist allocation uses softmax((Rel+.2storedspecweight)/.5), excluding the general expert.

### 5.3 `core.allocate_rendering(descriptors, budget=1152)`

Sort by descending priority, with lexicographic descriptor ties within 1e-6; request=clip3..8(3+round(5W)). If the remaining budget is insufficient, the final descriptor receives the remainder. Use Hamilton largest remainders, breaking ties by the lowest specialist id; the highest-weight expert receives at least 1 when realized>0. Descriptors do not carry over and are not cached for reevaluation next round after the budget is exhausted. Total allocated≤budget, and each descriptor's allocations sum to realized. An assertion checks that Hamilton allocation gives the highest-weight specialist at least one variant; no separate post-allocation transfer changes the counts.

### 5.4 `core.variant_seed` / `BlenderRenderer.render`

Join UTF8 fields `runseed,round,descriptorid,specialistid,variantindex` with byte 0x1f. The first 4 bytes of SHA256, interpreted as unsigned big-endian, give the 32-bit seed for NumPy PCG64. The Blender adapter uses the seed for focal-object selection, bounded scene perturbations and sensor corruption. The request preserves descriptor values and availability; [RENDERING.md](RENDERING.md) defines the implemented controls and measured outputs. The specialist changes allocation and seed only, not descriptor intervals.

The renderer returns the complete image/modalities, complete annotations, an associated source-target id, and realized attribute measurements. Source-class-side ambiguity through appearance interpolation and scale/depth-consistent placement must be traceable. **Requested controls must not be used directly as realized measurements.**

Local `.blend` assets and explicit class mappings supply scene geometry and labels. Depth remains in scene units until a recorded scene-to-dataset conversion is applied. NIR intensity and mask-based occlusion retain their proxy metadata; NIR and thermal materials are not physically simulated. Missing active fidelity measurements reject a case.

## 6. Five-gate verification

`server.construct_cases` freezes the bank; `core.verify_case` returns the gate audit and retention decision.

1. Rerun inference with the stored 3 experts. Target association **checks geometry IoU≥.5 only**, without prefiltering by source class or general class. Choose the highest IoU, breaking ties by score/index. Posteriors must be finite, share the same class dimension, lie in [0,1], and sum to 1.
2. coverage=sum of original stored weights over available outputs. Empty/≤0/<.8 fails immediately; the posterior is undefined.
3. Only after coverage passes, compute posterior=sum weight×p/coverage; source margin=p[source]-maxother≥.05. Class count must be≥2.
4. AE uses target-associated hypotheses, the original weights, and the original pair denominator. A pair contributes only when its classes agree; available<2→0. Gate .65.
5. G≥.60 and fidelity≥.65. fidelitymask=availability×(bin≥1 OR mappedconditionflag), with flag mapping `[0,z_mat,z_occ,z_illum,0,0,z_ctx,z_sens]`.
6. Fidelity requires active-weight mass Z>0; score=max(0,1-distance(measured,requestedinterval)/tolerance), followed by the weighted mean. The mask is fixed. A missing active measurement makes the whole quantity undefined and causes rejection; do not remove missing dimensions to obtain a pass. With a single active check, fidelity equals that check rather than being normalized across 8 attributes.

The five gates are coverage, margin, agreement, geometry, and fidelity; the interface and measurement rules are expanded into 6 steps above. The expert assigned for verification does not replace the stored expert mixture, and rerouting is not allowed. After retention, stored fidelity must not be recomputed from the client state.

## 7. State matching and reuse timing

`core.select_reuse(pool, receiving_state_t, round_t, config=None)` returns `selected` and item-level `audit`.

- Exclude created_round≥t first; the pool is cumulative, without age expiry.
- State source demand=clip(b_s+W_s×[1-q,1-p,h_illum,h_occ,h_ctx,h_mat,h_sens]); exact Ws and bs are in the SI table and are not trained.
- State specialist scores: illumination `.15(1-q)+.10(1-p)+.75hillum`; occlusion `.10,.10,.80`; context `.08,.12,.80`; maturity `.05,.15,.80`; sensing `.25,.05,.70`. Apply softmax(score/.5) over 5 specialists. This is **not the candidate-router posterior**.
- Amatch=exp(-sum(mask×abs(attrs-demand))/(.25summask+epsilon)).
- overlap=sum of current-state probabilities for stored specialists, excluding the general expert, clipped to [0,1].
- comp=.65match+.25overlap+.10storedfidelity; gate .55.
- Sort by (-comp,-fidelity,-W+,caseid), then greedily select at most 6, with 2 per pattern and 1 per source. Continue scanning after a skipped case; do not truncate to the top 6 before enforcing diversity.
- The same case may be selected in a later round or by another client. Each selection is an independent reuse event; repeated training presentations are counted separately.
- Audit records must include the available pool version, state key/snapshot, complete ranking, gate failures, reasons for source/pattern-diversity or cap exclusions, and selected ids.

## 8. Local training

`engine.train_local` assembles labeled observations, qualified target sets and previously selected correction cases. [TRAINING.md](TRAINING.md) defines active-head and target-weight policies.

### 8.1 Detection objective

J=Lsup + alpha/(|Q|+eps) Σcandidate R_i ell_det_i + beta/(|Hselected|+eps) Σcase ell_det(rendered,completeRendererTargets) + rhoLctr + gammaLcon.

- Qualified boxes from an image are assigned together by `WeightedNativeLoss`. Assigned positives receive target reliability times route weight; background receives the targets' mean weight. The native assignment normalizer stays unweighted, preserving the effect of mean reliability.
- Pseudo supervision is normalized by candidate count, not sum R. Correction supervision is normalized by case count, not box count, and uses complete renderer targets. Fidelity and compatibility add no loss factor.
- Correction cases activate stored-route student heads. Labeled supervision uses the union of original-stream routed heads, or a state-based Top2 fallback when that union is empty. Inactive heads retain no gradient.
- The assembled items are shuffled uniformly each local epoch. Count factors preserve labeled-item, pseudo-candidate and correction-case means. SGD and GradScaler are created per client-round call; optimizer state is not carried across rounds.
- CUDA AMP uses FP32 reductions. Gradients are unscaled before global-norm clipping at 10. Nonfinite FP32 gradients abort; AMP overflows skip the optimizer update and reduce the scale. Attempted, skipped and applied updates are logged separately.

### 8.2 Multimodal alignment

`RoutedYOLO.contrastive_regions` extracts matched candidate regions using actual `torchvision.ops.roi_align`, then a shared trainable linear projection. `contrastive_alignment` normalizes each projection by `norm+1e-8`.

| Model field | Default |
| --- | --- |
| `contrastive_dim` | 128 |
| `contrastive_feature_level` | -1, the final shared feature level |
| `contrastive_roi_size` | 1, a 1×1 pooled region |
| `contrastive_sampling_ratio` | 2; RoIAlign uses `aligned=True` |

All ordered pairs of distinct available modalities for the same candidate are positives. Other candidates in the destination modality from the current optimizer minibatch are negatives. The denominator includes the positive; temperature is .10 by default. Regions are the assembled labeled, qualified or renderer targets; unlabeled ground truth is never read. Fewer than two usable candidates or no available cross-modal pairs gives zero alignment loss. Modality-specific forwards preserve shared-trunk BN running statistics while retaining gradients; the projection belongs to the aggregated shared state.

### 8.3 Weak/strong consistency

`RoutedYOLO.decode_head` decodes differentiable one-to-many boxes for `soft_consistency`. Strong photometric views preserve geometry. Matching is greedy one-to-one, same class and IoU≥.5. Each matched pair contributes:

`0.5 KL(stopgrad weak posterior || strong posterior) + 0.5 (1−GIoU(stopgrad weak box, strong box))`.

Weak boxes and posteriors are detached; strong-view gradients reach the trunk. The driver normalizes over all matched pairs in the optimizer minibatch. Empty matches return a differentiable zero. Confidence/max-detection settings come from evaluation configuration, and `loss.consistency_temperature` defaults to 1. Consistency applies to pseudo training items; spatial transforms would require an explicit coordinate mapping.

## 9. Original aggregation and the shared backbone

`core.aggregate_updates(base_state, uploads, states, prototypes, previous_state=None, config=None)`:

- Per-head barR is the arithmetic mean of all defined candidate R values whose routes contain that head, counting each candidate once, including qualified/correction/suppressed candidates whenever R is defined; empty=0. Do not count training presentations.
- gain=clip((validationloss_before-validationloss_afterProjectedUpdate)/(before+eps)); the projected model applies only the shared backbone + head m, not the full multi-head update.
- h_s=normalize(state scores 5-vector), Rel=(1+cos(h_s,e_m))/2. The general prototype must also be supplied; adopt a consistent zero-norm cosine=0 policy.
- conflict=.5(1-cos(projectedDelta, previousCompletedServerProjectedDelta)); t0/either norm 0→0.
- xi=Rel×gain×barR×(1-conflict), v=expm1(xi/tau_s); expm1 improves precision for small quantities while remaining mathematically equivalent.
- headomega=v/(sumclientsv+eps); headnew=headold+.2Σomega×headDelta. All masses 0 means no update; weights sum to≤1 rather than being forced to sum to 1.
- Backbone activity is defined only by whether the corresponding headDelta has L2 norm>0. Client backbone mass=Σactivity×headmass/(Σactivity+eps), then normalize across clients and apply each client backboneDelta once. **Do not apply a separate backbone update for each head weight.**
- If the local backbone changes but every headDelta is exactly 0, its backbone mass is 0, as specified in the manuscript.
- bank_t is immutable and the new bank is constructed independently. Do not update bank_t in place and contaminate this round's rendering/verification.

## 10. Router-probe updates

After aggregation: calibration 15×3 weak augmentations, at most 45 (omit if no compatible expert); retained cases ranked by fidelity, top 32; descriptors ranked by W+, top 24. The retained probe bank includes only cases with `case.round < round_index`, excluding current-round creations. Retained observations use immutable source states. Calibration probes use their recorded validation state or the explicit frozen client-state argument; the audit records its origin. An unresolved state invalidates the probe.

- Calibration and retained targets require same-class IoU≥.5 for compatibility. Choose the compatible expert with the lowest detection loss, breaking ties by the lower expert id; omit when none is compatible.
- A descriptor probe uses the fixed Wd projection and targets the largest stored specialist weight, breaking ties by the lower expert id.
- qphi output/descriptor projection and prototypes jointly define a 6-expert softmax with temperature .5.
- Route loss=-sum(logtargetprob)/completebankN +.02sum(meanprob-overcompletebank-1/6)^2.
- Fixed descriptor probabilities have no direct encoder gradient, but they enter the complete-bank mean and affect load balancing and the scale of observation gradients. Do not artificially train descriptor probabilities through a straight-through estimator or replace complete-bank normalization with observation-only normalization.
- With Nobs=0 and only fixed descriptor probes, the router encoder has no gradient: explicitly skip the optimizer step. Likewise, skip an empty bank. A missing probe gradient must not be misreported as a successful update with 0 loss.
- `Router` uses a linear 17→5 projection, the normalized all-ones general prototype and five coordinate specialist prototypes. Calibration factors are `.95,1,1.05`; retained ties use case ID. `update_router` defaults to 20 Adam steps at learning rate .001, with the configured load-balance coefficient .02.

## 11. Round execution and checkpoints

`engine.run_experiment` executes the following order:

1. Copy round-start states and export the shared/head bank. Select eligible old cases for each receiving state.
2. Assess original-stream unlabeled observations with the round-start bank, build qualified target sets and descriptors, and register immutable source-state keys.
3. Train client copies using labeled, qualified and selected old correction data. Record deltas, original-stream `bar_R`, projected validation gains, next states and actual training counts.
4. Construct and verify new cases against frozen copies of the round-start bank. Append retained cases to the cumulative pool; they remain ineligible for local training until the next round.
5. Aggregate expert heads and apply each client backbone update once. Ordinary baselines use their registered aggregation policy.
6. Build probes with the updated expert bank and prior retained cases, then update the router. Activate the completed next states.
7. Write round ledgers and optional read-only monitoring. Final evaluation uses the declared final round.

`save_checkpoint` writes model/router state dictionaries, resolved configuration, client states, zero-based `round_index`, `completed_rounds`, `model_schema_version` and implementation status. Model schema 2 includes the shared contrastive projection. Earlier checkpoint schemas are rejected: regenerate pretraining and its bound calibration with the current code.

`final.pool.json` stores the retained pool, and `source_states.json` stores the immutable source-state lookup. Final Local-only checkpoints additionally contain every client model and the calibration-selected new-client model identity. FedBN checkpoints contain the client BN states used for evaluation.

`resume.pt` separately stores atomic round snapshots: persistent teacher/client/BN state, pool/source records, aggregation direction, RNG, protocol/input bindings and committed file hashes. Optimizers and GradScaler are rebuilt each round by the protocol. `--resume` continues from the last completed round and rolls back unfinished outputs. [WORKFLOWS.md](WORKFLOWS.md) defines the recovery boundary; `final.pt` remains the initialization/evaluation checkpoint.

## 12. Commands and run artifacts

Prepare the private runtime key and local paths as described in [GETTING_STARTED.md](GETTING_STARTED.md). With `configs/local.toml` pointing to your manifest, scene and output paths:

```bash
python -m tefed methods
python -m tefed validate data/tomato/manifest.json --check-files
python -m tefed pretrain --config configs/local.toml --rounds 10 --seed 0 --output runs/initial_detector
python -m tefed calibrate --config configs/local.toml --checkpoint runs/initial_detector/final.pt --output data/tomato/calibration.json
```

Set `model.initial_checkpoint` and `experiment.calibration` to those artifacts before training:

```bash
python -m tefed train --config configs/local.toml --method tefed --seed 0 --output runs/tefed_seed0
python -m tefed evaluate --config configs/local.toml --checkpoint runs/tefed_seed0/final.pt --output runs/tefed_seed0_recheck
python scripts/run_suite.py --config configs/local.toml --plan comparisons --output runs/comparisons
python scripts/run_suite.py --config configs/local.toml --plan mechanisms --output runs/mechanisms --execute
```

The suite supports `comparisons`, `mechanisms`, `budgets` and `components`; it writes a plan unless `--execute` is present. `python -m tefed smoke --output runs/software_smoke` runs generated test inputs, saves a checkpoint and checks four evaluation views.

Completed training writes resolved configuration/calibration, environment and artifact hashes, capacity, candidate audits, verification records, selected events, round ledgers, checkpoint, pool/source states, predictions and metrics. Ledgers include applied/attempted/skipped steps, labeled/pseudo/correction presentations and auxiliary-loss counts. Round monitoring writes separate metrics and duration; it does not update the model or choose an early stopping round.

Case ledgers additionally record detection-supervised presentations and local zero-based step IDs. Verification records retain complete fidelity inputs and frozen target-associated expert predictions. Evaluation sidecars bind prediction bytes and the manifest, transform, state and model identities. [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md) defines the audit and figure workflows.

Evaluation uses frozen states for Participating/Unseen/Hard and a fixed validation prior for New-client. Manifests define the exact held-out identities; Unseen transformations are explicit configuration. DroneVehicle uses RGB detector inputs and registered IR reliability, with explicit OBB-to-horizontal conversion and sensor masking. AP uses native detector confidence, not the qualification threshold.

[EXPERIMENTS.md](EXPERIMENTS.md) defines declared-seed summaries, per-client-round exposure checks, process proportions, latency and measured energy. Resource integration uses device telemetry with recorded node identities and measurement scope.

## 13. Implemented test coverage

| Test modules | Coverage |
| --- | --- |
| `test_core.py`, `test_cues.py` | Numerical routing/fusion/gates, fixed route mass, state/reuse timing, allocation ties and missing sensor measurements. |
| `test_data_metrics.py`, `test_calibration.py` | Identity leakage, synchronized transforms, AP, declared seeds, measured lookup cells and missing/nonfinite calibration failures. |
| `test_input_protocol.py` | Shared sensor policy, calibration/checkpoint binding, early renderer checks and disabled correction paths. |
| `test_model_loss.py`, `test_native_training.py`, `test_training_contract.py`, `test_amp.py` | Native target weighting, active heads, RoIAlign/projection, minibatch modality pairs, matched half-KL/half-GIoU gradients, checkpoint schema and AMP accounting. |
| `test_server.py` | Frozen verification bank, immutable source routes, gate rejection, source/current/cyclic controls, router gradients and scene-depth conversion. |
| `test_baselines.py`, `test_analysis.py` | Aggregation/teacher policies, exposure/seed summaries, factorial analysis, telemetry integration, latency and process denominators. |
| `test_resume.py`, `test_transport.py`, `test_experiment_protocols.py` | Round recovery, worker-local execution, immutable replay and all eight factorial cells. |
| `test_cli_config.py`, `test_main.py`, `test_engine_integration.py`, `test_diagnostics.py` | CLI/configuration, native round execution, checkpoint/evaluation consistency, read-only monitoring and renderer-reference observations. |

Run from the installed checkout:

```bash
python -m unittest discover -s tests -t . -v
```

[TEST_REPORT.md](TEST_REPORT.md) records executed checks and environments. Dataset performance, seed stability and resource measurements require their corresponding completed runs.

## 14. Required local inputs

- Fixed labeled/unlabeled/validation/test/hard observation identities, synchronized sensor units and explicit evaluation views. [DATA.md](DATA.md) defines the manifest contract.
- A schema-compatible shared initial checkpoint and measured validation calibration bound to its SHA-256. Generated calibration records `data_origin = "generated"` and requires explicit `test_mode`.
- A private `TEFED_SOURCE_SECRET` of at least 32 UTF-8 bytes for data training. Keep it privately for exact source-identifier replay.
- Blender executable, `.blend` scene, task class mapping and any known scene-to-dataset depth conversion when correction rendering is active. Runtime preflight checks the executable, scene and entry scripts before creating a run output directory.
- Device telemetry and UTC round boundaries for resource analysis. [HARDWARE.md](HARDWARE.md) provides setup and device-check commands.

## 15. Execution requirements

- Local and HTTP tasks use the same client implementation.
- Recovery starts at a committed round boundary and requires unchanged code, protocol, inputs and backend. An unfinished round is recomputed.
- Dynamic matched/independent runs can develop different pools. Use the separate frozen-pool replay plan for strict recorded-exposure comparisons; insufficient eligibility or update-count differences reject the arm.
- The factorial plan executes the declared `tefed-factorial-v1` operations with `communication.mode = "local"`.
- Supply local data, trained weights and scene assets separately. Intensity/visibility proxies remain marked in renderer metadata.
