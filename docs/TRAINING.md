# Training contract

TE-Fed trains active routed heads with native detection losses, multimodal alignment and correction supervision. The policies below define head activation, weighting and aggregation.

## Active heads

- Ordinary baselines instantiate the general head only. Expert methods instantiate one general and five specialist heads with one shared trunk/neck.
- Qualified targets retain their round-start route. The union of positive-weight target routes activates heads for a training frame.
- A correction case activates the current student heads identified in its recorded source route. Those identities remain fixed during local presentations.
- Labeled supervision trains the union of the client's original-stream routed heads. If that union is empty, a state-based Top-2 route supplies the supervised entry. This supervised routing choice is explicit.
- Only activated heads are forwarded. Zero-coefficient penalties are not constructed, so unused heads cannot acquire zero gradients that would trigger weight decay. FedSTO additionally freezes the trunk and its BN statistics in its first local epoch.

## Detection loss

Each active head uses the pinned Ultralytics native target assignment, class criterion and localization criterion. Head widths are configurable; each run records measured parameter counts.

For qualified targets, every target has its own reliability multiplied by its head's recorded route weight. Assigned positive anchors inherit that target weight. Classification background is allocated uniformly across supplied targets, hence receives their mean weight. Native assignment normalizers remain unweighted: dividing by the sum of reliabilities would remove the intended effect of low mean reliability.

The entire target set is assigned once for an image/head. The code does not repeat the full-image background loss for each box. Non-routed targets carry zero foreground contribution rather than being relabeled as background. A genuinely empty labeled image retains native background supervision; an empty pseudo-label set does not create a training item.

Correction examples supply the complete renderer annotation set. Their route weights distribute supervision among recorded heads, with weights summing to one. Cases receive uniform outer weights. Fidelity, priority, and current compatibility are used in their stated decisions and do not multiply correction training loss.

The local assembled set is sampled uniformly. Item factors account for the labeled count, qualified **candidate** count, and selected **case** count separately. Qualified frames are multiplied by their number of qualified targets, preserving the candidate-indexed mean. Actual presentations and optimizer steps are recorded; equal epochs do not imply equal optimizer work.

## Additional losses and schedules

Contrastive learning applies RoIAlign to each candidate box and a shared trainable projection before L2 normalization. Candidates across the local minibatch provide negatives. InfoNCE averages over all ordered pairs of distinct available modalities; a missing sensor is excluded, while an available all-zero observation remains valid. Fewer than two candidates or no available modality pair gives zero loss. The model defaults are projection dimension 128, feature level -1, RoI output size 1 and sampling ratio 2, with aligned sampling.

Modality-specific contrastive forwards use frozen BN running statistics while retaining parameter gradients. This keeps masked sensor inputs from replacing the statistics learned from fused observations. The prior BN training flags are restored after each forward.

Consistency uses decoded one-to-many native predictions so the strong-view loss reaches the shared features. Detached, same-class weak predictions are greedily matched one-to-one with strong predictions at IoU >= 0.50, using the configured prediction-confidence floor and maximum detection count. Each match contributes `0.5 * KL(stopgrad(p_weak) || p_strong) + 0.5 * (1 - GIoU)`. The local minibatch loss divides the summed terms by the total matched-pair count; an empty match set gives zero. Strong views currently use photometric transforms with identity coordinate mapping. Spatial transforms require an explicit coordinate map.

SGD uses the declared momentum, weight decay, warm-up and cosine schedule. AMP is enabled only on CUDA; reductions accumulate in FP32. Global-norm clipping follows unscaling. Nonfinite losses and non-AMP gradients abort the step; GradScaler skips overflowed AMP updates and lowers its scale. Logs distinguish attempted, skipped and applied steps. Native one-to-many/one-to-one loss scheduling advances by the global local-epoch index instead of restarting its schedule every communication round.

## Aggregation and validation

The original `bar_R` factor averages each valid original-stream candidate once for each routed expert. A head trained only by correction cases can therefore have zero aggregation mass if it has no routed original-stream candidates. The code preserves that behavior.

Validation loss uses held-out observations with gradients and BN updates disabled. Per-head projected updates include the shared trunk and that head only. The most recent completed server direction supplies the conflict reference. Client backbone masses average unnormalized masses over heads with nonzero head updates; one normalized weight then applies each client's backbone update once.

Floating parameters and floating BN buffers are aggregated. Integer tracking counters remain local/non-aggregated and are not treated as trainable parameters. FedBN retains its specified affine/buffer keys and reloads frozen client BN states at evaluation. The Local-only cold-start detector is selected once by minimum server calibration loss; its test predictions do not tune that selection.

## Checkpoint schema

Model schema 2 includes the shared contrastive projection in checkpointing, initialization and aggregation. Checkpoints without that schema or projection are rejected. Regenerate earlier pretraining and calibration together; schema-2 initialization checkpoints remain supported.

`final.pt` supports evaluation and initialization. V2.6 writes atomic `resume.pt` snapshots with persistent client/teacher/BN state, pool/source records, aggregation direction, RNG and committed output hashes. Resume continues from the last completed round. Optimizers and GradScaler retain their existing per-round lifecycle. See [WORKFLOWS.md](WORKFLOWS.md) for commands and supported recovery boundaries.
