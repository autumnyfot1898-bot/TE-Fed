# Baseline strategies

TE-Fed runs the following detector adaptations and internal controls under a
shared training and evaluation protocol. The operator definitions below specify
the adaptations used for each comparison.

The common protocol uses zero-based `round_index=0..29` and `epoch_index=0,1`.
The corresponding displayed round is `round_index+1`. Local, FedAvg, FedProx and
FedBN use **labeled observations only**. Confidence self-training adds unlabeled
observations with threshold 0.75, and the teacher methods add their respective
teacher/student rules. Expert controls use their documented routed qualification.
The shared runner applies the corresponding supervised or pseudo-label policy.

## Registry and policy

`policy_dict(method, round_index, epoch_index)` returns strategy fields. It does
not train a model by itself. The runner must apply each requested operator.

| Method key | Strategy |
| --- | --- |
| `local` | Labeled-only client training; no aggregation |
| `fedavg` | Labeled-only training and sample-count-weighted model averaging |
| `fedprox` | Labeled-only FedAvg plus proximal coefficient 0.01 |
| `fedbn` | Labeled-only FedAvg with client-local BN affine parameters and buffers |
| `confidence` | Confidence self-training with threshold 0.75 |
| `consistent_teacher_fedawa` | EMA 0.999, consistency weight 0.50, threshold 0.75, adaptive aggregation described below |
| `fedsto` | EMA 0.999, reset teacher from broadcast at every round start; epoch 0 heads only, epoch 1 shared layers and active heads; orthogonality coefficient 0.05 |
| `fsslod` | EMA `0.99+0.009*(2*round_index+epoch_index)/59`; consistency 0.50; elastic anchoring coefficient 0.10 |
| `server_moe` | Routed expert model without correction rendering/reuse |
| `generic_hc` | Generic hard-case correction with the same rendering budget and state-matched reuse schedule as TE-Fed |
| `tefed` | Source-evidence verification and state-matched reuse |

The supplementary table gives FSSLOD's EMA endpoints and per-epoch constancy.
`update_ema_teacher` is called **after every optimizer step**, using the decay
fixed for that epoch. The final displayed round 30, epoch 2 has EMA 0.999.
`reset_teacher_each_round` is a **round-start event**, not a batch-level reset.

The manuscript's Generic HC comparator shares TE-Fed's rendering budget and
state-matched reuse schedule. Its verification/qualification differs; disabling
state matching is a separate ablation. The policy therefore sets
`use_state_matched_reuse=True` for both `generic_hc` and `tefed`.

## FedAvg and FedBN

`sample_weighted_aggregate(states, sample_counts)` accepts the model export tree:

```python
{"backbone": {"trunk.0.conv.weight": numpy_array, ...},
 "heads": {"0": {"detect...weight": numpy_array, ...}, ...}}
```

Each floating tensor is averaged using normalized nonnegative client sample
counts, with float64 accumulation and the original output dtype. Trees, shapes,
dtypes and finite values are checked. No input arrays are modified. Integer and
Boolean state leaves, if supplied, are copied from the largest-weight client;
ties use input order. The detector export currently includes floating states
only, so integer counters do not enter its FedAvg tree.

`batchnorm_state_keys(model)` identifies BN modules by their PyTorch class and
returns exact affine `weight/bias`, running mean/variance and counter keys.
`is_batchnorm_key(name, known_keys=...)` supplies a preservation predicate. The
string-only fallback recognizes conventional `bn`, `bn1`, `batchnorm` segments;
use the exact key set for numerically named Sequential modules.

FedBN requires a **separate cached local BN state per client**. At broadcast,
merge global non-BN states while preserving that client's BN parameters and
buffers. At aggregation, either preserve the server BN state or exclude BN
leaves, then restore each client's own state at the next broadcast. Passing
`keep_predicate` and `previous` to the aggregator supports preservation; one
shared `previous` object cannot replace the per-client cache.

## Teachers and local penalties

- `make_ema_teacher(student)` deep-copies the model, disables gradients and sets
  evaluation mode. No parameters or storage are shared with the student.
- `update_ema_teacher(teacher, student, decay)` updates all floating parameters
  and buffers using EMA; integer buffers are copied. Float16/bfloat16 arithmetic
  accumulates in float32; float64 remains float64. Teacher state stays frozen.
- `proximal_penalty(model, broadcast_reference, mu=.01)` returns
  `mu/2 * sum ||parameter-reference||²` over trainable parameters, not BN buffers.
- `fedsto_set_trainable(model, epoch_index, active_heads)` freezes both trunk and
  input-fusion parameters in epoch 0, and sets these shared modules to evaluation
  mode so that their BN statistics do not drift. Epoch 1 unfreezes them. Call it
  **after** the model's ordinary `.train()` call at each epoch.
- `orthogonality_penalty(..., coefficient=.05, active_heads=...)` uses this
  FedSTO detector adaptation: flatten each active head's matrix/convolution weights, use
  the smaller row/column Gram matrix after unit-norm normalization, and average
  `mean((Gram-I)²)` across matrices.
- `elastic_penalty(model, broadcast_reference, coefficient=.10)` uses this
  FSSLOD detector adaptation: average each trainable parameter tensor's mean squared
  distance to the frozen round-start broadcast reference and multiply by
  `coefficient/2`. Comparisons use this mean-per-block anchoring definition.

The runner should use the ordinary native detection and consistency losses
defined for the common detector. None of these helpers silently rescales or
replaces those losses.

## Explicit adaptive aggregation adaptation

`adaptive_aggregation_weights(update_vectors, validation_gains=None, ...)`
implements the following detector adaptation for the combined
Consistent-Teacher + FedAWA comparator.

1. Flatten shared-backbone parameter **deltas** in lexicographic parameter-key
   order with `flatten_update_vector`. Do not pass model weights, AP scores, or
   different specialist-head vector spaces as update vectors.
2. Normalize every nonzero client delta direction. A zero delta has direction
   zero. Form the sample-count-weighted mean direction (uniform by default).
3. Score each direction by its cosine similarity to that mean direction. An
   optional client calibration-set AP improvement is added as a fractional
   value in `[-1,1]`, e.g. `0.02` means two percentage points. Evaluation/test
   gains must never be used here.
4. Apply softmax with temperature 1.0 and proportional water filling onto a
   unit simplex capped at 0.50 per client.

The cap is feasible only if `number_of_clients * cap >= 1`. A single client is
rejected under cap 0.50. Exactly two clients necessarily receive weights 0.50,
0.50; there is no adaptive freedom in that case. Four all-zero update vectors
with no gains receive uniform weights. Stable norm/softmax calculations handle
large finite vectors without treating overflow as a valid direction.

## Verification

```text
python -m unittest tests.test_baselines -v
```

NumPy tests check weighted averaging, copy semantics, client-local BN
preservation, invalid states, cap feasibility, client permutation and EMA policy
endpoints. Tensor tests additionally check exact BN keys, EMA integer-buffer
handling, proximal gradients, orthogonality and stage-wise BN freezing. Tensor
tests explicitly skip if the training dependency group is absent.
