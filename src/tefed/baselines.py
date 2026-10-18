"""Baseline detector adaptations and federated update policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from numbers import Real
import re
from typing import Any, Callable

import numpy as np


METHOD_REGISTRY = {
    "local": "Local-only detector; no server aggregation",
    "fedavg": "Sample-count-weighted aggregation",
    "fedprox": "FedAvg with local proximal coefficient 0.01",
    "fedbn": "FedAvg with client-local BatchNorm parameters and buffers",
    "confidence": "Fixed-confidence self-training at 0.75",
    "consistent_teacher_fedawa": "EMA consistency with an explicit adaptive aggregation adaptation",
    "fedsto": "Round-reset EMA teacher and two-stage local training",
    "fsslod": "Scheduled EMA teacher with an explicit elastic anchoring adaptation",
    "server_moe": "Routed expert model without rendered correction cases",
    "generic_hc": "Generic hard-case correction with the matched rendering/reuse schedule",
    "tefed": "Source-evidence verification and state-matched reuse",
}


def _nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _sample_weights(sample_counts: Sequence[Real], count: int) -> np.ndarray:
    values = np.asarray(sample_counts)
    if values.shape != (count,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("sample_counts must be a finite, nonnegative vector with one entry per client")
    values = values.astype(np.float64)
    # Scale before summing to avoid overflow for large finite counts.
    if count == 0 or values.max(initial=0) <= 0:
        raise ValueError("At least one client must have a positive sample count")
    values /= values.max()
    return values / values.sum()


def sample_weighted_aggregate(
    states: Sequence[Mapping[str, Any]],
    sample_counts: Sequence[Real],
    *,
    previous: Mapping[str, Any] | None = None,
    keep_predicate: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Aggregate nested backbone/head states by client weight.

    Float leaves accumulate in float64. Integer/bool leaves use the largest
    weight, with input order breaking ties. keep_predicate takes dotted names
    and copies matching leaves from previous. Inputs are not mutated or aliased."""
    if not isinstance(states, Sequence) or not states:
        raise ValueError("states must be a nonempty client sequence")
    weights = _sample_weights(sample_counts, len(states))
    largest = int(weights.argmax())
    if keep_predicate is not None and previous is None:
        raise ValueError("previous state is required when preserving local leaves")

    def model_path(parts):
        return ".".join(parts[1:] if parts and parts[0] == "backbone" else parts)

    def visit(nodes, prior, parts):
        first = nodes[0]
        location = ".".join(parts) or "<root>"
        if isinstance(first, Mapping):
            keys = set(first)
            if any(not isinstance(node, Mapping) or set(node) != keys for node in nodes):
                raise ValueError(f"Model state keys differ at {location}")
            if prior is not None and (not isinstance(prior, Mapping) or set(prior) != keys):
                raise ValueError(f"Previous state keys differ at {location}")
            return {key: visit([node[key] for node in nodes], None if prior is None else prior[key], parts + [str(key)]) for key in first}
        arrays = [np.asarray(node) for node in nodes]
        shape = arrays[0].shape
        dtype = arrays[0].dtype
        if dtype.kind not in "biuf" or any(a.shape != shape or a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Invalid numerical state leaf at {location}")
        if any(a.dtype != dtype for a in arrays):
            raise ValueError(f"State dtypes differ at {location}")
        if keep_predicate is not None and keep_predicate(model_path(parts)):
            reference = np.asarray(prior)
            if reference.shape != shape or reference.dtype != dtype or not np.isfinite(reference).all():
                raise ValueError(f"Invalid preserved state leaf at {location}")
            return reference.copy()
        if dtype.kind in "biu":
            return arrays[largest].copy()
        total = np.zeros(shape, dtype=np.float64)
        for weight, value in zip(weights, arrays):
            total += weight * value.astype(np.float64)
        result = total.astype(dtype)
        if not np.isfinite(result).all():
            raise ValueError(f"Aggregated state overflows at {location}")
        return result

    return visit(list(states), previous, [])


fedavg_aggregate = sample_weighted_aggregate


def batchnorm_state_keys(model) -> set[str]:
    """Return BN parameter and buffer keys from the model."""
    from torch.nn.modules.batchnorm import _BatchNorm
    names = set()
    for prefix, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            names.update(f"{prefix}.{key}" if prefix else key for key in module.state_dict())
    return names


def is_batchnorm_key(name: str, model=None, *, known_keys: set[str] | None = None) -> bool:
    """Match BN parameters and buffers using exact keys when available.

    Name-only matching requires a named BN segment; numeric module paths
    cannot identify BN affine parameters."""
    if not isinstance(name, str):
        raise TypeError("state key must be a string")
    if name.startswith("backbone."):
        name = name[len("backbone."):]
    keys = known_keys if known_keys is not None else batchnorm_state_keys(model) if model is not None else None
    if keys is not None:
        return name in keys
    suffix = name.rsplit(".", 1)[-1]
    if suffix in {"running_mean", "running_var", "num_batches_tracked"}:
        return True
    return suffix in {"weight", "bias"} and any(re.fullmatch(r"(?:bn|batchnorm|batch_norm)\d*", part.lower()) for part in name.split(".")[:-1])


def make_ema_teacher(student):
    teacher = deepcopy(student)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def update_ema_teacher(teacher, student, decay: float) -> None:
    """Apply EMA to floating states and copy integer buffers."""
    import torch
    decay = _nonnegative(decay, "decay")
    if decay > 1:
        raise ValueError("decay must lie in [0, 1]")
    target_state, source_state = teacher.state_dict(), student.state_dict()
    if set(target_state) != set(source_state):
        raise ValueError("Teacher and student state schemas differ")
    for name in target_state:
        if target_state[name].shape != source_state[name].shape:
            raise ValueError(f"Teacher and student shapes differ for {name}")
    with torch.no_grad():
        for name, target in target_state.items():
            source = source_state[name].detach().to(device=target.device)
            if target.is_floating_point() or target.is_complex():
                work_dtype = torch.float64 if target.dtype == torch.float64 else torch.complex128 if target.dtype == torch.complex128 else torch.complex64 if target.is_complex() else torch.float32
                updated = target.to(dtype=work_dtype).mul(decay).add(source.to(dtype=work_dtype), alpha=1 - decay)
                target.copy_(updated.to(dtype=target.dtype))
            else:
                target.copy_(source.to(dtype=target.dtype))
    teacher.eval()
    teacher.requires_grad_(False)


def _reference_parameters(model, reference):
    values = dict(reference.named_parameters()) if hasattr(reference, "named_parameters") else reference
    if not isinstance(values, Mapping):
        raise TypeError("reference must be a model or a named parameter mapping")
    parameters = dict(model.named_parameters())
    missing = set(parameters) - set(values)
    if missing:
        raise ValueError(f"Missing reference parameters: {sorted(missing)}")
    for name, value in parameters.items():
        if tuple(values[name].shape) != tuple(value.shape):
            raise ValueError(f"Reference shape differs for {name}")
    return parameters, values


def proximal_penalty(model, reference, mu: float = 0.01):
    """Return mu/2 times parameter squared distance, excluding buffers."""
    import torch
    mu = _nonnegative(mu, "mu")
    parameters, reference_values = _reference_parameters(model, reference)
    first = next(iter(parameters.values()), None)
    total = first.sum() * 0 if first is not None else torch.zeros(())
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            anchor = torch.as_tensor(reference_values[name], device=parameter.device, dtype=parameter.dtype).detach()
            total = total + (parameter - anchor).square().sum()
    return total * (mu / 2)


def elastic_penalty(model, reference, coefficient: float = 0.10):
    """FSSLOD detector adaptation: mean L2 distance to broadcast weights."""
    import torch
    coefficient = _nonnegative(coefficient, "coefficient")
    parameters, reference_values = _reference_parameters(model, reference)
    first = next(iter(parameters.values()), None)
    zero = first.sum() * 0 if first is not None else torch.zeros(())
    terms = [(parameter - torch.as_tensor(reference_values[name], device=parameter.device, dtype=parameter.dtype).detach()).square().mean()
             for name, parameter in parameters.items() if parameter.requires_grad and parameter.numel()]
    return (sum(terms, zero) / len(terms) if terms else zero) * (coefficient / 2)


def orthogonality_penalty(model, coefficient: float = 0.05, active_heads: Sequence[int] | None = None):
    """FedSTO adaptation: penalize normalized Gram matrices of active heads."""
    import torch
    import torch.nn.functional as functional
    coefficient = _nonnegative(coefficient, "coefficient")
    if hasattr(model, "heads"):
        indices = list(range(len(model.heads))) if active_heads is None else list(active_heads)
        if len(indices) != len(set(indices)) or any(not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(model.heads) for index in indices):
            raise ValueError("active_heads must contain distinct valid head indices")
        parameters = [p for index in indices for p in model.heads[index].parameters()]
    else:
        if active_heads is not None:
            raise ValueError("active_heads requires a model.heads collection")
        parameters = list(model.parameters())
    first = next(iter(model.parameters()), None)
    zero = first.sum() * 0 if first is not None else torch.zeros(())
    terms = []
    for parameter in parameters:
        if not parameter.requires_grad or parameter.ndim < 2:
            continue
        matrix = parameter.flatten(1).float()
        if matrix.shape[0] > matrix.shape[1]:
            matrix = matrix.T
        matrix = functional.normalize(matrix, dim=1, eps=1e-8)
        gram = matrix @ matrix.T
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        terms.append((gram - identity).square().mean())
    return (sum(terms, zero) / len(terms) if terms else zero) * coefficient


def fedsto_set_trainable(model, epoch_index: int, active_heads: Sequence[int] | None = None) -> dict[str, Any]:
    """Freeze shared modules and their BN at epoch 0; unfreeze at epoch 1."""
    if not isinstance(epoch_index, int) or isinstance(epoch_index, bool) or epoch_index not in (0, 1):
        raise ValueError("FedSTO uses zero-based epoch_index 0 or 1")
    if not hasattr(model, "trunk") or not hasattr(model, "heads"):
        raise ValueError("FedSTO stage control requires model.trunk and model.heads")
    chosen_list = list(range(len(model.heads))) if active_heads is None else list(active_heads)
    if len(chosen_list) != len(set(chosen_list)) or any(not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(model.heads) for index in chosen_list):
        raise ValueError("active_heads must contain distinct valid head indices")
    chosen = set(chosen_list)
    freeze = epoch_index == 0
    for name in ("trunk", "fusion", "contrastive_projection"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(not freeze)
            module.train(bool(model.training) and not freeze)
    for index, head in enumerate(model.heads):
        head.requires_grad_(index in chosen)
        head.train(bool(model.training) and index in chosen)
    return {"freeze_trunk": freeze, "active_heads": sorted(chosen)}


def fsslod_ema_decay(round_index: int, epoch_index: int) -> float:
    if not isinstance(round_index, int) or isinstance(round_index, bool) or not 0 <= round_index < 30:
        raise ValueError("round_index must be zero-based in [0, 29]")
    if not isinstance(epoch_index, int) or isinstance(epoch_index, bool) or epoch_index not in (0, 1):
        raise ValueError("epoch_index must be zero-based 0 or 1")
    return 0.99 + 0.009 * (2 * round_index + epoch_index) / 59


def flatten_update_vector(update: Mapping[str, Any], *, shared_only: bool = True) -> np.ndarray:
    """Concatenate floating parameter deltas in key order.

    Shared-backbone deltas give all clients the same comparison space."""
    if not isinstance(update, Mapping):
        raise TypeError("update must be a nested mapping")
    node = update.get("backbone", update) if shared_only else update
    values = []
    def visit(mapping):
        for key in sorted(mapping, key=str):
            value = mapping[key]
            if isinstance(value, Mapping):
                visit(value)
            else:
                array = np.asarray(value)
                if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                    raise ValueError(f"Invalid update array: {key}")
                if array.dtype.kind == "f":
                    values.append(array.astype(np.float64).ravel())
    visit(node)
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def adaptive_aggregation_weights(update_vectors, validation_gains=None, *, sample_counts=None, temperature: float = 1.0, cap: float = 0.50) -> np.ndarray:
    """CT+FedAWA adaptation using shared-parameter delta vectors.

    Combine directional cosine scores with clipped fractional validation gains,
    then project softmax weights onto a simplex bounded by cap. This is not the
    official FedAWA optimizer; two clients with cap=.5 must receive equal weight."""
    vectors = np.asarray(update_vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 2 or not np.isfinite(vectors).all():
        raise ValueError("update_vectors must be a finite n-by-d matrix with n >= 2")
    temperature = _nonnegative(temperature, "temperature")
    cap = _nonnegative(cap, "cap")
    if temperature <= 0 or not 0 < cap <= 1 or len(vectors) * cap < 1 - 1e-12:
        raise ValueError("Require positive temperature and feasible cap (n * cap >= 1)")
    count = len(vectors)
    base = _sample_weights([1] * count if sample_counts is None else sample_counts, count)
    scales = np.max(np.abs(vectors), axis=1, keepdims=True, initial=0)
    scaled = np.divide(vectors, scales, out=np.zeros_like(vectors), where=scales > 0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    directions = np.divide(scaled, norms, out=np.zeros_like(vectors), where=norms > 0)
    reference = (base[:, None] * directions).sum(axis=0)
    ref_norm = np.linalg.norm(reference)
    alignments = directions @ (reference / ref_norm) if ref_norm > 0 else np.zeros(count)
    gains = np.zeros(count) if validation_gains is None else np.asarray(validation_gains, dtype=np.float64)
    if gains.shape != (count,) or not np.isfinite(gains).all() or np.any(np.abs(gains) > 1):
        raise ValueError("validation_gains must be finite AP fractions in [-1, 1]")
    scores = alignments + gains
    with np.errstate(over="ignore", under="ignore"):
        raw = np.exp(np.clip((scores - scores.max()) / temperature, -700, 0))
    result = np.zeros(count)
    free = np.ones(count, dtype=bool)
    remaining = 1.0
    while free.any():
        proposal = remaining * raw[free] / raw[free].sum()
        exceeded = proposal > cap + 1e-14
        free_ids = np.flatnonzero(free)
        if not exceeded.any():
            result[free_ids] = proposal
            break
        fixed_ids = free_ids[exceeded]
        result[fixed_ids] = cap
        free[fixed_ids] = False
        remaining = 1 - result.sum()
    if not np.isclose(result.sum(), 1) or np.any(result > cap + 1e-12):
        raise RuntimeError("Adaptive weight projection failed")
    return result


def policy_dict(method: str, round_index: int, epoch_index: int) -> dict[str, Any]:
    """Return the zero-based 30-round, two-epoch policy.

    Read once per epoch; reset_teacher_each_round applies only at round start.
    Local/FedAvg/FedProx/FedBN use labels; teacher/self-training uses threshold
    .75; expert methods use routed reliability."""
    if not isinstance(method, str) or method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHOD_REGISTRY)}")
    # Constant EMA methods still require valid protocol settings.
    scheduled = fsslod_ema_decay(round_index, epoch_index)
    experts = method in {"server_moe", "generic_hc", "tefed"}
    teacher = method in {"consistent_teacher_fedawa", "fedsto", "fsslod"}
    return {
        "method": method, "round_index": round_index, "epoch_index": epoch_index,
        "pseudo_threshold": 0.75,
        "use_unlabeled": method not in {"local", "fedavg", "fedprox", "fedbn"},
        "use_experts": experts,
        "active_heads": "routed" if experts else [0],
        "use_teacher": teacher,
        "teacher_decay": scheduled if method == "fsslod" else 0.999 if teacher else None,
        "teacher_update_frequency": "after_optimizer_step" if teacher else None,
        "reset_teacher_each_round": method == "fedsto",
        "freeze_trunk": method == "fedsto" and epoch_index == 0,
        "proximal_mu": 0.01 if method == "fedprox" else 0.0,
        "orthogonal_coefficient": 0.05 if method == "fedsto" else 0.0,
        "elastic_coefficient": 0.10 if method == "fsslod" else 0.0,
        "consistency_weight": 0.50 if teacher else 0.0,
        "preserve_client_batchnorm": method == "fedbn",
        "aggregation": "none" if method == "local" else "adaptive_directional" if method == "consistent_teacher_fedawa" else "expert_state_quality" if experts else "sample_weighted",
        "aggregation_temperature": 1.0,
        "aggregation_cap": 0.50 if method == "consistent_teacher_fedawa" else None,
        "use_corrections": method in {"generic_hc", "tefed"},
        "use_source_verification": method == "tefed",
        "use_state_matched_reuse": method in {"generic_hc", "tefed"},
        "is_new_detector_adaptation": True,
    }
