"""Maintain client state {q, p, h[5]} from TE-Fed Supplementary S7.

h orders illumination, occlusion, context, class boundary, and sensing.
State stays frozen within a round and updates at the next boundary."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

FAMILIES = ("illum", "occ", "ctx", "mat", "sens")
ATTRIBUTES = ("size", "cls", "occ", "illum", "view", "geo", "ctx", "sens")

STATE_BIAS = np.array([.0617, .0467, .0377, .0425, .0531, .0490, .0355, .0271])
STATE_MATRIX = np.array([
    [.1218, .0845, .0199, .2618, .0376, .0526, .0939],
    [.0476, .1919, .0177, .0283, .0610, .3468, .0285],
    [.0238, .0405, .0070, .5859, .0657, .0254, .0440],
    [.0539, .0509, .6261, .0198, .0180, .0167, .0498],
    [.0449, .0644, .0553, .1198, .0365, .0403, .0561],
    [.0903, .0786, .0308, .1684, .0636, .0493, .1533],
    [.0196, .0459, .0191, .0440, .5976, .0194, .0264],
    [.2676, .0476, .0317, .0315, .0286, .0205, .6033],
])


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    return result


def validate_state(state: Mapping[str, Any]) -> dict:
    """Validate and copy state; reject NaNs and non-normalized profiles."""
    q, p = _unit(state["q"], "state.q"), _unit(state["p"], "state.p")
    h = [_unit(x, f"state.h[{i}]") for i, x in enumerate(state["h"])]
    if len(h) != 5 or not np.isclose(sum(h), 1., atol=1e-7, rtol=0.):
        raise ValueError("state.h must contain five entries summing to one")
    return {"q": q, "p": p, "h": h}


def _get(stats: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in stats:
            return stats[name]
    raise ValueError(f"missing measured state statistic: {names[0]}")


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _state_measurements(stats: Mapping[str, Any]) -> tuple[float, list[float]]:
    # Renormalize defined terms; failed enabled sensors keep their SI fallback.
    q_rgb = _get(stats, "q_rgb", "q_RGB")
    valid_depth = _get(stats, "valid_depth_fraction", "v_depth", "v_D")
    q_nir = _get(stats, "q_nir", "q_NIR")
    mean_am = _get(stats, "mean_am_pre", "mean_AM_pre")
    terms=[]
    for name,value,weight,invert in (("q_rgb",q_rgb,.30,True),
            ("valid_depth_fraction",valid_depth,.30,False),("q_nir",q_nir,.20,True),
            ("mean_am_pre",mean_am,.20,False)):
        if value is not None:
            measured=_unit(value,name)
            terms.append((weight,1-measured if invert else measured))
    if not terms:
        raise ValueError("state sensing quality has no defined measurements")
    q = float(np.clip(sum(w*v for w,v in terms)/sum(w for w,_ in terms),0.,1.))
    raw_counts = stats.get("failure_counts", [0]*5)
    if isinstance(raw_counts, Mapping):
        raw_counts = [raw_counts.get(name, 0) for name in FAMILIES]
    if len(raw_counts) != 5:
        raise ValueError("failure_counts must have five condition-family counts")
    counts = [_count(x, "failure_counts") for x in raw_counts]
    h = [(n+1)/(sum(counts)+5) for n in counts]
    return q, h


def _pseudo_raw(stats: Mapping[str, Any]) -> float | None:
    assessed = _count(_get(stats, "assessed_count", "n_assessed"), "assessed_count")
    qualified = _count(_get(stats, "qualified_count", "n_qualified"), "qualified_count")
    if qualified > assessed:
        raise ValueError("qualified_count exceeds assessed_count")
    valid_count = _count(stats.get("valid_candidate_count", assessed), "valid_candidate_count")
    if valid_count > assessed or qualified > valid_count:
        raise ValueError("inconsistent valid_candidate_count")
    if assessed == 0 or valid_count == 0:
        # Preserve p when all cues are undefined; the paper gives no estimator.
        return None
    reliability = _unit(_get(stats, "mean_r", "mean_R"), "mean_r")
    consistency = _unit(_get(stats, "mean_s", "mean_S"), "mean_s")
    agreement = _unit(_get(stats, "mean_ae", "mean_AE"), "mean_ae")
    return .25*(qualified/assessed) + .35*reliability + .20*consistency + .20*agreement


def initialize_state(validation_stats: Mapping[str, Any], *, empty_p: float = .5) -> dict:
    """Initialize from validation p_raw, using empty_p when no candidate is valid."""
    q, h = _state_measurements(validation_stats)
    raw = _pseudo_raw(validation_stats)
    p = _unit(empty_p, "empty_p") if raw is None else raw
    return validate_state({"q": q, "p": p, "h": h})


def update_state(previous: Mapping[str, Any], completed_round_stats: Mapping[str, Any]) -> dict:
    """Compute next-round state without mutating inputs.

    Require sensor quality and candidate counts/scores. valid_candidate_count
    distinguishes all-invalid populations; score means are needed when nonempty."""
    previous = validate_state(previous)
    q, h = _state_measurements(completed_round_stats)
    raw = _pseudo_raw(completed_round_stats)
    p = previous["p"] if raw is None else .60*previous["p"] + .40*raw
    return validate_state({"q": q, "p": p, "h": h})


def state_demand(state: Mapping[str, Any]) -> list[float]:
    """Return eight demands in descriptor attribute order."""
    s = validate_state(state)
    x = np.array([1-s["q"], 1-s["p"], *s["h"]], dtype=np.float64)
    return np.clip(STATE_BIAS + STATE_MATRIX @ x, 0., 1.).tolist()


def state_specialist_scores(state: Mapping[str, Any]) -> list[float]:
    s = validate_state(state)
    bad_q, bad_p = 1-s["q"], 1-s["p"]
    weights = [(.15, .10, .75), (.10, .10, .80), (.08, .12, .80),
               (.05, .15, .80), (.25, .05, .70)]
    return [a*bad_q+b*bad_p+c*h for (a,b,c),h in zip(weights, s["h"])]


def state_specialist_probabilities(state: Mapping[str, Any], temperature: float = .5) -> dict[str, float]:
    """Return probabilities over specialists 1..5."""
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("state temperature must be positive and finite")
    logits = np.array(state_specialist_scores(state))/temperature
    p = np.exp(logits-logits.max())
    p /= p.sum()
    return {str(i+1): float(v) for i,v in enumerate(p)}


def state_relevance(state: Mapping[str, Any], prototype: Sequence[float], epsilon: float = 1e-8) -> float:
    """Return affine cosine relevance, including the general prototype e_0."""
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    a = np.asarray(state_specialist_scores(state), dtype=np.float64)
    e = np.asarray(prototype, dtype=np.float64)
    if e.shape != (5,) or not np.isfinite(e).all():
        raise ValueError("prototype must contain five finite values")
    na, ne = np.linalg.norm(a), np.linalg.norm(e)
    cosine = 0. if na == 0 or ne == 0 else float(np.dot(a, e)/(na*ne))
    return float(np.clip((1+cosine)/2, 0., 1.))
