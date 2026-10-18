"""TE-Fed routing, qualification, verification, reuse and aggregation.

Boxes use normalized xyxy; model state leaves are NumPy arrays."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import hmac
import json
from math import exp, isclose, isfinite, log
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .state import (ATTRIBUTES, FAMILIES, state_demand, state_relevance,
                    state_specialist_probabilities, validate_state)

BIN_EDGES = [
    [.20,.45,.70], [.25,.45,.65], [.15,.35,.60], [.18,.38,.62],
    [.20,.42,.68], [.18,.36,.58], [.22,.45,.70], [.20,.40,.65],
]
FIDELITY_TOLERANCES = [.18,.20,.18,.22,.20,.18,.22,.20]
FIDELITY_WEIGHTS = [.12,.15,.14,.12,.10,.16,.10,.11]
DESCRIPTOR_MATRIX = np.array([
    [.05,.05,.03,.58,.08,.05,.06,.10],
    [.10,.04,.58,.02,.08,.10,.06,.02],
    [.06,.06,.10,.03,.06,.07,.58,.04],
    [.06,.58,.04,.04,.05,.08,.05,.10],
    [.07,.04,.04,.06,.08,.14,.05,.52],
])
DEFAULTS = {
    "reliability_coefficients": [.22,.24,.16,.21,.24,.08],
    "delta_d": .35, "delta_c": .50, "delta_g": .50,
    "tau_cov": .80, "tau_cls": .05, "tau_e": .65,
    "verification_geometry_threshold": .60, "tau_r": .65,
    "epsilon": 1e-8, "reuse_cap": 6, "reuse_gate": .55,
    "pattern_cap": 2, "source_cap": 1, "tau_state": .50,
    "tau_s": 1., "lambda_s": .20, "allocation_tau": .50,
    "allocation_chi": .20, "priority_coefficients": [.15,.15,.25,.30,.25],
    "routing_temperature": .50,
}


def _cfg(config: Mapping | None, key: str, default: Any = None) -> Any:
    if config is not None:
        if key in config:
            return config[key]
        found = []
        def visit(obj):
            for k,v in obj.items():
                if k == key:
                    found.append(v)
                elif isinstance(v, Mapping):
                    visit(v)
        visit(config)
        if len(found) > 1:
            raise ValueError(f"ambiguous nested config key: {key}")
        if found:
            return found[0]
    return DEFAULTS.get(key, default)


def _real(x: Any, name: str, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x):
        raise ValueError(f"{name} must be a finite real")
    x = float(x)
    if (lower is not None and x < lower) or (upper is not None and x > upper):
        raise ValueError(f"{name} outside [{lower}, {upper}]")
    return x


def _vector(values, size: int, name: str, lower=0., upper=1.) -> list[float]:
    if values is None or len(values) != size:
        raise ValueError(f"{name} must contain {size} entries")
    return [_real(x, name, lower, upper) for x in values]


def _binary(values, size: int, name: str) -> list[int]:
    if values is None or len(values) != size:
        raise ValueError(f"{name} must contain {size} binary entries")
    if any(not isinstance(v, (int, np.integer, bool)) or v not in (0,1) for v in values):
        raise ValueError(f"{name} must contain binary integers")
    return [int(v) for v in values]


def _route(route, weights) -> tuple[list[int], list[float]]:
    if (len(route) != 3 or any(isinstance(v, bool) or not isinstance(v, (int,np.integer)) for v in route)
            or len(set(route)) != 3 or 0 not in route or not set(route) <= set(range(6))):
        raise ValueError("route requires general 0 and two distinct specialists 1..5")
    w = _vector(weights, 3, "route weights")
    if any(x <= 0 for x in w) or not isclose(sum(w), 1., abs_tol=1e-7, rel_tol=0):
        raise ValueError("route weights must be strictly positive and sum to one")
    return list(map(int,route)), w


def _outputs_get(outputs, expert):
    return outputs.get(str(expert), outputs.get(expert))


def _box(box) -> list[float]:
    b = _vector(box, 4, "normalized xyxy box")
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("box must have positive width and height")
    return b


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return IoU for normalized xyxy boxes; degenerate boxes give zero."""
    a, b = _vector(box_a, 4, "box_a"), _vector(box_b, 4, "box_b")
    aa = max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    ab = max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    inter = max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union = aa+ab-inter
    return inter/union if union > 0 else 0.


def _detection(detection: Mapping) -> dict:
    posterior = list(detection["posterior"])
    if len(posterior) < 2:
        raise ValueError("at least two class probabilities are required")
    p = _vector(posterior, len(posterior), "posterior")
    if not isclose(sum(p), 1., abs_tol=1e-6, rel_tol=0.):
        raise ValueError("posterior must sum to one")
    class_id = int(np.argmax(p))
    if "class_id" in detection and detection["class_id"] != class_id:
        raise ValueError("class_id does not match calibrated posterior argmax")
    return {**dict(detection), "box": _box(detection["box"]), "posterior": p,
            "class_id": class_id, "index": int(detection.get("index",0))}


def softmax(values: Sequence[float], temperature: float = 1.) -> list[float]:
    tau = _real(temperature, "temperature")
    if tau <= 0 or not len(values):
        raise ValueError("positive temperature and nonempty logits required")
    logits = np.asarray([_real(v,"logit") for v in values], dtype=np.float64)/tau
    probabilities = np.exp(logits-logits.max())
    probabilities /= probabilities.sum()
    return probabilities.tolist()


def route_scores(embedding: Sequence[float], prototypes: Mapping | Sequence, temperature: float=.5) -> dict:
    """Score six fixed prototypes from a normalized five-dimensional embedding."""
    z = np.asarray(_vector(embedding,5,"routing embedding",None,None))
    p = [_prototype(prototypes,m) for m in range(6)]
    scores = [float(np.dot(z,e)) for e in p]
    result = select_route(softmax(scores,temperature))
    return {**result, "scores": scores}


def select_route(probabilities: Sequence[float], top_k: int=2) -> dict:
    """Select general plus top two specialists; ties favor the lower ID."""
    if top_k != 2:
        raise ValueError("the paper's stored route requires exactly Top-2 specialists")
    p = _vector(probabilities,6,"routing probabilities")
    if not isclose(sum(p),1.,abs_tol=1e-6,rel_tol=0.):
        raise ValueError("routing probabilities must sum to one")
    # Remove float32 softmax rounding error before saving.
    total_probability=sum(p)
    p=[value/total_probability for value in p]
    specialists = sorted(range(1,6), key=lambda m:(-p[m],m))[:2]
    route = [0,*specialists]
    total = sum(p[m] for m in route)
    if total <= 0 or any(p[m] <= 0 for m in route):
        raise ValueError("stored active route probabilities must be positive")
    weights = [p[m]/total for m in route]
    return {"route": route, "weights": weights, "probabilities": p}


def associate_hypothesis(hypotheses, reference_box, class_id=None) -> dict | None:
    """Match at IoU >= .5, optionally by class; break ties by score then index."""
    if hypotheses is None:
        return None
    if isinstance(hypotheses,Mapping):
        hypotheses = [hypotheses]
    matches = []
    for item in hypotheses:
        try:
            d = _detection(item)
        except (ValueError,KeyError,TypeError):
            continue
        if class_id is not None and d["class_id"] != class_id:
            continue
        overlap = iou(d["box"],reference_box)
        if overlap >= .50:
            matches.append((overlap,max(d["posterior"]),d["index"],d))
    return min(matches,key=lambda x:(-x[0],-x[1],x[2]))[3] if matches else None


def expert_agreement(outputs: Mapping, route: Sequence[int], weights: Sequence[float],
                     target_box=None, mode: str="source") -> float:
    """Compute agreement with the original route-pair denominator; missing pairs give zero.

    Source outputs are anchor-associated. Verification associates by target_box
    geometry without filtering on the predicted class."""
    route,w = _route(route,weights)
    if mode not in ("source","verification"):
        raise ValueError("unknown expert agreement mode")
    selected = {}
    for m in route:
        output = _outputs_get(outputs,m)
        if mode == "verification":
            output = associate_hypothesis(output,target_box)
        elif output is not None:
            try:
                output = _detection(output)
            except (ValueError,KeyError,TypeError):
                output = None
        selected[m] = output
    if sum(d is not None for d in selected.values()) < 2:
        return 0.
    numerator,denominator = 0.,0.
    for i,m in enumerate(route):
        for j in range(i+1,len(route)):
            n = route[j]
            pair_weight = w[i]*w[j]
            denominator += pair_weight
            dm,dn = selected[m],selected[n]
            if dm is not None and dn is not None and dm["class_id"] == dn["class_id"]:
                numerator += pair_weight*iou(dm["box"],dn["box"])
    return float(np.clip(numerator/denominator,0.,1.))


def fuse_candidate(anchor: Mapping, expert_outputs: Mapping, route: Sequence[int], weights: Sequence[float]) -> dict:
    """Fuse matched experts, preserving original route mass for C and AE."""
    route,w = _route(route,weights)
    anchor = _detection(anchor)
    outputs = {str(0): anchor}
    for m in route:
        if m != 0:
            outputs[str(m)] = associate_hypothesis(_outputs_get(expert_outputs,m),anchor["box"],anchor["class_id"])
    n_classes = len(anchor["posterior"])
    for d in outputs.values():
        if d is not None and len(d["posterior"]) != n_classes:
            raise ValueError("expert class dimensions disagree")
    mass = sum(weight for m,weight in zip(route,w) if outputs[str(m)] is not None)
    box = sum(weight*np.asarray(outputs[str(m)]["box"]) for m,weight in zip(route,w) if outputs[str(m)] is not None)/mass
    posterior = sum(weight*np.asarray(outputs[str(m)]["posterior"]) for m,weight in zip(route,w) if outputs[str(m)] is not None)/mass
    # Clamp boundary rounding in the convex combination.
    box=np.clip(box,0.,1.)
    posterior=np.clip(posterior,0.,1.)
    posterior/=posterior.sum()
    cls = int(np.argmax(posterior))
    confidence = sum(weight*outputs[str(m)]["posterior"][cls] for m,weight in zip(route,w) if outputs[str(m)] is not None)
    agreement = expert_agreement(outputs,route,w)
    disagreement = 1-agreement-sum(weight*log(weight) for weight in w)/log(len(w))
    return {"id": str(anchor.get("id",anchor.get("index",0))), "anchor_id":str(anchor.get("id",anchor.get("index",0))),
            "box":box.tolist(),"posterior":posterior.tolist(),"class_id":cls,
            "route":route,"weights":w,"outputs":outputs,"C":float(confidence),
            "AE":agreement,"D_E":float(disagreement)}


def modality_agreement(reference: Mapping, modalities: Mapping, weights: Mapping | None=None) -> float:
    """Compute agreement using boxes mapped to RGB coordinates.

    None omits an unavailable branch. Empty predictions and class conflicts
    contribute zero while keeping their branch weight."""
    reference = _detection(reference)
    weights = weights or {"RGB":.40,"D":.35,"NIR":.25,
                          "rgb":.40,"depth":.35,"nir":.25}
    numerator,denominator = 0.,0.
    for name,hypotheses in modalities.items():
        if hypotheses is None:
            continue
        if name not in weights:
            raise ValueError(f"missing modality weight: {name}")
        weight = _real(weights[name],"modality weight",0.)
        denominator += weight
        # Modality overlap uses the best hypothesis without the .5 temporal cutoff.
        items = [hypotheses] if isinstance(hypotheses,Mapping) else hypotheses
        valid = []
        for item in items:
            try:
                d = _detection(item)
            except (ValueError,KeyError,TypeError):
                continue
            valid.append((iou(d["box"],reference["box"]),max(d["posterior"]),d["index"],d))
        if valid:
            overlap,_,_,d = min(valid,key=lambda x:(-x[0],-x[1],x[2]))
            if d["class_id"] == reference["class_id"]:
                numerator += weight*overlap
    return numerator/denominator if denominator > 0 else 0.


def candidate_reliability(cues: Mapping, weights: Sequence[float], config: Mapping | None=None) -> dict:
    """Return valid=False for missing or invalid scalar cues."""
    try:
        omitted_cross_modal = bool(_cfg(config,"omit_cross_modal",False))
        values = [_real(cues.get(k),k,0.,1.) if k != "AM" or not omitted_cross_modal
                  else 0. for k in ("C","AE","AM","G","S")]
        w = _vector(weights,3,"route weights")
        if any(x <= 0 for x in w) or not isclose(sum(w),1.,abs_tol=1e-7,rel_tol=0.):
            raise ValueError("route weights must be positive and normalized")
    except (ValueError,TypeError) as exc:
        return {"valid":False,"R":None,"D_E":None,"reason":str(exc)}
    d = 1-values[1]-sum(x*log(x) for x in w)/log(3)
    coefficients = _vector(_cfg(config,"reliability_coefficients"),6,"reliability coefficients",0.,None)
    raw = sum(a*b for a,b in zip(coefficients[:5],values))-coefficients[5]*d
    if omitted_cross_modal:
        raw -= coefficients[2]*values[2]
        denominator = sum(coefficients[:5])-coefficients[2]
        if denominator <= 0:
            raise ValueError("active positive reliability weight must be positive")
        raw /= denominator
    return {"valid":True,"R":float(np.clip(raw,0.,1.)),"D_E":float(d),"reason":"defined"}


def threshold_for_state(state: Mapping) -> float:
    """Frozen validation lookup. Exact edges belong to the upper interval."""
    s = validate_state(state)
    entries = [
        (s["q"],[.548,.697,.819,.917],[.007,.003,0.,-.005,-.009]),
        (s["p"],[.502,.619,.741,.847],[.006,.003,0.,-.004,-.007]),
        (max(s["h"]),[.079,.157,.242,.358],[-.004,-.001,.002,.005,.009]),
    ]
    offset = sum(offsets[int(np.searchsorted(edges,value,side="right"))] for value,edges,offsets in entries)
    return max(.70,.72+offset)


def qualify_candidate(candidate: Mapping, state: Mapping, config: Mapping | None=None) -> dict:
    """Classify one candidate, including candidates with invalid cues."""
    route,weights = _route(candidate["route"],candidate["weights"])
    cues = candidate.get("cues",candidate)
    reliability = candidate_reliability(cues,weights,config)
    result = {**deepcopy(dict(candidate)),**reliability,"route":route,"weights":weights,"threshold":threshold_for_state(state)}
    if not reliability["valid"]:
        result["disposition"] = "suppressed"
    elif result["R"] >= result["threshold"]:
        result["disposition"] = "qualified"
    elif (result["D_E"] >= _cfg(config,"delta_d") and cues["C"] >= _cfg(config,"delta_c") and cues["G"] >= _cfg(config,"delta_g")):
        result["disposition"] = "correction"
    else:
        result["disposition"] = "suppressed"
    return result


def attribute_bins(attrs: Sequence[float]) -> list[int]:
    attrs = _vector(attrs,8,"descriptor attributes")
    return [int(np.searchsorted(edges,value,side="right")) for edges,value in zip(BIN_EDGES,attrs)]


def descriptor_intervals(attrs: Sequence[float]) -> list[list[float]]:
    return [[float(([0.]+edges+[1.])[b]),float(([0.]+edges+[1.])[b+1])]
            for edges,b in zip(BIN_EDGES,attribute_bins(attrs))]


def fidelity_mask(attrs,availability,rule_flags) -> list[int]:
    availability = _binary(availability,8,"source availability")
    flags = _binary(rule_flags,5,"condition flags")
    mapped = [0,flags[3],flags[1],flags[0],0,0,flags[2],flags[4]]
    return [int(a and (b >= 1 or flag)) for a,b,flag in zip(availability,attribute_bins(attrs),mapped)]


def _canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def make_descriptor(candidate: Mapping, attrs, availability, rule_flags, client_id, local_id,
                    round_id: int, state: Mapping, run_secret: bytes | str) -> dict:
    """Encode an eligible rejection without observations or dense features."""
    if candidate.get("disposition") != "correction":
        raise ValueError("only correction-eligible rejections produce descriptors")
    availability = _binary(availability,8,"source availability")
    if attrs is None or len(attrs)!=8:
        raise ValueError("descriptor attributes require eight entries")
    # None marks an inactive coordinate and stays masked in fidelity checks.
    attrs = _vector([0. if x is None and not a else x for x,a in zip(attrs,availability)],8,"descriptor attributes")
    flags = _binary(rule_flags,5,"condition flags")
    route,weights = _route(candidate["route"],candidate["weights"])
    state = validate_state(state)
    if isinstance(round_id,bool) or not isinstance(round_id,int) or round_id < 0:
        raise ValueError("round_id must be a nonnegative integer")
    source_class = candidate.get("class_id")
    if isinstance(source_class,bool) or not isinstance(source_class,(int,np.integer)) or source_class < 0:
        raise ValueError("source class must be a nonnegative integer")
    secret = run_secret.encode() if isinstance(run_secret,str) else run_secret
    if not isinstance(secret,bytes) or not secret:
        raise ValueError("run_secret must be nonempty bytes or text")
    def key(fields):
        return hmac.new(secret,_canonical(fields),"sha256").hexdigest()[:32]
    source = key([str(client_id),str(local_id)])
    state_key = key([str(client_id),round_id,state])
    pattern = sum(4**q*b for q,b in enumerate(attribute_bins(attrs)))
    descriptor_id = sha256(_canonical([source,round_id,str(candidate.get("id",candidate.get("anchor_id","")))])).hexdigest()[:32]
    result = {"id":descriptor_id,"client_id":str(client_id),"round":round_id,
              "attrs":attrs,"availability":availability,"rule_flags":flags,
              "mask":fidelity_mask(attrs,availability,flags),"source_class":int(source_class),
              "route":route,"weights":weights,"pattern":pattern,"branch":pattern,
              "source":source,"source_state_key":state_key,"priority":0.}
    return validate_descriptor(result)


def validate_descriptor(descriptor: Mapping, state_lookup: Mapping | None=None, n_classes: int | None=None) -> dict:
    d = deepcopy(dict(descriptor))
    d["attrs"] = _vector(d["attrs"],8,"descriptor attributes")
    d["availability"] = _binary(d["availability"],8,"source availability")
    d["rule_flags"] = _binary(d["rule_flags"],5,"condition flags")
    d["route"],d["weights"] = _route(d["route"],d["weights"])
    cls = d.get("source_class")
    if isinstance(cls,bool) or not isinstance(cls,(int,np.integer)) or cls < 0 or (n_classes is not None and cls >= n_classes):
        raise ValueError("invalid source class")
    for key in ("id","source","source_state_key"):
        if not isinstance(d.get(key),str) or not d[key]:
            raise ValueError(f"missing descriptor {key}")
    expected_pattern = sum(4**q*b for q,b in enumerate(attribute_bins(d["attrs"])))
    if d.get("pattern",d.get("branch")) != expected_pattern:
        raise ValueError("pattern does not encode descriptor bins")
    d["pattern"] = d["branch"] = expected_pattern
    expected_mask = fidelity_mask(d["attrs"],d["availability"],d["rule_flags"])
    if "mask" in d and _binary(d["mask"],8,"fidelity mask") != expected_mask:
        raise ValueError("fidelity mask differs from fixed source availability/activation")
    d["mask"] = expected_mask
    d["priority"] = _real(d.get("priority",0.),"priority",0.)
    if state_lookup is not None:
        if d["source_state_key"] not in state_lookup:
            raise ValueError("unresolved immutable source-state key")
        validate_state(state_lookup[d["source_state_key"]])
    return d


def descriptor_projection(descriptor: Mapping, epsilon: float=1e-8) -> list[float]:
    d = validate_descriptor(descriptor)
    v = DESCRIPTOR_MATRIX @ (np.asarray(d["attrs"])*np.asarray(d["availability"]))
    eps = _real(epsilon,"epsilon")
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    return (v/(np.linalg.norm(v)+eps)).tolist()


def _prototype(prototypes, m: int) -> np.ndarray:
    raw = _outputs_get(prototypes,m) if isinstance(prototypes,Mapping) else prototypes[m]
    p = np.asarray(_vector(raw,5,"expert prototype",None,None),dtype=np.float64)
    if not isclose(float(np.linalg.norm(p)),1.,abs_tol=1e-6,rel_tol=0.):
        raise ValueError("expert prototypes must be fixed unit-normalized vectors")
    return p


def descriptor_relevance(descriptor: Mapping, prototypes) -> list[float]:
    z = np.asarray(descriptor_projection(descriptor))
    norm = float(np.linalg.norm(z))
    return [float(np.clip((1+(float(np.dot(z,_prototype(prototypes,m)))/norm if norm else 0.))/2,0.,1.)) for m in range(1,6)]


def validation_loss_score(descriptor: Mapping, loss_lookup: Mapping | None) -> float:
    """Read a frozen validation artifact; require all values."""
    d = validate_descriptor(descriptor)
    family_attrs = [3,2,6,1,7]
    active = [(d["attrs"][q],i,q) for i,q in enumerate(family_attrs) if d["rule_flags"][i] and d["availability"][q]]
    if not active:
        return 0.
    if loss_lookup is None:
        raise ValueError("active descriptors require a frozen validation loss lookup artifact")
    if str(d["pattern"]) in loss_lookup:
        return _real(loss_lookup[str(d["pattern"])],"lookup loss",0.,1.)
    _,family,q = min(active,key=lambda x:(-x[0],x[1]))
    bins = attribute_bins(d["attrs"])
    base = loss_lookup.get("base",loss_lookup)
    family_base = base.get(FAMILIES[family]) if isinstance(base,Mapping) else base[family]
    if family_base is None or len(family_base) != 4:
        raise ValueError("validation base lookup needs four bins per family")
    value = _real(family_base[bins[q]],"base lookup loss",0.,1.)
    geo = [0.,.0271,.0594,.1087][bins[5]]
    modifier = [0.,.0136,.0324,.0511][min(len(active),4)-1]
    return float(np.clip(value+geo+modifier,0.,1.))


def score_descriptors(descriptors: Sequence[Mapping], participating_clients: Sequence,
                      prototypes, loss_lookup: Mapping | None, config: Mapping | None=None) -> list[dict]:
    ids = [str(x) for x in participating_clients]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("participating clients must be nonempty and unique")
    ds = [validate_descriptor(d) for d in descriptors]
    if len({d["id"] for d in ds}) != len(ds):
        raise ValueError("duplicate descriptor id")
    if any(str(d.get("client_id")) not in ids for d in ds):
        raise ValueError("descriptor source is not a participating client")
    reporting = {}
    for d in ds:
        reporting.setdefault(d["pattern"],set()).add(str(d["client_id"]))
    alpha = _vector(_cfg(config,"priority_coefficients"),5,"priority coefficients",0.,None)
    chi = _real(_cfg(config,"allocation_chi"),"allocation_chi",0.)
    output = []
    for d in ds:
        rel = descriptor_relevance(d,prototypes)
        specialist_weights = [w for m,w in zip(d["route"],d["weights"]) if m]
        normalized = np.asarray(specialist_weights)/sum(specialist_weights)
        entropy = -float(np.sum(normalized*np.log(normalized)))/log(len(normalized)) if len(normalized)>1 else 0.
        recurrence = len(reporting[d["pattern"]])/len(ids)
        loss = validation_loss_score(d,loss_lookup)
        risk = sum(d["attrs"])/8
        values = [recurrence,entropy,loss,max(rel),risk]
        priority = max(0.,sum(a*b for a,b in zip(alpha[:4],values[:4]))-alpha[4]*risk)
        spec_weights = {m:w for m,w in zip(d["route"],d["weights"]) if m}
        scores = [r+chi*spec_weights.get(m,0.) for m,r in enumerate(rel,start=1)]
        allocation = softmax(scores,_cfg(config,"allocation_tau"))
        output.append({**d,"priority":float(priority),"priority_terms":dict(zip(("recurrence","entropy","loss","relevance","risk"),values)),
                       "assignment_probabilities":allocation})
    return output


def allocate_rendering(descriptors: Sequence[Mapping], budget: int=1152) -> dict:
    """Truncate to budget, then allocate integers with Hamilton rounding.

    Group ties within 1e-6 of the highest remaining priority, then sort by ID."""
    if isinstance(budget,bool) or not isinstance(budget,int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    validated = [validate_descriptor(d) | {"assignment_probabilities":d.get("assignment_probabilities")} for d in descriptors]
    if len({d["id"] for d in validated}) != len(validated):
        raise ValueError("duplicate descriptor id")
    ranked = sorted([d for d in validated if d["priority"] > 0],key=lambda d:(-d["priority"],d["id"]))
    ordered=[]
    while ranked:
        anchor=ranked[0]["priority"]
        group=[d for d in ranked if anchor-d["priority"] <= 1e-6]
        ordered.extend(sorted(group,key=lambda d:d["id"]))
        ranked=ranked[len(group):]
    remaining=budget
    allocations,audit=[],[]
    for d in ordered:
        request=max(3,min(8,3+round(5*d["priority"])))
        realized=min(request,remaining)
        if realized == 0:
            audit.append({"descriptor_id":d["id"],"reason":"budget_exhausted","requested":request,"realized":0})
            continue
        p=_vector(d["assignment_probabilities"],5,"assignment probabilities")
        if not isclose(sum(p),1.,abs_tol=1e-7,rel_tol=0.):
            raise ValueError("assignment probabilities must sum to one")
        quotas=np.asarray(p)*realized
        counts=np.floor(quotas).astype(int)
        order=sorted(range(5),key=lambda m:(-(quotas[m]-counts[m]),m))
        for m in order[:realized-int(counts.sum())]:
            counts[m]+=1
        if counts[int(np.argmax(p))] < 1:
            raise AssertionError("Hamilton allocation omitted the largest-weight specialist")
        for m,n in enumerate(counts,start=1):
            for j in range(int(n)):
                allocations.append({"descriptor":deepcopy(d),"specialist_id":m,"variant_index":j})
        remaining-=realized
        audit.append({"descriptor_id":d["id"],"reason":"allocated","requested":request,"realized":realized,"counts":counts.tolist()})
    for d in validated:
        if d["priority"] <= 0:
            audit.append({"descriptor_id":d["id"],"reason":"nonpositive_priority","requested":0,"realized":0})
    return {"allocations":allocations,"used":budget-remaining,"remaining":remaining,"audit":audit}


def variant_seed(run_seed,round_id,descriptor_id,specialist_id,variant_index) -> int:
    fields=[str(x).encode("utf-8") for x in (run_seed,round_id,descriptor_id,specialist_id,variant_index)]
    if any(b"\x1f" in f for f in fields):
        raise ValueError("seed fields may not contain the field separator")
    return int.from_bytes(sha256(b"\x1f".join(fields)).digest()[:4],"big")


def verification_posterior(expert_probs: Mapping,route,weights,*,tau_cov: float=.8) -> dict:
    """Check coverage before renormalizing the posterior."""
    route,w=_route(route,weights)
    tau=_real(tau_cov,"tau_cov",0.,1.)
    available=[]
    classes=None
    for m,weight in zip(route,w):
        raw=_outputs_get(expert_probs,m)
        if raw is None:
            continue
        if isinstance(raw,Mapping):
            raw=raw["posterior"]
        p=_vector(raw,len(raw),"expert posterior")
        if len(p)<2 or not isclose(sum(p),1.,abs_tol=1e-6,rel_tol=0.):
            raise ValueError("invalid expert posterior")
        if classes is not None and len(p)!=classes:
            raise ValueError("expert class dimensions disagree")
        classes=len(p)
        available.append((m,weight,p))
    coverage=sum(weight for _,weight,_ in available)
    result={"coverage":float(coverage),"posterior":None,"available_experts":[m for m,_,_ in available]}
    if coverage<=0 or coverage<tau:
        return {**result,"passed":False,"reason":"coverage"}
    posterior=[sum(w*p[c] for _,w,p in available)/coverage for c in range(classes)]
    return {**result,"posterior":posterior,"passed":True,"reason":"covered"}


def fidelity_audit(descriptor: Mapping,measured_attrs=None,*,evaluated=False,
                   checks=None,valid=None,score=None,reason="not_evaluated",
                   first_failed_gate=None) -> dict:
    """Record the fixed fidelity inputs, including unavailable measurements."""
    d=validate_descriptor(descriptor)
    raw=list(measured_attrs) if measured_attrs is not None and len(measured_attrs)==8 else [None]*8
    values,status=[],[]
    for active,x in zip(d["mask"],raw):
        finite=isinstance(x,Real) and not isinstance(x,bool) and isfinite(float(x))
        values.append(float(x) if finite else None)
        status.append("inactive" if not active else "undefined" if not finite else
                      "out_of_range" if not 0<=float(x)<=1 else "measured")
    return {"schema_version":1,"rule_version":"tefed-fidelity-v1",
            "attribute_names":list(ATTRIBUTES),"descriptor_attrs":list(d["attrs"]),
            "requested_intervals":descriptor_intervals(d["attrs"]),
            "measured_severities":values,"measurement_status":status,
            "active_mask":list(d["mask"]),"weights":list(FIDELITY_WEIGHTS),
            "normalizers":list(FIDELITY_TOLERANCES),
            "attribute_fidelities":list(checks) if checks is not None else [None]*8,
            "active_weight_mass":sum(w*m for w,m in zip(FIDELITY_WEIGHTS,d["mask"])),
            "evaluated":bool(evaluated),"valid":valid,"score":score,"reason":reason,
            "first_failed_gate":first_failed_gate}


def descriptor_fidelity(measured_attrs,descriptor: Mapping) -> dict:
    d=validate_descriptor(descriptor)
    def finish(valid,score,checks,reason):
        return {"valid":valid,"fidelity":score,"checks":checks,"reason":reason,
                "audit":fidelity_audit(d,measured_attrs,evaluated=True,checks=checks,
                    valid=valid,score=score,reason=reason,first_failed_gate=None if valid else "fidelity")}
    if measured_attrs is None or len(measured_attrs)!=8:
        return finish(False,None,[None]*8,"missing_attribute_measurements")
    mass=sum(w*m for w,m in zip(FIDELITY_WEIGHTS,d["mask"]))
    if mass<=0:
        return finish(False,None,[None]*8,"empty_fidelity_mask")
    checks=[]
    for i,(active,interval) in enumerate(zip(d["mask"],descriptor_intervals(d["attrs"]))):
        if not active:
            checks.append(None)
            continue
        try:
            x=_real(measured_attrs[i],"active realized attribute",0.,1.)
        except ValueError:
            return finish(False,None,checks+[None]*(8-len(checks)),"undefined_active_measurement")
        distance=max(interval[0]-x,0.,x-interval[1])
        checks.append(max(0.,1-distance/FIDELITY_TOLERANCES[i]))
    score=sum(w*v for w,v in zip(FIDELITY_WEIGHTS,checks) if v is not None)/mass
    return finish(True,float(np.clip(score,0.,1.)),checks,"measured")


def verify_case(variant: Mapping,descriptor: Mapping,expert_outputs: Mapping,
                config: Mapping | None=None,state_lookup: Mapping | None=None) -> dict:
    """Apply five retention gates using target geometry and the fixed source route.

    Require variant id, target_box, attrs_measured[8], and geometry, with
    predictions from the frozen round-start bank."""
    d=validate_descriptor(descriptor,state_lookup)
    result={"retained":False,"reason":"unassessed","coverage":None,"posterior":None,
            "margin":None,"agreement":None,"geometry":None,"fidelity":None,
            "fidelity_audit":fidelity_audit(d,variant.get("attrs_measured"))}
    def finish(reason,retained=False):
        failed=None if retained else ("fidelity" if reason in ("empty_fidelity_mask",
            "missing_attribute_measurements","undefined_active_measurement") else
            "geometry" if reason=="undefined_geometry" else reason)
        result["fidelity_audit"]["first_failed_gate"]=failed
        return {**result,"retained":retained,"reason":reason,"first_failed_gate":failed}
    if not any(d["mask"]):
        return finish("empty_fidelity_mask")
    target=_box(variant["target_box"])
    associated={str(m):associate_hypothesis(_outputs_get(expert_outputs,m),target) for m in d["route"]}
    posterior=verification_posterior({m:out["posterior"] if out is not None else None for m,out in associated.items()},d["route"],d["weights"],tau_cov=_cfg(config,"tau_cov"))
    result.update(coverage=posterior["coverage"],posterior=posterior["posterior"])
    if not posterior["passed"]:
        return finish("coverage")
    p=posterior["posterior"]
    source=d["source_class"]
    if source>=len(p):
        return finish("invalid_source_class")
    margin=p[source]-max(p[i] for i in range(len(p)) if i!=source)
    agreement=expert_agreement(associated,d["route"],d["weights"])
    result.update(margin=float(margin),agreement=agreement)
    try:
        geometry=_real(variant.get("geometry"),"geometry",0.,1.)
    except ValueError:
        return finish("undefined_geometry")
    fidelity=descriptor_fidelity(variant.get("attrs_measured",[None]*8),d)
    result.update(geometry=geometry,fidelity=fidelity["fidelity"],fidelity_audit=fidelity["audit"])
    if not fidelity["valid"]:
        return finish(fidelity["reason"])
    gates=[("margin",_cfg(config,"tau_cls")),("agreement",_cfg(config,"tau_e")),
           ("geometry",_cfg(config,"verification_geometry_threshold")),("fidelity",_cfg(config,"tau_r"))]
    for key,threshold in gates:
        if result[key]<_real(threshold,key+" threshold",0.,1.):
            return finish(key)
    return finish("retained",True)


def case_compatibility(case: Mapping,state: Mapping,config: Mapping | None=None) -> dict:
    d=validate_descriptor(case["d"])
    fidelity=_real(case["fidelity"],"stored fidelity",0.,1.)
    if not any(d["mask"]):
        raise ValueError("retained case has an empty fixed fidelity mask")
    eps=_real(_cfg(config,"epsilon"),"epsilon")
    if eps<=0:
        raise ValueError("epsilon must be positive")
    demand=state_demand(state)
    error=sum(abs(b-s) for b,s,m in zip(d["attrs"],demand,d["mask"]) if m)
    match=exp(-error/(.25*sum(d["mask"])+eps))
    probabilities=state_specialist_probabilities(state,_cfg(config,"tau_state"))
    overlap=float(np.clip(sum(probabilities[str(m)] for m in d["route"] if m),0.,1.))
    return {"compatibility":.65*match+.25*overlap+.10*fidelity,"attribute_match":match,"route_overlap":overlap,"fidelity":fidelity}


def select_reuse(pool: Sequence[Mapping],state: Mapping,round_id: int,config: Mapping | None=None) -> dict:
    """Select eligible prior-round cases by current state, rank, and diversity."""
    validate_state(state)
    if isinstance(round_id,bool) or not isinstance(round_id,int) or round_id<0:
        raise ValueError("round_id must be a nonnegative integer")
    cap=int(_cfg(config,"reuse_cap"))
    pattern_cap=int(_cfg(config,"pattern_cap"))
    source_cap=int(_cfg(config,"source_cap"))
    if min(cap,pattern_cap,source_cap)<0:
        raise ValueError("reuse and diversity caps cannot be negative")
    if len({case["id"] for case in pool})!=len(pool):
        raise ValueError("duplicate retained case id")
    audit,ranked=[],[]
    for case in pool:
        if case["round"]>=round_id:
            audit.append({"case_id":case["id"],"reason":"not_yet_eligible"})
            continue
        score=case_compatibility(case,state,config)
        entry={"case_id":case["id"],**score}
        if score["compatibility"]<_cfg(config,"reuse_gate"):
            audit.append({**entry,"reason":"compatibility_gate"})
        else:
            ranked.append((score["compatibility"],case,entry))
    ranked.sort(key=lambda x:(-x[0],-x[1]["fidelity"],-x[1]["d"]["priority"],str(x[1]["id"])))
    selected,patterns,sources=[],Counter(),Counter()
    for rank,(_,case,entry) in enumerate(ranked):
        pattern=case["d"]["pattern"]
        source=case["d"]["source"]
        if len(selected)>=cap:
            reason="reuse_cap"
        elif patterns[pattern]>=pattern_cap:
            reason="pattern_diversity"
        elif sources[source]>=source_cap:
            reason="source_diversity"
        else:
            selected.append(deepcopy(case))
            patterns[pattern]+=1
            sources[source]+=1
            reason="selected"
        audit.append({**entry,"rank":rank,"reason":reason})
    return {"selected":selected,"audit":audit}


def _flatten_tree(tree: Mapping) -> np.ndarray:
    arrays=[]
    for name in sorted(tree):
        value=tree[name]
        arrays.append(_flatten_tree(value) if isinstance(value,Mapping) else np.asarray(value,dtype=np.float64).ravel())
    return np.concatenate(arrays) if arrays else np.empty(0,dtype=np.float64)


def _same_tree(base: Mapping,delta: Mapping,name: str) -> None:
    if set(base)!=set(delta):
        raise ValueError(f"state structure mismatch: {name}")
    for key in base:
        if np.asarray(base[key]).shape!=np.asarray(delta[key]).shape or not np.isfinite(np.asarray(delta[key],dtype=float)).all():
            raise ValueError(f"invalid update tensor: {name}.{key}")


def _head(tree,m):
    return _outputs_get(tree["heads"],m)


def _projection(tree,m):
    return np.concatenate([_flatten_tree(tree["backbone"]),_flatten_tree(_head(tree,m))])


def aggregate_updates(base_state: Mapping,uploads: Sequence[Mapping],states: Mapping,
                      prototypes,previous_state: Mapping | None=None,config: Mapping | None=None) -> dict:
    """Aggregate bar_R-weighted updates, with one backbone update per client.

    Uploads contain client_id, delta, bar_R[6], and gain[6]. Use the preceding
    server state as previous_state, except at round 0. Inactive deltas are zeros."""
    eps=_real(_cfg(config,"epsilon"),"epsilon")
    tau=_real(_cfg(config,"tau_s"),"tau_s")
    step=_real(_cfg(config,"lambda_s"),"lambda_s",0.,1.)
    if eps<=0 or tau<=0:
        raise ValueError("epsilon and aggregation temperature must be positive")
    ids=[str(u["client_id"]) for u in uploads]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate client uploads")
    for m in range(6):
        if _head(base_state,m) is None:
            raise ValueError("base state requires six heads")
    for u in uploads:
        _same_tree(base_state["backbone"],u["delta"]["backbone"],"backbone")
        for m in range(6):
            _same_tree(_head(base_state,m),_head(u["delta"],m),f"head{m}")
    if previous_state is not None:
        _same_tree(base_state["backbone"],previous_state["backbone"],"previous backbone")
        for m in range(6):
            _same_tree(_head(base_state,m),_head(previous_state,m),f"previous head{m}")
    masses=np.zeros((len(uploads),6))
    activity=np.zeros_like(masses)
    details=[]
    for k,u in enumerate(uploads):
        state=states.get(ids[k],states.get(u["client_id"]))
        validate_state(state)
        bar_r=_vector(u["bar_R"],6,"mean candidate reliability")
        gains=_vector(u["gain"],6,"validation gains")
        client_details=[]
        for m in range(6):
            delta=_projection(u["delta"],m)
            activity[k,m]=float(np.linalg.norm(_flatten_tree(_head(u["delta"],m)))>0)
            reference=np.zeros_like(delta) if previous_state is None else _projection(base_state,m)-_projection(previous_state,m)
            norms=np.linalg.norm(delta)*np.linalg.norm(reference)
            conflict=0. if norms==0 else float(np.clip(.5*(1-float(np.dot(delta,reference)/norms)),0.,1.))
            relevance=state_relevance(state,_prototype(prototypes,m),eps)
            quality=relevance*gains[m]*bar_r[m]*(1-conflict)
            masses[k,m]=np.expm1(quality/tau)
            client_details.append({"expert":m,"relevance":relevance,"gain":gains[m],"bar_R":bar_r[m],"conflict":conflict,"quality":quality})
        details.append({"client_id":ids[k],"experts":client_details})
    head_weights=masses/(masses.sum(axis=0,keepdims=True)+eps)
    backbone_masses=(activity*masses).sum(axis=1)/(activity.sum(axis=1)+eps)
    backbone_weights=backbone_masses/(backbone_masses.sum()+eps)
    result={"backbone":{},"heads":{str(m):{} for m in range(6)}}
    for name,value in base_state["backbone"].items():
        base=np.asarray(value)
        change=sum((backbone_weights[k]*np.asarray(u["delta"]["backbone"][name]) for k,u in enumerate(uploads)),np.zeros_like(base,dtype=float))
        result["backbone"][name]=(base+step*change).astype(base.dtype,copy=False)
    for m in range(6):
        for name,value in _head(base_state,m).items():
            base=np.asarray(value)
            change=sum((head_weights[k,m]*np.asarray(_head(u["delta"],m)[name]) for k,u in enumerate(uploads)),np.zeros_like(base,dtype=float))
            result["heads"][str(m)][name]=(base+step*change).astype(base.dtype,copy=False)
    return {"state":result,"client_ids":ids,"masses":masses.tolist(),"head_weights":head_weights.tolist(),
            "backbone_masses":backbone_masses.tolist(),"backbone_weights":backbone_weights.tolist(),"details":details}
