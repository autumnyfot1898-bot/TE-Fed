"""Render and verify cases, then update the router after aggregation.

Retention uses Blender output, recomputed model cues and all five gates."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import hmac
from typing import Mapping

import numpy as np

from . import core
from .cues import extract_cues
from .state import FAMILIES
from .diagnostics import renderer_target_observation


def _services(config):
    """Inject Python-only test services; TOML cannot configure these."""
    return config.get("_server_services",{})


def _freeze(model):
    snapshot=deepcopy(model)
    snapshot.eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def _prototypes(router):
    value=router.prototypes
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=float).tolist()


def _assess(model,router,sample,state,calibration,config,round_index):
    callback=_services(config).get("assess_frame")
    if callback is None:
        from .engine import assess_frame
        callback=assess_frame
    return callback(model,router,sample,state,calibration,config,round_id=round_index)


def _predict_all(model,sample,calibration,config):
    callback=_services(config).get("predict_outputs")
    if callback is not None:
        return callback(model,sample,calibration,config)
    from .model import sample_tensor
    device=next(model.parameters()).device
    x=sample_tensor(sample,device,rgb_only=config.get("model",{}).get("detector_rgb_only",False))
    predictions=model.predict_heads(x,tuple(range(len(model.heads))),
        confidence=float(config.get("evaluation",{}).get("prediction_confidence",.01)),
        max_detections=int(config.get("evaluation",{}).get("max_detections",100)),
        temperature=float(calibration["T_C"]))
    counts=getattr(model,"last_proposal_counts",{})
    total=counts.get(0,[None])[0] if isinstance(counts,Mapping) else None
    return {"outputs":{str(m):predictions[m][0] for m in predictions},"proposal_count":total}


def _load_case(case,config):
    from .engine import apply_input_policy
    callback=_services(config).get("load_case")
    if callback is not None:
        sample=callback(case,config)
    elif isinstance(case.get("rgb"),np.ndarray):
        sample=deepcopy(case)
    else:
        from .data import Annotation, SampleRecord, load_sample
        record=SampleRecord(id=str(case["id"]),client_id=str(case.get("client_id","server")),
            split="labeled",rgb=Path(case["rgb"]),
            depth=Path(case["depth"]) if case.get("depth") else None,
            nir=Path(case["nir"]) if case.get("nir") else None,
            labels=tuple(Annotation(tuple(t["box"]),int(t["class_id"])) for t in case["targets"]))
        sample=load_sample(record,size=int(config.get("model",{}).get("image_size",640)))
    # Convert scene units, then apply the fixed client depth normalizer.
    normalizer=float(config.get("model",{}).get("depth_scale",1.))
    if not np.isfinite(normalizer) or normalizer<=0:
        raise ValueError("model.depth_scale network normalizer must be positive")
    conversion=case.get("depth_unit_conversion",case.get("depth_scale"))
    sample["depth_scale"]=normalizer
    sample["depth_unit_conversion"]=conversion
    if conversion is None:
        sample["depth"]=None
        sample["depth_units_status"]="unknown_scene_conversion_depth_omitted"
    else:
        conversion=float(conversion)
        if not np.isfinite(conversion) or conversion<=0:
            raise ValueError("depth_unit_conversion must be positive")
        applied=sample.get("_depth_unit_conversion_applied")
        if applied is not None and not np.isclose(float(applied),conversion,rtol=0.,atol=1e-12):
            raise ValueError("sample was already converted using a different depth unit factor")
        if sample.get("depth") is not None and applied is None:
            sample["depth"]=np.asarray(sample["depth"])*conversion
        sample["_depth_unit_conversion_applied"]=conversion
        sample["depth_units_status"]="converted_to_dataset_units"
    return apply_input_policy(sample,config)


def _render_backend(config):
    service=_services(config).get("renderer")
    if service is not None:
        return service
    from .rendering import BlenderRenderer
    cfg=config.get("renderer",{})
    return BlenderRenderer(cfg.get("blender","blender"),cfg.get("scene"),
                           timeout=float(cfg.get("timeout",600)),samples=int(cfg.get("samples",8)),
                           script_path=cfg.get("script_path"))


def _generic_descriptor(descriptor,calibration):
    d=deepcopy(descriptor)
    family_indices=[3,2,6,1,7]
    active=[(d["attrs"][q],i) for i,q in enumerate(family_indices) if d["rule_flags"][i] and d["availability"][q]]
    if not active:
        raise ValueError("Generic HC has no active family for its calibrated coarse prior")
    family=min(active,key=lambda x:(-x[0],x[1]))[1]
    value=calibration.get("generic_priors",{}).get(FAMILIES[family])
    if value is None:
        raise ValueError(f"missing held-out generic prior for {FAMILIES[family]}")
    if isinstance(value,Mapping):
        attrs=value["attrs"]
        if "availability" in value:
            d["availability"]=[int(a and b) for a,b in zip(d["availability"],value["availability"])]
    else:
        attrs=value
    d["attrs"]=core._vector(attrs,8,"generic prior attributes")
    d["rule_flags"]=[int(i==family) for i in range(5)]
    d["mask"]=core.fidelity_mask(d["attrs"],d["availability"],d["rule_flags"])
    d["pattern"]=d["branch"]=sum(4**i*b for i,b in enumerate(core.attribute_bins(d["attrs"])))
    return core.validate_descriptor(d)


def _current_route(assessment,target_box):
    candidates=assessment.get("candidates",[])
    # Derive the route from a detected general anchor.
    matches=[c for c in candidates if core.iou(c["box"],target_box)>=.5 and len(c.get("route",[]))==3]
    if not matches:
        return None
    chosen=min(matches,key=lambda c:(-core.iou(c["box"],target_box),-max(c["posterior"]),str(c.get("id",""))))
    return {"route":list(chosen["route"]),"weights":list(chosen["weights"])}


def build_cyclic_routes(candidate_events,manifest,run_secret,round_index,*,run_seed):
    """Shift complete candidate routes by eleven positions within a round."""
    secret=run_secret.encode() if isinstance(run_secret,str) else run_secret
    if not isinstance(secret,bytes) or not secret:
        raise ValueError('Cyclic route control requires the source identifier key')
    identities={}
    for record in manifest.by_split('unlabeled'):
        identity=hmac.new(secret,str(record.id).encode(),'sha256').hexdigest()[:32]
        identities[(str(record.client_id),identity)]=str(record.id)
    population=[];seen=set()
    for event in candidate_events:
        if event.get('round_index')!=round_index:
            raise ValueError('Cyclic candidate population mixes rounds')
        client=str(event['client_id']);local_identity=str(event['local_identity'])
        image_id=identities.get((client,local_identity))
        if image_id is None:
            raise ValueError('Cyclic candidate observation is not in the unlabeled manifest')
        candidate_id=str(event.get('candidate_id',event.get('id','')))
        key=(image_id,candidate_id)
        if not candidate_id or key in seen:
            raise ValueError('Cyclic candidate identities must be present and unique')
        seen.add(key)
        route,weights=core._route(event['route'],event['weights'])
        source=hmac.new(secret,core._canonical([client,image_id]),'sha256').hexdigest()[:32]
        if event.get('source_observation_key',source)!=source:
            raise ValueError('Cyclic candidate source key differs')
        population.append({'image_id':image_id,'client_id':client,'local_identity':local_identity,
            'candidate_id':candidate_id,'descriptor_id':event.get('descriptor_id'),
            'source_observation_key':source,'route':route,'weights':weights})
    population.sort(key=lambda row:(row['image_id'],row['candidate_id']))
    ordering=[{key:row[key] for key in ('client_id','local_identity','candidate_id')} for row in population]
    ordering_sha256=sha256(core._canonical(ordering)).hexdigest()
    result={};count=len(population)
    for index,recipient in enumerate(population):
        descriptor_id=recipient['descriptor_id']
        if descriptor_id is None:
            continue
        if descriptor_id in result:
            raise ValueError('Cyclic descriptor identity repeats')
        donor_index=(index+11)%count;donor=population[donor_index]
        result[str(descriptor_id)]={'route':list(donor['route']),'weights':list(donor['weights']),
            'profile':'candidate-route-tuple-shift11-v1','shift':11,'population_size':count,
            'ordering_fields':['manifest_image_id','candidate_id'],'ordering_sha256':ordering_sha256,
            'run_seed':int(run_seed),'round_index':int(round_index),'recipient_position':index,
            'recipient_descriptor_id':str(descriptor_id),'recipient_candidate_id':recipient['candidate_id'],
            'donor_position':donor_index,'donor_client_id':donor['client_id'],
            'donor_local_identity':donor['local_identity'],'donor_candidate_id':donor['candidate_id'],
            'donor_source_observation_key':donor['source_observation_key']}
    return result


def _cyclic_reference(cyclic_routes,descriptor_id,round_index,run_seed):
    if not isinstance(cyclic_routes,Mapping) or descriptor_id not in cyclic_routes:
        raise ValueError('Cyclic verification requires the complete candidate route mapping')
    reference=deepcopy(cyclic_routes[descriptor_id])
    required={'profile','shift','population_size','ordering_fields','ordering_sha256','run_seed','round_index',
        'recipient_position','recipient_descriptor_id','recipient_candidate_id','donor_position',
        'donor_client_id','donor_local_identity','donor_candidate_id','donor_source_observation_key','route','weights'}
    if not isinstance(reference,Mapping) or not required<=set(reference):
        raise ValueError('Cyclic route reference is incomplete')
    count=reference['population_size'];index=reference['recipient_position']
    if (reference['profile']!='candidate-route-tuple-shift11-v1' or reference['shift']!=11
            or reference['ordering_fields']!=['manifest_image_id','candidate_id']
            or reference['run_seed']!=run_seed or reference['round_index']!=round_index
            or reference['recipient_descriptor_id']!=descriptor_id
            or type(count) is not int or count<1 or type(index) is not int or not 0<=index<count
            or reference['donor_position']!=(index+11)%count
            or not isinstance(reference['ordering_sha256'],str) or len(reference['ordering_sha256'])!=64):
        raise ValueError('Cyclic route reference differs from the declared shift protocol')
    route,weights=core._route(reference['route'],reference['weights'])
    reference.update(route=route,weights=weights)
    return reference


def construct_cases(descriptors,model,router,source_states,participating_clients,
                    calibration,config,round_index,run_seed,output_dir,*,cyclic_routes=None):
    """Return retained cases and exclusions before server aggregation.

    A frozen copy protects the round-start bank. d preserves the source route;
    route controls live separately in verification_reference."""
    output_dir=Path(output_dir)
    method=str(config.get("experiment",{}).get("method","tefed"))
    route_mode=config.get("decision",{}).get("verification_route","stored")
    if method=="generic_hc":
        route_mode="current"
    if route_mode not in ("stored","current","cyclic"):
        raise ValueError("verification_route must be stored/current/cyclic")
    if not config.get("renderer",{}).get("enabled",True):
        return {"cases":[],"audit":[{"reason":"renderer_disabled"}],"scored_descriptors":[]}
    valid,audit=[],[]
    seen=set()
    for raw in descriptors:
        try:
            d=core.validate_descriptor(raw,source_states,n_classes=int(config.get("model",{}).get("classes",2)))
            if d["id"] in seen:
                raise ValueError("duplicate descriptor id")
            seen.add(d["id"])
            valid.append(d)
        except (ValueError,KeyError,TypeError) as exc:
            audit.append({"descriptor_id":raw.get("id"),"reason":"descriptor_rejected","detail":str(exc)})
    if not valid:
        return {"cases":[],"audit":audit,"scored_descriptors":[]}
    if route_mode=='cyclic':
        for descriptor in valid:
            _cyclic_reference(cyclic_routes,descriptor['id'],round_index,run_seed)
    frozen_model,frozen_router=_freeze(model),_freeze(router)
    scored=core.score_descriptors(valid,participating_clients,_prototypes(frozen_router),calibration.get("loss_lookup"),config.get("decision"))
    schedule=core.allocate_rendering(scored,int(config.get("decision",{}).get("render_budget",1152)))
    audit.extend(schedule["audit"])
    if not schedule["allocations"]:
        return {"cases":[],"audit":audit,"scored_descriptors":scored}
    renderer=_render_backend(config)
    cases=[]
    for allocation in schedule["allocations"]:
        original=allocation["descriptor"]
        try:
            rendering_d=_generic_descriptor(original,calibration) if method=="generic_hc" else deepcopy(original)
        except ValueError as exc:
            audit.append({"descriptor_id":original["id"],"reason":"generic_prior_invalid","detail":str(exc)})
            continue
        specialist=allocation["specialist_id"]
        index=allocation["variant_index"]
        seed=core.variant_seed(run_seed,round_index,original["id"],specialist,index)
        case_id=sha256(f"{run_seed}:{round_index}:{original['id']}:{specialist}:{index}".encode()).hexdigest()[:32]
        identity={"case_id":case_id,"rendered_variant_id":case_id,
                  "descriptor_id":original["id"],"creation_round":int(round_index),
                  "source_observation_key":original["source"],"source_client_id":original["client_id"],
                  "source_state_key":original["source_state_key"],"pattern_key":original["pattern"],
                  "source_route":list(original["route"]),"source_weights":list(original["weights"])}
        folder=output_dir/case_id
        request={"descriptor":rendering_d,"seed":seed,"case_id":case_id,
                 "specialist_id":specialist,"variant_index":index,
                 "image_size":int(config.get("model",{}).get("image_size",640))}
        if config.get("renderer",{}).get("scene_class_map") is not None:
            request["scene_class_map"]=deepcopy(config["renderer"]["scene_class_map"])
        if calibration.get("luminance_median") is not None:
            request["descriptor"]={**rendering_d,"calibration":{"luminance_median":calibration["luminance_median"]}}
        response=renderer.render(request,folder)
        depth_scale=config.get("renderer",{}).get("depth_unit_conversion",config.get("renderer",{}).get("depth_scale"))
        response={**response,"id":case_id,"depth_unit_conversion":depth_scale}
        sample=_load_case(response,config)
        measured_calibration=deepcopy(calibration)
        if depth_scale is None:
            # Convert Blender depth to dataset units.
            measured_calibration["depth_validation_median"]=None
            measured_calibration.pop("median_validation_depth",None)
        else:
            depth_scale=float(depth_scale)
            if not np.isfinite(depth_scale) or depth_scale<=0:
                raise ValueError("renderer.depth_scale must be positive")
        source_state=source_states[original["source_state_key"]]
        assessment=_assess(frozen_model,frozen_router,sample,source_state,measured_calibration,config,round_index)
        actual=_predict_all(frozen_model,sample,measured_calibration,config)
        target_index=int(response.get("focal_target_index",0))
        target_box=response.get("target_box",sample["targets"][target_index]["box"])
        verification_d=deepcopy(rendering_d)
        current_reference=_current_route(assessment,target_box)
        cyclic_reference=_cyclic_reference(cyclic_routes,original['id'],round_index,run_seed) if cyclic_routes is not None else None
        frozen_outputs={}
        for head in range(6):
            matched=core.associate_hypothesis(core._outputs_get(actual['outputs'],head),target_box)
            frozen_outputs[str(head)]=[deepcopy(matched)] if matched is not None else []
        frozen_inputs={'descriptor':deepcopy(rendering_d),'target_box':list(target_box),
            'original_source_descriptor':deepcopy(original),'target_class':int(sample['targets'][target_index]['class_id']),
            'renderer_target':deepcopy(sample['targets'][target_index]),
            'attrs_measured':None,'geometry':None,'expert_outputs':frozen_outputs,
            'current_reference':deepcopy(current_reference),'cyclic_reference':deepcopy(cyclic_reference)}
        if route_mode=="current":
            current=current_reference
            if current is None:
                reason="current_route_missing_general_anchor"
                audit.append({**identity,"reason":reason,"first_failed_gate":reason,
                              "verification_route":route_mode,"entered_verification":True,"retained":False,
                              "frozen_verification_inputs":frozen_inputs,
                              "fidelity_audit":core.fidelity_audit(verification_d,first_failed_gate=reason)})
                continue
            verification_d.update(current)
        elif route_mode=="cyclic":
            verification_d.update(route=list(cyclic_reference['route']),weights=list(cyclic_reference['weights']))
        verification_reference=deepcopy(cyclic_reference) if route_mode=='cyclic' else {
            'route':list(verification_d['route']),'weights':list(verification_d['weights'])}
        expert_outputs=actual["outputs"]
        rta=renderer_target_observation(expert_outputs,verification_d['route'],verification_d['weights'],
            sample['targets'][target_index],scope='round-start verification bank')
        associated={str(m):core.associate_hypothesis(core._outputs_get(expert_outputs,m),target_box) for m in verification_d["route"]}
        posterior=core.verification_posterior({m:d["posterior"] if d else None for m,d in associated.items()},
            verification_d["route"],verification_d["weights"],tau_cov=float(config.get("decision",{}).get("tau_cov",.8)))
        if not posterior["passed"]:
            audit.append({**identity,"reason":"coverage","first_failed_gate":"coverage",
                          "verification_route":route_mode,
                          "verification_reference":verification_reference,
                          "frozen_verification_inputs":frozen_inputs,
                          "entered_verification":True,"retained":False,"rta_observation":rta,
                          "fidelity_audit":core.fidelity_audit(verification_d,first_failed_gate="coverage"),**posterior})
            continue
        # Use rendered geometry and the frozen target-associated route posterior.
        measurement_anchor={"box":target_box,"posterior":posterior["posterior"],"index":target_index}
        evidence={**sample,"foreground_detections":core._outputs_get(expert_outputs,0)}
        if actual.get("proposal_count") is not None:
            evidence.update(proposal_count=int(actual["proposal_count"]),proposal_count_includes_target=True)
        cues=extract_cues(evidence,measurement_anchor,assessment.get("modalities",{}),[assessment.get("references",[])],measured_calibration)
        renderer_attrs=list(response["attrs_measured"])
        measured=list(cues["attrs"])
        for q in (0,2):
            # Use visible-mask size and occlusion measurements.
            measured[q]=renderer_attrs[q]
        variant={"id":case_id,"target_box":target_box,"attrs_measured":measured,"geometry":cues["G"]}
        frozen_inputs.update(attrs_measured=deepcopy(measured),geometry=cues['G'])
        verification=core.verify_case(variant,verification_d,expert_outputs,config.get("decision"),source_states)
        entry={**identity,"verification_route":route_mode,
               "entered_verification":True,"rta_observation":rta,
               "verification_reference":verification_reference,"frozen_verification_inputs":frozen_inputs,
               "renderer_measurements":renderer_attrs,"fidelity_measurements":measured,
               "cue_audit":cues["cue_audit"],"depth_scale_configured":depth_scale is not None,**verification}
        audit.append(entry)
        if verification["retained"]:
            cases.append({"id":case_id,"round":int(round_index),"d":deepcopy(rendering_d),
                "fidelity":verification["fidelity"],"rgb":response["rgb"],"depth":response.get("depth"),
                "nir":response.get("nir"),"targets":deepcopy(response["targets"]),
                "focal_target_index":target_index,"target_box":list(target_box),
                "seed":seed,"render_specialist":specialist,"attrs_measured":measured,
                "original_source_route":{"route":original["route"],"weights":original["weights"]},
                "original_source_descriptor":deepcopy(original),
                "verification_reference":entry["verification_reference"],"verification":verification,
                "rta_observation":rta,
                "render_metadata":response.get("metadata",{}),
                "depth_unit_conversion":depth_scale,"depth_scale":depth_scale,
                "depth_network_normalizer":float(config.get("model",{}).get("depth_scale",1.))})
    return {"cases":cases,"audit":audit,"scored_descriptors":scored}


def router_objective(observation_probabilities,descriptor_probabilities,
                     observation_targets,descriptor_targets,load_balance=.02):
    """Compute the complete-bank objective with fixed descriptor probabilities."""
    import torch
    descriptors=descriptor_probabilities.detach()
    probabilities=torch.cat([observation_probabilities,descriptors],dim=0)
    if not len(probabilities):
        raise ValueError("router objective needs a nonempty probe bank")
    targets=torch.cat([observation_targets,descriptor_targets],dim=0).long()
    if probabilities.shape!=(len(targets),6):
        raise ValueError("router probes require six probabilities and one target each")
    target_loss=-probabilities.clamp_min(1e-12).log().gather(1,targets[:,None]).mean()
    balance=((probabilities.mean(0)-1/6)**2).sum()
    return {"loss":target_loss+float(load_balance)*balance,"target_loss":target_loss,"balance_loss":balance}


def _probe_sample(sample,model,router,state,calibration,config,round_index,target_index=0):
    assessed=_assess(model,router,sample,state,calibration,config,round_index)
    if not sample.get("targets"):
        return None,"probe_has_no_labeled_target"
    target=sample["targets"][target_index]
    candidates=assessed.get("candidates",[])
    matched=[(core.iou(c["box"],target["box"]),i,c) for i,c in enumerate(candidates) if core.iou(c["box"],target["box"])>=.5]
    if not matched:
        return None,"probe_missing_general_anchor"
    _,index,candidate=min(matched,key=lambda x:(-x[0],str(x[2].get("id",""))))
    outputs=_predict_all(model,sample,calibration,config)["outputs"]
    compatible=[]
    callback=_services(config).get("native_losses")
    losses=callback(model,sample,calibration,config) if callback is not None else None
    for m in range(6):
        associated=core.associate_hypothesis(core._outputs_get(outputs,m),target["box"],target["class_id"])
        if associated is not None:
            if losses is None:
                losses=_native_losses(model,sample,config)
            value=float(losses[m] if not isinstance(losses,Mapping) else losses.get(m,losses.get(str(m))))
            if np.isfinite(value):
                compatible.append((value,m))
    if not compatible:
        return None,"probe_has_no_compatible_expert"
    target_expert=min(compatible)[1]
    vector=candidate["routing_vector"] if "routing_vector" in candidate else assessed["vectors"][index]
    return {"vector":vector,"target":target_expert,"sample_id":sample["id"]},None


def _native_losses(model,sample,config):
    import torch
    from .losses import WeightedNativeLoss
    from .model import sample_tensor
    device=next(model.parameters()).device
    x=sample_tensor(sample,device,rgb_only=config.get("model",{}).get("detector_rgb_only",False))
    with torch.no_grad():
        raw=model(x,tuple(range(6)))
        return {m:float(WeightedNativeLoss(model,m)(raw[m],sample["targets"],[1.]*len(sample["targets"]),device)[0]) for m in range(6)}


def update_router(model,router,manifest,pool,source_states,client_states,calibration,config,round_index,*,descriptors=()):
    """Update the router after aggregation using validation and retained cases.

    Use three photometric variants, up to 32 retained cases and 24 descriptors.
    Calibration uses the first annotation; test IDs are excluded."""
    import torch
    from .data import load_sample
    from .engine import apply_input_policy
    frozen_model,frozen_router=_freeze(model),_freeze(router)
    audit=[]
    observations=[]
    cfg=config.get("router",{})
    service=_services(config)
    for sample_id in manifest.calibration_ids:
        record=manifest.by_id[str(sample_id)]
        if record.split!="validation":
            raise ValueError("router calibration ids must remain validation identities")
        recorded=calibration.get("probe_states",{})
        state=recorded.get(str(sample_id),recorded.get(str(record.client_id)))
        state_origin="recorded_validation_state"
        if state is None:
            state=client_states.get(str(record.client_id),client_states.get(record.client_id))
            state_origin="explicit_client_state_argument"
        if state is None:
            audit.append({"sample_id":str(sample_id),"reason":"calibration_state_missing"});continue
        loader=service.get("load_sample",load_sample)
        sample=loader(record,size=int(config.get("model",{}).get("image_size",640)))
        sample=apply_input_policy(sample,config)
        for variant,scale in enumerate((.95,1.,1.05)):
            view=deepcopy(sample)
            view["rgb"]=np.clip(np.asarray(sample["rgb"],dtype=float)*scale,0,255).astype(np.uint8)
            probe,reason=_probe_sample(view,frozen_model,frozen_router,state,calibration,config,round_index)
            if probe is None:
                audit.append({"sample_id":str(sample_id),"variant":variant,"reason":reason})
            else:
                observations.append({**probe,"state_origin":state_origin,"type":"calibration","variant":variant})
    prior=[c for c in pool if c["round"]<round_index]
    retained=sorted(prior,key=lambda c:(-float(c["fidelity"]),str(c["id"])))[:int(cfg.get("retained_probe_cap",32))]
    for case in retained:
        state=source_states.get(case["d"]["source_state_key"])
        if state is None:
            audit.append({"case_id":case["id"],"reason":"retained_source_state_missing"});continue
        sample=_load_case(case,config)
        probe,reason=_probe_sample(sample,frozen_model,frozen_router,state,calibration,config,round_index,int(case.get("focal_target_index",0)))
        if probe is None:
            audit.append({"case_id":case["id"],"reason":reason})
        else:
            observations.append({**probe,"state_origin":"immutable_source_state","type":"retained"})
    validated=[]
    for raw in descriptors:
        try:
            validated.append(core.validate_descriptor(raw,source_states))
        except (ValueError,KeyError,TypeError) as exc:
            audit.append({"descriptor_id":raw.get("id"),"reason":"router_descriptor_invalid","detail":str(exc)})
    if validated and any("priority_terms" not in d for d in validated):
        # Descriptor probes do not render, so B_round=0 still permits them.
        active_clients=config.get("experiment",{}).get("active_client_ids",list(client_states))
        validated=core.score_descriptors(validated,active_clients,_prototypes(frozen_router),calibration.get("loss_lookup"),config.get("decision"))
    unique={}
    for d in validated:
        if d["id"] in unique:
            raise ValueError("duplicate uploaded descriptor id in router probes")
        unique[d["id"]]=d
    descriptor_sources=sorted([d for d in unique.values() if d["priority"]>0],key=lambda d:(-d["priority"],d["id"]))[:int(cfg.get("descriptor_probe_cap",24))]
    fixed=[];fixed_targets=[]
    prototypes=_prototypes(frozen_router)
    for d in descriptor_sources:
        z=core.descriptor_projection(d)
        fixed.append(core.softmax([float(np.dot(z,e)) for e in prototypes],float(config.get("decision",{}).get("routing_temperature",.5))))
        fixed_targets.append(min([( -w,m) for m,w in zip(d["route"],d["weights"]) if m])[1])
    report={"audit":audit,"observation_probes":len(observations),"descriptor_probes":len(fixed),
            "descriptor_population":"positive_priority_uploaded_descriptors",
            "observation_targets":[{k:v for k,v in p.items() if k!="vector"} for p in observations],
            "model_bank_version":"after_expert_aggregation","steps":0}
    if not observations:
        return {**report,"status":"no_observation_gradient"}
    device=next(router.parameters()).device
    vectors=torch.tensor([p["vector"] for p in observations],dtype=torch.float32,device=device)
    targets=torch.tensor([p["target"] for p in observations],dtype=torch.long,device=device)
    fixed_probabilities=torch.tensor(fixed,dtype=torch.float32,device=device).reshape(-1,6)
    descriptor_targets=torch.tensor(fixed_targets,dtype=torch.long,device=device)
    optimizer=torch.optim.Adam(router.parameters(),lr=float(cfg.get("learning_rate",.001)))
    training=router.training
    router.train()
    try:
        for step in range(int(cfg.get("steps",20))):
            optimizer.zero_grad(set_to_none=True)
            probabilities=router(vectors)
            losses=router_objective(probabilities,fixed_probabilities,targets,descriptor_targets,float(cfg.get("load_balance",.01)))
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("nonfinite router loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(),10.)
            optimizer.step()
            report.update(steps=step+1,loss=float(losses["loss"].detach()),target_loss=float(losses["target_loss"].detach()),balance_loss=float(losses["balance_loss"].detach()))
    finally:
        router.train(training)
    return {**report,"status":"updated"}
