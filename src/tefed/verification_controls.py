"""Compare verification references on one frozen rendered population."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from . import core
from .diagnostics import renderer_target_observation


def compare_frozen_verification(records,source_states,decision=None,*,run_seed,allow_incomplete=False):
    if type(run_seed) is not int or run_seed<0:
        raise ValueError("run_seed must be the recorded nonnegative integer seed")
    modes=("stored","current","cyclic")
    events={mode:[] for mode in modes}
    seen=set();incomplete=[]
    for row in records:
        if not row.get("entered_verification"):
            continue
        case_id=row.get("case_id")
        if not isinstance(case_id,str) or not case_id or case_id in seen:
            raise ValueError("rendered case IDs must be unique nonempty strings")
        seen.add(case_id)
        frozen=row.get("frozen_verification_inputs")
        if not isinstance(frozen,dict):
            raise ValueError("missing frozen verification inputs; use a run with complete variant audits")
        d=core.validate_descriptor(frozen["descriptor"],source_states)
        target={"box":frozen["target_box"],"class_id":frozen["target_class"]}
        renderer_target=frozen.get("renderer_target",target)
        if (renderer_target.get("class_id")!=target["class_id"]
            or any(abs(a-b)>1e-9 for a,b in zip(core._box(renderer_target["box"]),core._box(target["box"])) )):
            raise ValueError("compressed frozen predictions require identical verification and renderer target geometry")
        outputs=frozen["expert_outputs"]
        variant={"id":case_id,"target_box":target["box"],
                 "attrs_measured":frozen.get("attrs_measured"),"geometry":frozen.get("geometry")}
        for mode in modes:
            reference={"route":d["route"],"weights":d["weights"]} if mode=="stored" else frozen.get(mode+"_reference")
            if reference is None:
                if mode!="current":
                    raise ValueError("missing fixed-population cyclic route reference")
                result={"retained":False,"reason":"current_route_missing_general_anchor",
                        "first_failed_gate":"current_route_missing_general_anchor"}
                observation=None
            else:
                if mode=="cyclic":
                    from .server import _cyclic_reference
                    reference=_cyclic_reference({d["id"]:reference},d["id"],d["round"],run_seed)
                    if (any(c not in "0123456789abcdef" for c in reference["ordering_sha256"])
                        or type(reference["donor_position"]) is not int
                        or any(not isinstance(reference[key],str) or not reference[key] for key in
                            ("recipient_candidate_id","donor_client_id","donor_local_identity",
                             "donor_candidate_id","donor_source_observation_key"))):
                        raise ValueError("cyclic reference identities or ordering SHA-256 are invalid")
                routed={**deepcopy(d),"route":reference["route"],"weights":reference["weights"]}
                controlled=deepcopy(variant)
                associated={str(head):core.associate_hypothesis(core._outputs_get(outputs,head),target["box"])
                            for head in reference["route"]}
                posterior=core.verification_posterior({head:value["posterior"] if value else None
                    for head,value in associated.items()},reference["route"],reference["weights"],
                    tau_cov=core._cfg(decision,"tau_cov"))
                if posterior["passed"] and isinstance(controlled["attrs_measured"],list) and len(controlled["attrs_measured"])==8:
                    ordered=sorted(posterior["posterior"])
                    controlled["attrs_measured"][1]=max(0.,min(1.,1-(ordered[-1]-ordered[-2])))
                result=core.verify_case(controlled,routed,outputs,decision,source_states)
                if result["reason"] in {"undefined_geometry","missing_attribute_measurements","undefined_active_measurement"}:
                    incomplete.append({"case_id":case_id,"mode":mode,"reason":result["reason"]})
                observation=renderer_target_observation(outputs,reference["route"],reference["weights"],target,
                    scope="fixed rendered variant and frozen expert predictions")
            events[mode].append({"case_id":case_id,"reference":reference,"verification":result,
                                 "renderer_target_observation":observation})
    if not seen:
        raise ValueError("no rendered verification variants supplied")
    if incomplete and not allow_incomplete:
        raise ValueError("frozen population has missing measurements needed by a control; no complete comparison can be reported")
    summaries={}
    for mode,rows in events.items():
        retained=[row for row in rows if row["verification"]["retained"]]
        agreed=sum(row["renderer_target_observation"]["agreement"] for row in retained)
        complete=not any(row["mode"]==mode for row in incomplete)
        summaries[mode]={"rendered_count":len(rows),"retained_count":len(retained),
            "complete":complete,"retention_fraction":len(retained)/len(rows) if complete else None,
            "retained_rta_matches":agreed,"retained_rta_count":len(retained),
            "retained_rta":agreed/len(retained) if retained and complete else None,
            "first_failure_counts":dict(Counter(row["verification"]["first_failed_gate"]
                for row in rows if not row["verification"]["retained"]))}
    return {"kind":"fixed_variant_verification_controls","schema_version":1,
            "complete":not incomplete,"population_size":len(seen),"summary":summaries,
            "events":events,"incomplete":incomplete,
            "fixed_inputs":["rendered variant","target","descriptor attributes","nonclass measurements","geometry","expert predictions"],
            "varied_inputs":["verification route identities and normalized weights","class severity derived from route posterior"],
            "measurement_policy":"fixed geometry and nonclass severities; cls=1-(top1-top2) of each covered route posterior",
            "rta_definition":"micro proportion over retained cases, each counted once; renderer target is not independent ground truth",
            "count_alignment":"none; no historical K or percentages embedded"}


def analyze_frozen_run(run,*,allow_incomplete=False):
    root=Path(run).resolve()
    sources=[root/"configuration.json",root/"source_states.json",root/"status.json",root/"environment.json",*sorted(root.glob("verification_round_*.json"))]
    if len(sources)<5:
        raise ValueError("run has no verification records")
    payloads=[path.read_bytes() for path in sources]
    config=json.loads(payloads[0]);states=json.loads(payloads[1]);status=json.loads(payloads[2]);environment=json.loads(payloads[3])
    rounds=config.get("experiment",{}).get("rounds")
    if (type(rounds) is not int or rounds<=0 or status.get("status")!="completed"
        or type(status.get("completed_rounds")) is not int or status["completed_rounds"]!=rounds
        or {p.name for p in sources[4:]}!={f"verification_round_{t:03d}.json" for t in range(rounds)}):
        raise ValueError("fixed-variant controls require a completed run with every round audit")
    seed=environment.get("seed")
    if type(config["experiment"].get("seed")) is not int or config["experiment"]["seed"]!=seed:
        raise ValueError("configuration and recorded run seeds differ")
    records=[row for payload in payloads[4:] for row in json.loads(payload)]
    result=compare_frozen_verification(records,states,config.get("decision"),run_seed=seed,allow_incomplete=allow_incomplete)
    result["inputs"]=[{"path":str(path),"sha256":sha256(raw).hexdigest(),"bytes":len(raw)}
                      for path,raw in zip(sources,payloads)]
    return result
