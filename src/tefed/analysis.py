"""Analyze measured runs and device records; reject missing or inconsistent evidence."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DEFAULT_SEEDS=tuple(range(10))
ENERGY_NODES=("J1","J2","J3","J4","serverGPU")


def _finite(value,name):
    if isinstance(value,bool) or not isinstance(value,Real) or not isfinite(value):
        raise ValueError(f"{name} must be a defined finite numeric value")
    return float(value)


def _integer(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _seed_ids(seeds):
    result=[str(s) for s in seeds]
    if len(result)<2 or len(set(result))!=len(result):
        raise ValueError("sample SD requires at least two distinct declared seeds")
    return result


def summarize_seed_metric(records,metric="value",expected_seeds=DEFAULT_SEEDS):
    """Return mean and sample SD for the declared seed set."""
    expected=_seed_ids(expected_seeds)
    values={}
    for row in records:
        seed=str(row["seed"])
        if seed in values:
            raise ValueError(f"duplicate seed {seed}")
        values[seed]=_finite(row.get(metric),f"seed {seed} metric {metric}")
    if set(values)!=set(expected):
        raise ValueError(f"seed mismatch: missing={sorted(set(expected)-set(values))}, extra={sorted(set(values)-set(expected))}")
    array=np.array([values[s] for s in expected],dtype=float)
    return {"mean":float(array.mean()),"sample_sd":float(array.std(ddof=1)),
            "n":len(array),"seed_ids":expected,"values":array.tolist()}


def _exposure_units(rounds):
    result={}
    seen_rounds=set()
    for row in rounds:
        round_id=_integer(row.get("round_index"),"round_index")
        if round_id in seen_rounds:
            raise ValueError(f"duplicate round ledger {round_id}")
        seen_rounds.add(round_id)
        clients=row.get("clients")
        if not isinstance(clients,Mapping) or not clients:
            raise ValueError("round ledger requires nonempty actual client training ledgers")
        selected=row.get("reuse_selected_ids")
        if not isinstance(selected,Mapping):
            raise ValueError("missing actual reuse_selected_ids ledger")
        for client,training in clients.items():
            client=str(client)
            if client not in selected:
                raise ValueError(f"missing actual selection count for {client}, round {round_id}")
            ids=selected[client]
            if not isinstance(ids,list) or len(set(map(str,ids)))!=len(ids):
                raise ValueError("selected case ids must be an actual unique list")
            presentations=training.get("presentations",{})
            count=_integer(presentations.get("correction"),"correction presentations")
            steps=_integer(training.get("optimizer_steps"),"actual optimizer_steps")
            if "attempted_optimizer_steps" in training or "amp_skipped_steps" in training:
                attempted=_integer(training.get("attempted_optimizer_steps"),"attempted_optimizer_steps")
                skipped=_integer(training.get("amp_skipped_steps"),"amp_skipped_steps")
                if skipped>attempted or steps!=attempted-skipped:
                    raise ValueError("applied optimizer steps disagree with attempts minus AMP skips")
            if training.get("optimizer_steps_semantics") not in (None,"actual_applied_updates"):
                raise ValueError("optimizer steps do not represent actual applied updates")
            result[(client,round_id)]={"correction_presentations":count,"optimizer_steps":steps,"selection_count":len(ids)}
    if not result:
        raise ValueError("exposure comparison requires nonempty run ledgers")
    return result


def compare_exposure(left_rounds,right_rounds):
    """Compare counts for each client and round."""
    left,right=_exposure_units(left_rounds),_exposure_units(right_rounds)
    shared=set(left)&set(right)
    rows=[]
    for client,round_id in sorted(shared,key=lambda k:(k[1],k[0])):
        a,b=left[(client,round_id)],right[(client,round_id)]
        differences={key:b[key]-a[key] for key in a}
        rows.append({"client_id":client,"round_index":round_id,"left":a,"right":b,
                     "right_minus_left":differences,"matched":all(v==0 for v in differences.values())})
    missing_left=sorted(set(right)-set(left))
    missing_right=sorted(set(left)-set(right))
    matched=not missing_left and not missing_right and all(r["matched"] for r in rows)
    return {"matched":matched,"unit":"client_x_round","comparisons":rows,
            "missing_left":[list(k) for k in missing_left],"missing_right":[list(k) for k in missing_right],
            "claim":"exact_recorded_exposure_match" if matched else "actual_exposure_not_matched",
            "reporting_weights_used":False}


def factorial_contrasts(records,metric="value",expected_seeds=DEFAULT_SEEDS,scale=1.):
    """Compute within-seed 2x2x2 contrasts from seed, factors A/B/C, and metric.

    S8 scales signed sums by 1/4, 1/2, and 1 for main, pair, and triple effects.
    Factor bits describe the supplied rows; they do not verify executed operations."""
    expected=_seed_ids(expected_seeds)
    scale=_finite(scale,"contrast scale")
    cells={seed:{} for seed in expected}
    for row in records:
        seed=str(row["seed"])
        if seed not in cells:
            raise ValueError(f"unexpected factorial seed {seed}")
        factors=row.get("factors")
        if not isinstance(factors,Mapping) or set(factors)!={"A","B","C"}:
            raise ValueError("factorial records require explicit A/B/C factor bits")
        bits=tuple(factors[k] for k in ("A","B","C"))
        if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v not in (0,1) for v in bits):
            raise ValueError("factor bits must be integer0/1")
        if bits in cells[seed]:
            raise ValueError(f"duplicate factorial cell seed={seed}, factors={bits}")
        cells[seed][bits]=_finite(row.get(metric),metric)
    combinations={(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)}
    summaries={}
    per_seed=[]
    for seed in expected:
        if set(cells[seed])!=combinations:
            raise ValueError(f"factorial seed {seed} lacks all eight explicit configurations")
        contrasts={}
        for name,axes,weight in (("A",(0,),.25),("B",(1,),.25),("C",(2,),.25),
                ("AB",(0,1),.5),("AC",(0,2),.5),("BC",(1,2),.5),("ABC",(0,1,2),1.)):
            contrasts[name]=scale*weight*sum(np.prod([2*bits[i]-1 for i in axes])*value for bits,value in cells[seed].items())
            contrasts[name]=float(contrasts[name])
        per_seed.append({"seed":seed,**contrasts})
    for name in ("A","B","C","AB","AC","BC","ABC"):
        summaries[name]=summarize_seed_metric(per_seed,name,expected)
    return {"per_seed":per_seed,"summary":summaries,"scale":scale,
            "execution_inference":"none; supplied factor bits require separate operation definitions",
            "sign":"positive means larger supplied metric; no favorable-sign reversal inferred"}


def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
    return rows


def _recorded_hash(environment,key,*,required=False):
    value=environment.get(key)
    if value is None and not required:
        return None
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{key} must contain a recorded SHA-256 digest")
    return value.lower()


def _run_protocol(configuration,environment,path):
    """Bind source paths to recorded content identities; retain other settings."""
    protocol=json.loads(json.dumps(configuration,allow_nan=False))
    protocol.pop("_config_path",None)
    experiment=protocol.setdefault("experiment",{})
    model=protocol.setdefault("model",{})
    artifacts={}
    for section,key,recorded in ((experiment,"manifest","manifest_sha256"),
                                 (experiment,"calibration","calibration_sha256"),
                                 (model,"initial_checkpoint","initial_checkpoint_sha256")):
        artifacts[recorded]=_recorded_hash(environment,recorded,
            required=recorded=="manifest_sha256" or bool(section.get(key)))
        section.pop(key,None)
    for key in ("seed","seeds","output"):
        experiment.pop(key,None)
    calibration=path/"calibration.json"
    if calibration.is_file():
        snapshot=json.loads(calibration.read_text(encoding="utf-8-sig"))
        encoded=json.dumps(snapshot,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
        artifacts["calibration_snapshot_sha256"]=sha256(encoded).hexdigest()
    else:
        artifacts["calibration_snapshot_sha256"]=None
    workers=environment.get("workers")
    if configuration.get("communication",{}).get("mode","local")=="http":
        expected=set(configuration.get("communication",{}).get("workers",{}))
        if not isinstance(workers,dict) or not expected or set(workers)!=expected:
            raise ValueError("HTTP summaries require recorded worker identities and data bindings")
        for worker in workers.values():
            _recorded_hash(worker,"partition_sha256",required=True)
            _recorded_hash(worker,"logical_partition_sha256",required=True)
    return {"configuration":protocol,"artifacts":artifacts,"versions":environment.get("versions"),"workers":workers}


def analyze_run_dirs(run_dirs,expected_seeds=DEFAULT_SEEDS,metrics=("mAP50","AP75","mAP50_95")):
    """Read run metrics and execution ledgers."""
    paths=[Path(p).resolve() for p in run_dirs]
    if len(set(paths))!=len(paths):
        raise ValueError("duplicate run directories")
    groups={};runs=[];protocols={};seen_seeds=set()
    for path in paths:
        required=[path/name for name in ("environment.json","configuration.json","metrics.json","rounds.jsonl")]
        for file in required:
            if not file.is_file():
                raise FileNotFoundError(f"incomplete executed run: {file}")
        environment=json.loads(required[0].read_text(encoding="utf-8-sig"))
        configuration=json.loads(required[1].read_text(encoding="utf-8-sig"))
        scores=json.loads(required[2].read_text(encoding="utf-8-sig"))
        rounds=read_jsonl(required[3]);_exposure_units(rounds)
        seed=str(environment["seed"])
        if "seed" in configuration.get("experiment",{}) and str(configuration["experiment"]["seed"])!=seed:
            raise ValueError("configuration seed differs from recorded environment")
        method=str(configuration["experiment"]["method"])
        if (method,seed) in seen_seeds:
            raise ValueError(f"duplicate seed {seed} for method {method}")
        seen_seeds.add((method,seed))
        protocol=_run_protocol(configuration,environment,path)
        encoded=json.dumps(protocol,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
        protocol_hash=sha256(encoded).hexdigest()
        if method in protocols and protocols[method][0]!=protocol_hash:
            raise ValueError(f"method {method} requires one fixed protocol across its seeds; {path} differs from {protocols[method][1]}")
        protocols[method]=(protocol_hash,path)
        configured_rounds=int(configuration["experiment"].get("rounds",len(rounds)))
        if sorted(r["round_index"] for r in rounds)!=list(range(configured_rounds)):
            raise ValueError(f"incomplete round sequence in {path}")
        if not isinstance(scores,Mapping) or not scores:
            raise ValueError("run has no final evaluation views")
        for view,values in scores.items():
            for metric in metrics:
                value=_finite(values.get(metric),f"{method}/{seed}/{view}/{metric}")
                groups.setdefault((method,str(view),metric),[]).append({"seed":seed,"value":value})
        runs.append({"path":str(path),"method":method,"seed":seed,"rounds":len(rounds),
            "manifest_sha256":protocol["artifacts"]["manifest_sha256"],
            "protocol_sha256":protocol_hash,
            "files_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in required}})
    if not groups:
        raise ValueError("no completed runs supplied")
    summary=[]
    for (method,view,metric),rows in sorted(groups.items()):
        summary.append({"method":method,"view":view,"metric":metric,**summarize_seed_metric(rows,expected_seeds=expected_seeds)})
    return {"summary":summary,"runs":runs,"uncertainty":"sample SD across declared actual seeds",
            "historical_results_embedded":False,"energy":"not inferred from round duration"}


def read_power_csv(paths,*,require_manifests=True):
    rows=[];manifests=[]
    for raw in paths:
        path=Path(raw)
        sidecar=path.with_suffix(path.suffix+".manifest.json")
        if require_manifests:
            if not sidecar.is_file():
                raise FileNotFoundError(f"power scope/unit manifest required: {sidecar}")
            manifest=json.loads(sidecar.read_text(encoding="utf-8"))
            if manifest.get("power_unit")!="W" or manifest.get("time_unit")!="unix_utc_seconds":
                raise ValueError("power manifest must declare W and unix UTC seconds")
            if manifest.get("scope") not in ("server_gpu","jetson_board"):
                raise ValueError("unsupported energy scope (CPU/wall power is not server GPU power)")
            if manifest.get("status")!="completed":
                raise ValueError("power recording did not complete; inspect its manifest")
            manifests.append(manifest)
        with path.open(newline="",encoding="utf-8-sig") as stream:
            reader=csv.DictReader(stream)
            if not {"ts","power_w","node"}<=set(reader.fieldnames or ()):
                raise ValueError("power CSV requires ts,power_w,node")
            for row in reader:
                if require_manifests:
                    if row["node"]!=manifest.get("node"):
                        raise ValueError("CSV node differs from its measurement scope manifest")
                    if row["node"]=="serverGPU" and manifest["scope"]!="server_gpu":
                        raise ValueError("serverGPU must use a server GPU power measurement")
                    if row["node"] in {"J1","J2","J3","J4"} and manifest["scope"]!="jetson_board":
                        raise ValueError("Jetson nodes require board-power scope")
                rows.append({"ts":float(row["ts"]),"power_w":float(row["power_w"]),"node":row["node"],
                             "memory_mib":float(row["memory_mib"]) if row.get("memory_mib") else None})
    return {"rows":rows,"manifests":manifests}


def integrate_round_energy(power_rows,rounds,*,required_nodes=ENERGY_NODES,expected_interval_s=None,gap_tolerance=.5):
    """Integrate measured watts over UTC-second round boundaries by trapezoids.

    Each node must bracket the round; extrapolation and gaps above
    (1 + gap_tolerance) times the declared cadence are rejected."""
    if expected_interval_s is None:
        raise ValueError("declare the actual recorder sampling interval; it cannot be guessed")
    tolerance=_finite(gap_tolerance,"gap_tolerance")
    if not 0<=tolerance<1:
        raise ValueError("gap tolerance must be in [0,1) to reject a missing cadence point")
    nodes=list(required_nodes)
    if not nodes or len(set(nodes))!=len(nodes):
        raise ValueError("required energy nodes must be nonempty and unique")
    series={node:[] for node in nodes}
    for row in power_rows:
        node=str(row["node"])
        if node not in series:
            raise ValueError(f"unexpected energy node {node}; scope must be explicit")
        ts=_finite(row.get("ts"),"power timestamp")
        power=_finite(row.get("power_w"),"power_w")
        if power<0:
            raise ValueError("negative measured device power")
        if series[node] and ts<=series[node][-1][0]:
            raise ValueError(f"non-monotonic/duplicate power timestamp for {node}")
        series[node].append((ts,power))
    for node in nodes:
        if len(series[node])<2:
            raise ValueError(f"missing complete power samples for {node}")
    output=[];seen=set()
    for row in rounds:
        rid=_integer(row.get("round_index"),"round_index")
        if rid in seen:
            raise ValueError("duplicate energy round")
        seen.add(rid)
        start=row.get("start_ts",row.get("utc_start_s"))
        end=row.get("end_ts",row.get("utc_end_s"))
        start=_finite(start,"actual round start_ts")
        end=_finite(end,"actual round end_ts")
        if end<=start:
            raise ValueError("round end must follow start")
        energies={}
        for node in nodes:
            interval=expected_interval_s.get(node) if isinstance(expected_interval_s,Mapping) else expected_interval_s
            interval=_finite(interval,f"{node} sample interval")
            if interval<=0:
                raise ValueError("sample interval must be positive")
            samples=np.asarray(series[node])
            times,powers=samples[:,0],samples[:,1]
            if times[0]>start or times[-1]<end:
                raise ValueError(f"{node} samples do not cover both round {rid} boundaries")
            for i,gap in enumerate(np.diff(times)):
                if times[i]<end and times[i+1]>start and gap>interval*(1+tolerance)+1e-9:
                    raise ValueError(f"{node} missing sampling point in round {rid}: gap {gap}s")
            inside=(times>start)&(times<end)
            x=np.r_[start,times[inside],end]
            y=np.r_[np.interp(start,times,powers),powers[inside],np.interp(end,times,powers)]
            energies[node]=float(np.sum((y[:-1]+y[1:])*.5*np.diff(x)))
        output.append({"round_index":rid,"start_ts":start,"end_ts":end,"elapsed_s":end-start,
                       "node_energy_j":energies,"total_energy_j":sum(energies.values())})
    if not output:
        raise ValueError("energy integration requires explicit actual round boundaries")
    chronological=sorted(output,key=lambda r:r["start_ts"])
    if any(a["end_ts"]>b["start_ts"] for a,b in zip(chronological,chronological[1:])):
        raise ValueError("overlapping round windows would double-count energy")
    if [r["round_index"] for r in chronological]!=sorted(r["round_index"] for r in output):
        raise ValueError("round indices disagree with chronological execution order")
    return {"rounds":output,"total_energy_j":sum(r["total_energy_j"] for r in output),
            "power_unit":"W","time_unit":"unix_utc_seconds","energy_unit":"J",
            "nodes":nodes,"integration":"boundary-interpolated trapezoids; no extrapolation",
            "scope":"four Jetson boards plus server GPU; excludes server CPU and wall losses" if tuple(nodes)==ENERGY_NODES else "explicit supplied device nodes"}


def latency_p95_worst_client(rows,*,expected_nodes=("J1","J2","J3","J4"),samples_per_node=200):
    """Return the worst node P95 across four nodes with 200 requests each."""
    samples_per_node=_integer(samples_per_node,"samples_per_node")
    if samples_per_node<200:
        raise ValueError("the declared protocol requires at least200 recorded requests/node")
    values={node:[] for node in expected_nodes}
    if len(values)!=len(expected_nodes) or not values:
        raise ValueError("expected nodes must be nonempty and unique")
    for row in rows:
        node=str(row["node"])
        if node not in values:
            raise ValueError(f"unexpected latency node {node}")
        value=_finite(row.get("latency_ms"),"request latency_ms")
        if value<0:
            raise ValueError("latencies cannot be negative")
        values[node].append(value)
    if any(len(v)!=samples_per_node for v in values.values()):
        raise ValueError(f"each expected node needs exactly{samples_per_node} actual request latencies")
    p95={node:float(np.percentile(value,95,method="linear")) for node,value in values.items()}
    worst=max(p95,key=lambda node:(p95[node],node))
    return {"node_p95_ms":p95,"worst_node":worst,"worst_node_p95_ms":p95[worst],
            "samples_per_node":samples_per_node,"unit":"ms","quantile_method":"linear",
            "reduction":"node-specific P95, then maximum across nodes"}


def _ratio(numerator,denominator,unit):
    return {"numerator":numerator,"denominator":denominator,
            "proportion":numerator/denominator if denominator else None,"unit":unit}


def _rta_reference(record):
    obs=record.get("rta_observation") or record
    renderer_class=obs.get("renderer_class")
    renderer_box=obs.get("renderer_box")
    if renderer_class is None or renderer_box is None:
        targets=record.get("targets")
        if not targets:
            raise ValueError("RTA requires an actual renderer focal annotation; source class is not a substitute")
        focal=_integer(record.get("focal_target_index",0),"focal_target_index")
        if focal>=len(targets):
            raise ValueError("renderer focal target index out of bounds")
        renderer_class=targets[focal]["class_id"]
        renderer_box=targets[focal]["box"]
    renderer_class=_integer(renderer_class,"renderer_class")
    def box(value,name):
        if value is None or len(value)!=4:
            raise ValueError(f"{name} must contain normalized xyxy coordinates")
        result=[_finite(x,name) for x in value]
        if any(not 0<=x<=1 for x in result) or result[2]<=result[0] or result[3]<=result[1]:
            raise ValueError(f"{name} must be a positive normalized box")
        return result
    renderer_box=box(renderer_box,"renderer_box")
    pred_class=obs.get("predicted_class")
    pred_box=obs.get("predicted_box")
    if pred_class is None or pred_box is None:
        hit=False;overlap=None;pred_class=None;pred_box=None
    else:
        pred_class=_integer(pred_class,"predicted_class")
        pred_box=box(pred_box,"predicted_box")
        intersect=max(0.,min(pred_box[2],renderer_box[2])-max(pred_box[0],renderer_box[0]))*max(0.,min(pred_box[3],renderer_box[3])-max(pred_box[1],renderer_box[1]))
        pa=(pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1])
        ra=(renderer_box[2]-renderer_box[0])*(renderer_box[3]-renderer_box[1])
        overlap=intersect/(pa+ra-intersect)
        hit=pred_class==renderer_class and overlap>=.50
    if obs.get("agreement") is not None and bool(obs["agreement"])!=hit:
        raise ValueError("recorded RTA agreement conflicts with class/box matching")
    if overlap is not None and obs.get("iou") is not None and abs(_finite(obs["iou"],"recorded IoU")-overlap)>1e-6:
        raise ValueError("recorded RTA IoU conflicts with its boxes")
    signature=json.dumps([pred_class,pred_box,renderer_class,renderer_box],sort_keys=True,allow_nan=False)
    return {"hit":hit,"iou":overlap,"signature":signature}


def case_event_proportions(retained_cases=(),selected_events=(),verification_records=(),*,
                          candidate_records=None,qualified_count=None,assessed_count=None):
    """Compute micro proportions over distinct cases and process events.

    RTA requires the renderer focal class and IoU >= .5; missing responses fail.
    Events are unique (client_id, round_index, case_id). Identical duplicates
    collapse; conflicting duplicates raise an error."""
    retained={};events={};entered={}
    for record in retained_cases:
        uid=record.get("id",record.get("case_id"))
        if uid is None:
            raise ValueError("retained record lacks a case id")
        uid=str(uid)
        value=_rta_reference(record)
        if uid in retained and retained[uid]["signature"]!=value["signature"]:
            raise ValueError("conflicting RTA observations for the same retained case")
        retained[uid]=value
    for record in selected_events:
        if record.get("client_id") is None or record.get("case_id") is None:
            raise ValueError("selected event needs client_id and case_id")
        key=(str(record["client_id"]),_integer(record.get("round_index",record.get("round")),"selected round"),str(record["case_id"]))
        value=_rta_reference(record)
        if key in events and events[key]["signature"]!=value["signature"]:
            raise ValueError("conflicting predictions within one current-bank selection event")
        events[key]=value
    for record in verification_records:
        if record.get("entered_verification") is not True:
            if record.get("retained") is True:
                raise ValueError("retained verification record lacks an entered-verification marker")
            continue
        uid=record.get("case_id",record.get("id"))
        if uid is None or not isinstance(record.get("retained"),bool):
            raise ValueError("verification entry needs case id and retained boolean")
        uid=str(uid);value=record["retained"]
        if uid in entered and entered[uid]!=value:
            raise ValueError("conflicting verification dispositions for one case")
        entered[uid]=value
    if candidate_records is not None:
        if qualified_count is not None or assessed_count is not None:
            raise ValueError("supply candidate records or explicit qualification counts, not both")
        candidates={}
        for record in candidate_records:
            required=(record.get("client_id"),record.get("round_index"),record.get("local_identity"),record.get("id",record.get("candidate_id")))
            if any(value is None for value in required):
                raise ValueError("candidate audit needs client_id,round_index,local_identity,id")
            key=(str(required[0]),_integer(required[1],"candidate round"),str(required[2]),str(required[3]))
            disposition=record.get("disposition")
            if disposition not in ("qualified","correction","suppressed"):
                raise ValueError("unknown candidate disposition")
            if key in candidates and candidates[key]!=disposition:
                raise ValueError("conflicting candidate audit dispositions")
            candidates[key]=disposition
        qualified_count=sum(x=="qualified" for x in candidates.values())
        assessed_count=len(candidates)
    if qualified_count is None and assessed_count is None:
        qualification={"numerator":None,"denominator":None,"proportion":None,"unit":"assessed_candidates","status":"not_supplied"}
    else:
        qualified_count=_integer(qualified_count,"qualified_count")
        assessed_count=_integer(assessed_count,"assessed_count")
        if qualified_count>assessed_count:
            raise ValueError("qualified_count exceeds assessed_count")
        qualification=_ratio(qualified_count,assessed_count,"assessed_candidates")
    return {"retained_rta":_ratio(sum(r["hit"] for r in retained.values()),len(retained),"unique_retained_cases"),
            "selected_rta":_ratio(sum(r["hit"] for r in events.values()),len(events),"unique_client_round_case_selection_events"),
            "qualification":qualification,"retention":_ratio(sum(entered.values()),len(entered),"variants_entering_verification"),
            "reference":"renderer-generated focal annotation, not human ground truth",
            "rta_hit_rule":"predicted class equals renderer class and IoU>=0.50; missing prediction is a miss",
            "local_epoch_duplicates_add_events":False}
