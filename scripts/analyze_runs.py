"""Analyze run ledgers, scores and hardware telemetry."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tefed.analysis import (analyze_run_dirs,compare_exposure,factorial_contrasts,
    integrate_round_energy,latency_p95_worst_client,read_jsonl,read_power_csv,case_event_proportions)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    summary=sub.add_parser("summary",help="Exact declared seeds from completed engine runs")
    summary.add_argument("--runs",nargs="+",required=True)
    summary.add_argument("--seeds",nargs="+",default=list(map(str,range(10))))
    exposure=sub.add_parser("exposure",help="Actual client-round correction and update counts")
    exposure.add_argument("--left",required=True,help="rounds.jsonl or its run directory")
    exposure.add_argument("--right",required=True)
    exposure.add_argument("--require-match",action="store_true")
    factorial=sub.add_parser("factorial",help="Explicit eight-cell within-seed contrasts")
    factorial.add_argument("--records",required=True)
    factorial.add_argument("--definitions",required=True,help="JSON defining A/B/C on/off executed operations")
    factorial.add_argument("--metric",default="value")
    factorial.add_argument("--scale",type=float,default=1.)
    factorial.add_argument("--seeds",nargs="+",default=list(map(str,range(10))))
    energy=sub.add_parser("energy",help="Measured five-device energy over actual round boundaries")
    energy.add_argument("--power",nargs="+",required=True)
    energy.add_argument("--rounds",required=True)
    energy.add_argument("--interval",type=float,required=True,help="Actual declared recording cadence in seconds")
    energy.add_argument("--gap-tolerance",type=float,default=.5)
    latency=sub.add_parser("latency",help="Node P95 from200 requests each, then worst of four nodes")
    latency.add_argument("--csv",required=True)
    latency.add_argument("--samples-per-node",type=int,default=200)
    process=sub.add_parser("process",help="Distinct retained/selected RTA and separate process ratios")
    process.add_argument("--run",required=True)
    for command in (summary,exposure,factorial,energy,latency,process):
        command.add_argument("--out",required=True)
    return p


def _round_path(path):
    p=Path(path)
    return p/"rounds.jsonl" if p.is_dir() else p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=="summary":
        result=analyze_run_dirs(args.runs,expected_seeds=args.seeds)
    elif args.command=="exposure":
        result=compare_exposure(read_jsonl(_round_path(args.left)),read_jsonl(_round_path(args.right)))
    elif args.command=="factorial":
        definitions=json.loads(Path(args.definitions).read_text(encoding="utf-8-sig"))
        if set(definitions)!={"A","B","C"} or any(not isinstance(v,dict) or not all(isinstance(v.get(k),str) and v[k].strip() for k in ("on","off")) for v in definitions.values()):
            raise ValueError("definitions must explicitly describe A/B/C on and off executed operations")
        result=factorial_contrasts(read_jsonl(args.records),args.metric,args.seeds,args.scale)
        result["operation_definitions"]=definitions
    elif args.command=="energy":
        measured=read_power_csv(args.power)
        result=integrate_round_energy(measured["rows"],read_jsonl(_round_path(args.rounds)),
            expected_interval_s=args.interval,gap_tolerance=args.gap_tolerance)
        result["recording_manifests"]=measured["manifests"]
    elif args.command=="latency":
        with Path(args.csv).open(newline="",encoding="utf-8-sig") as stream:
            reader=csv.DictReader(stream)
            if not {"node","latency_ms"}<=set(reader.fieldnames or ()):
                raise ValueError("latency CSV requires node,latency_ms in explicit milliseconds")
            rows=[{"node":r["node"],"latency_ms":float(r["latency_ms"])} for r in reader]
        result=latency_p95_worst_client(rows,samples_per_node=args.samples_per_node)
    else:
        run=Path(args.run)
        verification=[]
        for path in sorted(run.glob("verification_round_*.json")):
            verification.extend(json.loads(path.read_text(encoding="utf-8-sig")))
        rounds=read_jsonl(run/"rounds.jsonl")
        if len(list(run.glob("verification_round_*.json")))!=len(rounds):
            raise ValueError("process analysis requires each round's actual verification audit")
        selected_path=run/"selected_events.jsonl"
        selected=read_jsonl(selected_path) if selected_path.is_file() else []
        expected=sum(len(ids) for row in rounds for ids in row.get("reuse_selected_ids",{}).values())
        keys={(str(r["client_id"]),r["round_index"],str(r["case_id"])) for r in selected}
        if len(keys)!=expected:
            raise ValueError("selected-event observations do not cover actual selection ledger")
        candidates=[];files=list(run.glob("client_*_candidate_audit.jsonl"))
        for path in files:
            client=path.name[len("client_"):-len("_candidate_audit.jsonl")]
            candidates.extend({"client_id":client,**row} for row in read_jsonl(path))
        retained=[r for r in verification if r.get("entered_verification") is True and r.get("retained") is True]
        result=case_event_proportions(retained,selected,verification,candidate_records=candidates if files else None)
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(str(out.resolve()))
    return 1 if args.command=="exposure" and args.require_match and not result["matched"] else 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f"Analysis failed: {exc}",file=sys.stderr)
        raise SystemExit(2)
