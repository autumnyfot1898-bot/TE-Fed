"""Record NVIDIA GPU or Jetson board power with explicit source units."""
from __future__ import annotations
import argparse
import csv
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
import time


def read_nvidia(gpu="0",executable="nvidia-smi"):
    result=subprocess.run([executable,f"--id={gpu}","--query-gpu=power.draw,memory.used",
        "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=True)
    lines=[line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines)!=1:
        raise ValueError("nvidia-smi must return exactly one selected GPU")
    fields=[value.strip() for value in lines[0].split(",")]
    if len(fields)!=2:
        raise ValueError("GPU telemetry needs measured power and memory fields")
    try:
        power,memory=map(float,fields)
    except ValueError as exc:
        raise ValueError("GPU power/memory telemetry is unavailable, not zero") from exc
    if not isfinite(power) or not isfinite(memory) or min(power,memory)<0:
        raise ValueError("invalid GPU telemetry")
    return {"power_w":power,"memory_mib":memory}


def read_sysfs(path,unit):
    factors={"W":1.,"mW":.001,"uW":.000001}
    if unit not in factors:
        raise ValueError("sysfs source unit must explicitly be W, mW or uW")
    source=Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"explicit Jetson board-power path not found: {source}")
    try:
        value=float(source.read_text(encoding="ascii").strip())
    except (ValueError,UnicodeError) as exc:
        raise ValueError("power sysfs path does not contain a numeric measurement") from exc
    if not isfinite(value) or value<0:
        raise ValueError("invalid measured board power")
    return {"power_w":value*factors[unit],"memory_mib":None}


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("nvidia-smi","sysfs"),required=True)
    parser.add_argument("--node",required=True,help="serverGPU or a Jetson node id such as J1")
    parser.add_argument("--out",required=True,help="New CSV filename; existing files are not overwritten")
    parser.add_argument("--duration",type=float,required=True,help="Measured recording duration in seconds")
    parser.add_argument("--interval",type=float,default=1.,help="Requested polling cadence in seconds")
    parser.add_argument("--gpu",default="0")
    parser.add_argument("--nvidia-smi",default="nvidia-smi")
    parser.add_argument("--power-path")
    parser.add_argument("--unit",choices=("W","mW","uW"))
    parser.add_argument("--scope",choices=("jetson_board",),help="Explicitly confirm the chosen sysfs rail is total board power")
    return parser


def record(args):
    if not isfinite(args.duration) or args.duration<=0 or not isfinite(args.interval) or args.interval<.05:
        raise ValueError("duration must be positive and interval at least .05 seconds")
    if not args.node.strip():
        raise ValueError("node id is required")
    if args.backend=="nvidia-smi":
        if args.node!="serverGPU":
            raise ValueError("the NVIDIA server backend uses node=serverGPU; Jetson boards use explicit sysfs")
        reader=lambda:read_nvidia(args.gpu,args.nvidia_smi)
        scope="server_gpu"
        source={"gpu_id":args.gpu,"executable":args.nvidia_smi}
    else:
        if not args.power_path or not args.unit or args.scope!="jetson_board":
            raise ValueError("sysfs recording requires --power-path, --unit and --scope jetson_board")
        if args.node not in {"J1","J2","J3","J4"}:
            raise ValueError("Jetson node must explicitly be J1, J2, J3 or J4")
        reader=lambda:read_sysfs(args.power_path,args.unit)
        scope="jetson_board"
        source={"path":str(Path(args.power_path).resolve()),"source_unit":args.unit}
    output=Path(args.out).resolve()
    sidecar=output.with_suffix(output.suffix+".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("choose fresh telemetry output paths")
    # Check telemetry before creating output files.
    first=reader()
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest={"node":args.node,"backend":args.backend,"scope":scope,"source":source,
        "power_unit":"W","memory_unit":"MiB" if scope=="server_gpu" else None,
        "time_unit":"unix_utc_seconds","interval_s":args.interval,
        "requested_duration_s":args.duration,"started_ts":time.time(),"samples":0,
        "status":"recording","timestamp":"host UTC at telemetry-read completion",
        "excludes":["server_CPU_power","wall_power_losses"],
        "scope_note":"Jetson board includes its on-board processors; server value is GPU board only"}
    sidecar.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    start=time.monotonic();next_read=start;previous_ts=None
    try:
        with output.open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["ts","power_w","memory_mib","node"])
            writer.writeheader()
            while True:
                value=first if manifest["samples"]==0 else reader()
                ts=time.time()
                if previous_ts is not None and ts<=previous_ts:
                    raise RuntimeError("host UTC clock moved backwards; record cannot be integrated safely")
                writer.writerow({"ts":f"{ts:.9f}","node":args.node,**value})
                stream.flush()
                previous_ts=ts;manifest["samples"]+=1
                if time.monotonic()-start>=args.duration:
                    break
                next_read+=args.interval
                wait=next_read-time.monotonic()
                if wait>0:
                    time.sleep(wait)
        manifest["status"]="completed"
    except KeyboardInterrupt:
        manifest["status"]="interrupted"
    except Exception as exc:
        manifest.update(status="failed",error=str(exc))
        raise
    finally:
        manifest["ended_ts"]=time.time()
        sidecar.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    return manifest


if __name__=="__main__":
    try:
        result=record(make_parser().parse_args())
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result["status"]=="completed" else 2)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
        print(f"Power recording failed: {exc}",file=sys.stderr)
        raise SystemExit(2)
