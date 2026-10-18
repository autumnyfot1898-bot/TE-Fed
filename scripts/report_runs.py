"""Audit completed runs and draw figures from explicit measured inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tefed.reporting import export_audit, plot_runs, qualitative_comparison


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    audit=commands.add_parser('audit')
    audit.add_argument('--run',required=True,type=Path)
    audit.add_argument('--out',required=True,type=Path)
    plots=commands.add_parser('plots')
    plots.add_argument('--runs',nargs='+',required=True,type=Path)
    plots.add_argument('--factorial-records',type=Path)
    plots.add_argument('--seeds',nargs='+',type=int)
    plots.add_argument('--out',required=True,type=Path)
    qualitative=commands.add_parser('qualitative')
    qualitative.add_argument('--manifest',required=True,type=Path)
    qualitative.add_argument('--view',required=True,choices=('participating','unseen','new-client','hard'))
    qualitative.add_argument('--ids',nargs='+',required=True)
    qualitative.add_argument('--prediction',action='append',required=True,metavar='LABEL=JSONL')
    qualitative.add_argument('--image-size',required=True,type=int)
    qualitative.add_argument('--threshold',required=True,type=float)
    qualitative.add_argument('--view-spec',type=Path)
    qualitative.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=='audit':
        result=export_audit(args.run,args.out)
    elif args.command=='plots':
        result=plot_runs(args.runs,args.out,factorial_records=args.factorial_records,seeds=args.seeds)
    else:
        predictions={}
        for value in args.prediction:
            label,separator,path=value.partition('=')
            if not separator or not label.strip() or not path or label in predictions:
                raise ValueError('Each --prediction must be a distinct LABEL=JSONL')
            predictions[label]=path
        result=qualitative_comparison(args.manifest,args.view,args.ids,predictions,args.out,
            image_size=args.image_size,threshold=args.threshold,view_spec=args.view_spec)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 1 if args.command=='audit' and result['status']!='passed' else 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'Report failed: {error}',file=sys.stderr)
        raise SystemExit(2)
