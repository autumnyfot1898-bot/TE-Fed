"""Freeze private correction cases and bind strict exposure counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from tefed.replay import freeze_pool,build_exposure_protocol


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze')
    freeze.add_argument('--run',required=True,type=Path)
    freeze.add_argument('--output',required=True,type=Path)
    protocol=commands.add_parser('protocol')
    protocol.add_argument('--pool',required=True,type=Path)
    protocol.add_argument('--runs',nargs='+',required=True,type=Path)
    protocol.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=='freeze':
        status=json.loads((args.run/'status.json').read_text(encoding='utf-8-sig'))
        if status.get('status')!='completed':
            raise ValueError('Freeze a completed run')
        cases=json.loads((args.run/'final.pool.json').read_text(encoding='utf-8-sig'))
        states=json.loads((args.run/'source_states.json').read_text(encoding='utf-8-sig'))
        result=freeze_pool(cases,states,args.output,provenance={'source_run':str(args.run.resolve())})
    else:
        result=build_exposure_protocol(args.pool,args.runs,args.output)
    print(json.dumps({'output':str(result)}))


if __name__=='__main__':
    main()
