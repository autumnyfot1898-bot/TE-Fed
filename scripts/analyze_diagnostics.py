"""Compute review calibration, externally labelled routing, and paired run differences."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tefed.offline import (FULL_BIN_EDGES, PAPER_BIN_EDGES, input_records,
                           paired_run_differences, reliability_diagnostics,
                           routing_diagnostics, write_new_json)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest='command', required=True)
    reliability = sub.add_parser('reliability', help='Independent human-reviewed binary outcomes (S11.3)')
    reliability.add_argument('--records', required=True)
    bins = reliability.add_mutually_exclusive_group()
    bins.add_argument('--bins', choices=('paper', 'full'), default='paper', help='paper: six 0.1-wide bins [.4,1]; full: ten bins [0,1]')
    bins.add_argument('--bin-edges', type=float, nargs='+', help='Explicit increasing edges; every event must be covered')
    routing = sub.add_parser('routing', help='First external condition label and specialist-only top-1 (S11.4)')
    routing.add_argument('--candidates', required=True)
    routing.add_argument('--conditions', required=True)
    routing.add_argument('--join-on', choices=('candidate', 'identity', 'client_round', 'event_id'), default='candidate')
    paired = sub.add_parser('paired', help='Same-seed right-minus-left scores from complete executed runs')
    paired.add_argument('--left', nargs='+', required=True)
    paired.add_argument('--right', nargs='+', required=True)
    paired.add_argument('--seeds', nargs='+', default=list(map(str, range(10))))
    paired.add_argument('--metrics', nargs='+', default=['mAP50', 'AP75', 'mAP50_95'])
    for item in (reliability, routing, paired):
        item.add_argument('--out', required=True, help='New JSON result path; existing files are preserved')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'reliability':
        records, manifest = input_records(args.records)
        edges = args.bin_edges if args.bin_edges is not None else FULL_BIN_EDGES if args.bins == 'full' else PAPER_BIN_EDGES
        policy = 'explicit_edges' if args.bin_edges is not None else 'full_unit_interval' if args.bins == 'full' else 'paper_s11_3'
        result = reliability_diagnostics(records, bin_edges=edges, bin_policy=policy)
        result['inputs'] = {'review_records': manifest}
    elif args.command == 'routing':
        candidates, candidate_manifest = input_records(args.candidates)
        conditions, condition_manifest = input_records(args.conditions)
        result = routing_diagnostics(candidates, conditions, join_on=args.join_on)
        result['inputs'] = {'candidates': candidate_manifest, 'conditions': condition_manifest}
    else:
        result = paired_run_differences(args.left, args.right, expected_seeds=args.seeds, metrics=args.metrics)
    print(write_new_json(args.out, result))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'Analysis failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
