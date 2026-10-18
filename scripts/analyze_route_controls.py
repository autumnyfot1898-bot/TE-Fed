"""Compare stored/current/shift-11 routes on frozen variant records."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tefed.offline import write_new_json
from tefed.verification_controls import analyze_frozen_run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--allow-incomplete",action="store_true",
                        help="export incomplete events with undefined summary fractions")
    args=parser.parse_args(argv)
    result=analyze_frozen_run(args.run,allow_incomplete=args.allow_incomplete)
    write_new_json(args.out,result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
