"""Dispatch method-first commands to tefed.cli.

Example: python -m tefed.main tefed --config configs/tomato.toml --output runs/tefed"""
from __future__ import annotations

import sys

from . import cli
from .baselines import METHOD_REGISTRY


def translate_argv(argv):
    """Map a registered method to the existing train command without overrides."""
    argv = list(argv)
    if argv and argv[0] in METHOD_REGISTRY:
        if any(value == '--method' or value.startswith('--method=') for value in argv[1:]):
            raise ValueError('The method is already selected by the first argument; omit --method.')
        return ['train', '--method', argv[0], *argv[1:]]
    return argv


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if list(argv) in (['--help'], ['-h']):
        print('Method-first usage: python -m tefed.main METHOD --config FILE --output DIR [--seed N]')
        print('Methods: ' + ', '.join(METHOD_REGISTRY))
        print('Regular commands are also available below.\n')
    try:
        translated = translate_argv(argv)
    except ValueError as error:
        cli.parser().error(str(error))
    return cli.main(translated)


if __name__ == '__main__':
    sys.exit(main())
