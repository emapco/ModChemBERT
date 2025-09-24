import argparse

from da4mt.adapt.__main__ import run_domain_adaptation
from da4mt.cli import (
    add_adapt_args,
    add_prepare_parser,
)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="Command")
prepare_parser = subparsers.add_parser("prepare", help="Run data preparation.")
prepare_parser = add_prepare_parser(prepare_parser)

adaptation_parser = subparsers.add_parser("adapt", help="Run domain adaptation.")
adaptation_parser = add_adapt_args(adaptation_parser)
adaptation_parser.set_defaults(func=run_domain_adaptation)

args = parser.parse_args()
args.func(args)
