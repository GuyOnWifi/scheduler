"""Benchmark the inference engine: python bench.py {latency,throughput,serve,all} --help"""

import argparse
from pathlib import Path

from bench import engine, latency, serve, throughput

MODES = {"latency": latency, "throughput": throughput, "serve": serve}
SHOW_DEFAULTS = argparse.ArgumentDefaultsHelpFormatter


def add_common_args(parser: argparse.ArgumentParser) -> None:
    engine.add_args(parser)
    parser.add_argument("--seed", type=int, default=0, help="for prompts and arrivals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name, mode in MODES.items():
        sub = subparsers.add_parser(
            name, help=mode.__doc__, description=mode.__doc__, formatter_class=SHOW_DEFAULTS
        )
        add_common_args(sub)
        sub.add_argument(
            "--output-json", metavar="PATH", help="write args + summary to this file"
        )
        mode.add_args(sub)

    run_all = subparsers.add_parser(
        "all",
        help="run every mode in turn, each with its own defaults",
        formatter_class=SHOW_DEFAULTS,
    )
    add_common_args(run_all)
    run_all.add_argument(
        "--output-dir", metavar="DIR", help="write <mode>.json for each mode here"
    )

    args = parser.parse_args()
    if args.mode != "all":
        MODES[args.mode].run(args)
        return

    for name, mode in MODES.items():
        # Re-parse through the mode's own subparser so it gets that mode's defaults.
        argv = [name, "--engine", args.engine, "--seed", str(args.seed)]
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            argv += ["--output-json", str(Path(args.output_dir) / f"{name}.json")]
        mode.run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
