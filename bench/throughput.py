"""Offline throughput: every request submitted at t=0; measures peak capacity."""

import argparse
from pathlib import Path

from bench import datasets, engine, metrics
from bench.driver import drive

THROUGHPUT_KEYS = (
    "completed",
    "duration_s",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)


def add_args(parser: argparse.ArgumentParser) -> None:
    datasets.add_args(parser)


def run(args: argparse.Namespace) -> None:
    requests = datasets.from_args(args)
    eng = engine.from_args(args)
    drive(eng, requests[:1])  # warmup
    records, duration = drive(eng, requests)

    full = metrics.summarize(records, duration, percentiles=[])
    summary = {k: full[k] for k in THROUGHPUT_KEYS}
    metrics.print_summary("Throughput Benchmark", summary)
    if args.output_json:
        metrics.save_json(Path(args.output_json), args, summary)
