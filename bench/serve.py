"""Online serving: requests arrive over time; measures latency under load."""

import argparse
import random
from dataclasses import replace
from pathlib import Path

from bench import datasets, engine, metrics
from bench.driver import arrival_times, drive


def parse_slo(text: str) -> tuple[str, float]:
    name, _, limit = text.partition(":")
    if name not in ("ttft", "tpot", "e2el"):
        raise argparse.ArgumentTypeError(f"unknown SLO metric {name!r}")
    return name, float(limit)


def add_args(parser: argparse.ArgumentParser) -> None:
    datasets.add_args(parser)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        metavar="REQ/S",
        help="mean arrival rate; inf sends everything at t=0",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        metavar="SHAPE",
        help="gamma shape of arrival gaps: 1 = Poisson, <1 burstier, >1 more even",
    )
    parser.add_argument(
        "--percentiles",
        type=lambda s: [float(p) for p in s.split(",")],
        default=[99.0],
        metavar="P,P,...",
        help="latency percentiles to report, e.g. 50,90,99",
    )
    parser.add_argument(
        "--goodput",
        type=parse_slo,
        nargs="+",
        metavar="METRIC:MS",
        help="SLOs a request must meet to count, e.g. ttft:500 tpot:50",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=1,
        metavar="N",
        help="requests run untimed first",
    )


def run(args: argparse.Namespace) -> None:
    requests = datasets.from_args(args)
    times = arrival_times(
        len(requests), args.request_rate, args.burstiness, random.Random(args.seed)
    )
    requests = [replace(r, arrival_time=t) for r, t in zip(requests, times)]

    eng = engine.from_args(args)
    drive(eng, requests[: args.num_warmups])
    records, duration = drive(eng, requests)

    summary = metrics.summarize(
        records, duration, args.percentiles, dict(args.goodput or [])
    )
    metrics.print_summary("Serving Benchmark", summary)
    if args.output_json:
        metrics.save_json(Path(args.output_json), args, summary)
