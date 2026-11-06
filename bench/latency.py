"""Single-batch latency: one fixed batch, nothing else running; the latency floor."""

import argparse
import time
from pathlib import Path

from bench import engine, metrics
from bench.datasets import random_dataset
from bench.engine import Engine
from bench.types import Request


def time_batch(eng: Engine, requests: list[Request]) -> list[float]:
    """Submit the whole batch, step until done, and return each step's duration (s).

    The first step is the prefill (plus first token); the rest are decode steps.
    """
    start = time.perf_counter()
    for r in requests:
        eng.add_request(r)

    res = []
    prev = start
    while eng.has_unfinished():
        eng.step()
        cur = time.perf_counter()
        res.append(cur - prev)
        prev = cur

    return res


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size", type=int, default=8, metavar="N", help="requests in the batch"
    )
    parser.add_argument(
        "--input-len", type=int, default=32, metavar="TOKENS", help="prompt length"
    )
    parser.add_argument(
        "--output-len", type=int, default=128, metavar="TOKENS", help="tokens generated"
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=3,
        metavar="N",
        help="untimed runs of the batch first",
    )
    parser.add_argument(
        "--num-iters", type=int, default=10, metavar="N", help="timed runs of the batch"
    )


def run(args: argparse.Namespace) -> None:
    batch = random_dataset(
        args.batch_size,
        args.input_len,
        args.output_len,
        range_ratio=0.0,
        seed=args.seed,
    )
    eng = engine.from_args(args)
    for _ in range(args.num_iters_warmup):
        time_batch(eng, batch)
    runs = [time_batch(eng, batch) for _ in range(args.num_iters)]

    percentiles = [50.0, 90.0, 99.0]
    summary = {
        "batch_ms": metrics.distribution_ms([sum(r) for r in runs], percentiles),
        "prefill_ms": metrics.distribution_ms([r[0] for r in runs], percentiles),
        "decode_step_ms": metrics.distribution_ms(
            [s for r in runs for s in r[1:]], percentiles
        ),
    }
    metrics.print_summary("Latency Benchmark", summary)
    if args.output_json:
        metrics.save_json(Path(args.output_json), args, summary)
