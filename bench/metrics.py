import argparse
import json
import statistics
from pathlib import Path

from bench.types import RequestRecord


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def distribution_ms(values: list[float], percentiles: list[float]) -> dict[str, float]:
    ms = [v * 1000 for v in values]
    stats = {"mean": statistics.fmean(ms), "median": statistics.median(ms)}
    stats |= {f"p{p:g}": percentile(ms, p) for p in percentiles}
    return stats


def meets_slos(record: RequestRecord, slos_ms: dict[str, float]) -> bool:
    measured = {"ttft": record.ttft, "tpot": record.tpot, "e2el": record.e2el}
    return all(
        measured[name] is None or measured[name] * 1000 <= limit
        for name, limit in slos_ms.items()
    )


def summarize(
    records: list[RequestRecord],
    duration: float,
    percentiles: list[float],
    slos_ms: dict[str, float] | None = None,
) -> dict:
    input_tokens = sum(r.prompt_len for r in records)
    output_tokens = sum(r.output_len for r in records)
    summary = {
        "completed": len(records),
        "duration_s": duration,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput": len(records) / duration,
        "output_throughput": output_tokens / duration,
        "total_token_throughput": (input_tokens + output_tokens) / duration,
        "ttft_ms": distribution_ms([r.ttft for r in records], percentiles),
        "tpot_ms": distribution_ms(
            [r.tpot for r in records if r.tpot is not None], percentiles
        ),
        "itl_ms": distribution_ms([x for r in records for x in r.itls], percentiles),
        "e2el_ms": distribution_ms([r.e2el for r in records], percentiles),
    }
    if slos_ms:
        good = sum(meets_slos(r, slos_ms) for r in records)
        summary["goodput"] = good / duration
    return summary


def print_summary(title: str, summary: dict) -> None:
    print(f"{f' {title} ':=^50}")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{f' {key} ':-^50}")
            for stat, v in value.items():
                print(f"  {stat:<38}{v:>10.2f}")
        elif isinstance(value, int):
            print(f"{key:<40}{value:>10}")
        else:
            print(f"{key:<40}{value:>10.2f}")
    print("=" * 50)


def save_json(path: Path, args: argparse.Namespace, summary: dict) -> None:
    path.write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2))
