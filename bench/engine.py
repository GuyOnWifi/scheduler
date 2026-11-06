import argparse
from typing import Protocol

from bench.types import Request, TokenEvent


class Engine(Protocol):
    def add_request(self, request: Request) -> None: ...

    def step(self) -> list[TokenEvent]:
        """Run one scheduler iteration; return one event per token produced."""
        ...

    def has_unfinished(self) -> bool: ...


def static_qwen3() -> Engine:
    # Imported here: engines/ imports this module for the Engine protocol, and
    # `bench.py --help` shouldn't have to load torch.
    import torch

    from engines.static_batch import StaticBatching
    from model.qwen3_lm import Qwen3LM

    return StaticBatching(Qwen3LM.from_pretrained(dtype=torch.bfloat16))


ENGINES = {"static": static_qwen3}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="static",
        help="static: our Qwen3-0.6B, static batches of 8, no KV cache",
    )


def from_args(args: argparse.Namespace) -> Engine:
    return ENGINES[args.engine]()
