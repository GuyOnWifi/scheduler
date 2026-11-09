import argparse
from typing import Protocol

from bench.types import Request, TokenEvent


class Engine(Protocol):
    def add_request(self, request: Request) -> None: ...

    def step(self) -> list[TokenEvent]:
        """Run one scheduler iteration; return one event per token produced."""
        ...

    def has_unfinished(self) -> bool: ...


def make(dtype=None, **engine_kwargs) -> Engine:
    """Our Qwen3-0.6B behind the one engine in engine.py (edited in place every rung).

    Now: 8 slots, every step runs all decodes plus prefill chunks within a
    512-token budget; attention is FlashInfer's paged prefill kernel (bf16 only),
    each slot one 2048-token page.
    """
    # Imported here: engine.py imports this module for the Engine protocol, and
    # `bench.py --help` shouldn't have to load torch.
    import torch

    from engine import ChunkedEngine
    from model.qwen3_lm import Qwen3LM

    lm = Qwen3LM.from_pretrained(dtype=dtype or torch.bfloat16)
    return ChunkedEngine(lm, **engine_kwargs)


def from_args(args: argparse.Namespace) -> Engine:
    return make()
