from tokenizers import Tokenizer
import torch
from typing import Protocol

from model.batch_meta import BatchMeta


class Model(Protocol):
    eos_token_ids: frozenset[int]
    vocab_size: int
    tokenizer: Tokenizer

    def forward(self, input_ids: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
        """Right-padded input_ids [B, T] with real lengths meta.lengths [B] -> last-token logits [B, vocab]."""
        ...
