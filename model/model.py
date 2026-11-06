from tokenizers import Tokenizer
import torch
from typing import Protocol


class Model(Protocol):
    eos_token_ids: frozenset[int]
    vocab_size: int
    tokenizer: Tokenizer

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Right-padded input_ids [B, T] with real lengths [B] -> last-token logits [B, vocab]."""
        ...
