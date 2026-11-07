from tokenizers import Tokenizer
import torch
from typing import Protocol

from model.batch_meta import BatchMeta
from model.kv_cache import KVCache


class Model(Protocol):
    eos_token_ids: frozenset[int]
    vocab_size: int
    tokenizer: Tokenizer

    def alloc_cache(self, batch: int, max_len: int) -> KVCache:
        """An empty cache for `batch` rows of up to `max_len` tokens each."""
        ...

    def forward(
        self, input_ids: torch.Tensor, meta: BatchMeta, kv_cache: KVCache
    ) -> torch.Tensor:
        """New tokens input_ids [B, T], right-padded, with real lengths meta.lengths [B] -> last-token logits [B, vocab].

        Writes the new tokens into kv_cache at meta.positions; each attends to every
        cached position <= its own.
        """
        ...
