from tokenizers import Tokenizer
import torch
from typing import Protocol

from model.kv_cache import KVCache


class Model(Protocol):
    eos_token_ids: frozenset[int]
    vocab_size: int
    tokenizer: Tokenizer

    def alloc_cache(self, batch: int, max_len: int) -> KVCache:
        """An empty cache for `batch` rows of up to `max_len` tokens each."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        kv_cache: KVCache,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """New tokens input_ids [B, T], right-padded, with real lengths [B] -> last-token logits [B, vocab].

        Batch row i lives in cache row slot_ids[i]. Writes the new tokens into those rows
        and advances kv_cache.seq_len[slot_ids] by lengths; other rows are untouched.
        """
        ...
