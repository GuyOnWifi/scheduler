from tokenizers import Tokenizer
import torch
from typing import Protocol

from model.batch_meta import BatchMeta
from model.kv_cache import PagedKVCache


class Model(Protocol):
    eos_token_ids: frozenset[int]
    vocab_size: int
    tokenizer: Tokenizer

    def alloc_cache(self, num_pages: int, page_size: int) -> PagedKVCache:
        """An empty pool of `num_pages` KV pages of `page_size` tokens each, shared by all
        requests; which pages a request owns is the engine's business."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """New tokens of B requests, packed, input_ids [N] -> last-token logits [B, vocab].

        Request i's tokens are the next meta.lengths[i] entries of input_ids (no padding;
        N = lengths.sum()), at absolute positions given by meta.positions [N]. They are
        written into cache row meta.slot_ids[i] and attend to everything already in that
        row; other rows are untouched. Row i of the result is the logits after request
        i's last new token.
        """
        ...
