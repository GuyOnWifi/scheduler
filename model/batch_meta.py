from dataclasses import dataclass, replace
from typing import Self

import torch


@dataclass
class BatchMeta:
    """Per-step description of a batch, built by the engine; the cache is the long-lived state.

    Requests are packed: input_ids is [N], N = indptr[-1], requests back to back in order.

    paged_kv_indptr:  [B+1] int32, request i's pages are
                      paged_kv_indices[paged_kv_indptr[i]:paged_kv_indptr[i+1]]
    paged_kv_indices: [paged_kv_indptr[-1]] int32, KV page ids, in order, covering each request's
                      KV after this step (new tokens included)
    slot_mappings:    [N] where each new token's K/V goes in the flattened pool:
                      page * page_size + offset within the page
    positions:        [N] absolute position of each new token within its request
    indptr:           [B+1] int32, request i's new tokens are input_ids[indptr[i]:indptr[i+1]]
                      (cumsum of each request's new-token count, with a leading 0)
    last_page_lens:   [B] int32, tokens in each request's last page after this step,
                      1..page_size
    page_size:        tokens per KV page (a plain int, stays on the host)
    """

    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    slot_mappings: torch.Tensor
    positions: torch.Tensor
    indptr: torch.Tensor
    last_page_lens: torch.Tensor
    page_size: int

    def to(self, device) -> Self:
        return replace(
            self,
            paged_kv_indptr=self.paged_kv_indptr.to(device),
            paged_kv_indices=self.paged_kv_indices.to(device),
            slot_mappings=self.slot_mappings.to(device),
            positions=self.positions.to(device),
            indptr=self.indptr.to(device),
            last_page_lens=self.last_page_lens.to(device),
        )
