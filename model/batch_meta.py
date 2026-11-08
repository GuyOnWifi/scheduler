from dataclasses import dataclass, replace
from typing import Self

import torch


@dataclass
class BatchMeta:
    """Per-step description of a batch, built by the engine; the cache is the long-lived state.

    Requests are right-padded: input_ids is [B, T], T = lengths.max().

    lengths:   [B] real new tokens per row; row i's output is the logits after its last
               real token, input_ids[i, lengths[i] - 1]
    slot_ids:  [B] cache row that batch row i lives in
    positions: [B, T] absolute position of each new token within its request; its K/V
               is written to that position of the request's cache row
    """

    lengths: torch.Tensor
    slot_ids: torch.Tensor
    positions: torch.Tensor

    def to(self, device) -> Self:
        return replace(
            self,
            lengths=self.lengths.to(device),
            slot_ids=self.slot_ids.to(device),
            positions=self.positions.to(device),
        )
