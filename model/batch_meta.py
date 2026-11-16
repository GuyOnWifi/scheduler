from dataclasses import dataclass, replace
from typing import Self

import torch


@dataclass
class BatchMeta:
    """Per-step description of a batch, built by the engine; the cache is the long-lived state.

    Requests are packed: input_ids is [N], N = lengths.sum(), requests back to back in order.

    lengths:   [B] number of new tokens of request i
    slot_ids:  [B] cache row that batch row i lives in
    positions: [N] absolute position of each new token within its request; its K/V
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
