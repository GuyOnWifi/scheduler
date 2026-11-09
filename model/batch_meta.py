from dataclasses import dataclass, replace
from typing import Self

import torch


@dataclass
class BatchMeta:
    """Per-step description of a batch, built by the engine; the cache is the long-lived state.

    lengths:   [B] real number of new tokens in each batch row (input_ids is right-padded)
    slot_ids:  [B] cache row that batch row i lives in
    positions: [B, T] absolute position of each new token (padding continues the count)
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
