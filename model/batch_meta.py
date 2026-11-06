from dataclasses import dataclass

import torch


@dataclass
class BatchMeta:
    """Per-step description of a batch, built by the engine.

    Requests are right-padded: input_ids is [B, T], T = lengths.max().

    lengths: [B] real tokens per row; row i's output is the logits after its last
             real token, input_ids[i, lengths[i] - 1]
    """

    lengths: torch.Tensor
