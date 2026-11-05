from dataclasses import fields
from typing import Self
import json
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_id: int
    torch_dtype: torch.dtype

    @classmethod
    def from_json(cls, config: str) -> Self:
        obj = json.loads(config)

        kwargs = {f.name: obj[f.name] for f in fields(cls)}
        kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        return cls(**kwargs)
