from typing import Self
from dataclasses import dataclass
import torch


@dataclass
class KVCache:
    k: torch.Tensor  # (layers, batch, num_kv_heads, max_len, head_dim)
    v: torch.Tensor
    seq_len: torch.Tensor  # (batch)

    @classmethod
    def alloc(
        cls, layers, batch, num_kv_heads, max_len, head_dim, device, dtype
    ) -> Self:
        return cls(
            torch.zeros(
                (layers, batch, num_kv_heads, max_len, head_dim),
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                (layers, batch, num_kv_heads, max_len, head_dim),
                device=device,
                dtype=dtype,
            ),
            torch.zeros((batch,), device=device, dtype=torch.int32),
        )


@dataclass
class PagedKVCache:
    kv: torch.Tensor  # (layers, num_blocks, 2, num_kv_heads, block_size, head_dim)

    @classmethod
    def alloc(
        cls, layers, num_pages, num_kv_heads, page_size, head_dim, device, dtype
    ) -> Self:
        return cls(
            torch.zeros(
                (layers, num_pages, 2, num_kv_heads, page_size, head_dim),
                dtype=dtype,
                device=device,
            )
        )
