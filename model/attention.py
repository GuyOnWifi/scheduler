from flashinfer import BatchPrefillWithPagedKVCacheWrapper
import torch
from model.rmsnorm import RMSNorm
from torch import nn


def rope_tables(head_dim, base=1_000_000, context_len=4096):
    thetas = base ** (-2 * torch.arange(0, head_dim // 2) / head_dim)
    positions = torch.arange(0, context_len)

    angles = positions.unsqueeze(1) * thetas.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=-1)

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return (sin, cos)


def apply_rope(x, sin, cos):
    nb, nh, nt, head_dim = x.shape
    # split it into two
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]

    cos = cos[:, None]
    sin = sin[:, None]

    x_rot = x * cos + torch.cat([-x2, x1], dim=-1) * sin
    return x_rot.to(dtype=x.dtype)


class GroupQueryAttention(nn.Module):
    def __init__(self, embd_dim, head_dim, num_q_heads, num_kv_heads, dtype):
        super().__init__()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_q_heads // num_kv_heads
        self.out_dim = num_q_heads * head_dim

        self.q_proj = nn.Linear(embd_dim, self.out_dim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(
            embd_dim, num_kv_heads * head_dim, dtype=dtype, bias=False
        )
        self.v_proj = nn.Linear(
            embd_dim, num_kv_heads * head_dim, dtype=dtype, bias=False
        )

        self.q_norm = RMSNorm(head_dim, dtype)
        self.k_norm = RMSNorm(head_dim, dtype)

        self.o_proj = nn.Linear(self.out_dim, embd_dim, dtype=dtype, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        meta,
        kv_cache,
        rope_params,
        prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper,
    ):
        # x is (num_batch, num_toks, embd_dim)
        # mask is (num_batch, num_toks, max_toks), True = blocked
        # positions is (num_batch, num_toks), absolute position of each new token
        # kv_cache is this layer's (k, v), each (batch, num_kv_heads, max_len, head_dim)
        # slot_ids is (num_batch), the cache row of each batch row
        (nb, nt, embd_dim) = x.shape

        q = self.q_proj(x)  # (num_batch, num_toks, q_heads * head_dim)
        k = self.k_proj(x)  # (num_batch, num_toks, kv_heads * head_dim)
        v = self.v_proj(x)  # (num_batch, num_toks, kv_heads * head_dim)

        q_head = q.view(nb, nt, self.num_q_heads, self.head_dim).transpose(
            1, 2
        )  # (nb, qh, nt, hd)
        k_head = k.view(nb, nt, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_head = v.view(nb, nt, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_norm = self.q_norm(q_head)
        k_norm = self.k_norm(k_head)

        (sin, cos) = rope_params
        q_rope = apply_rope(q_norm, sin, cos)
        k_rope = apply_rope(k_norm, sin, cos)

        (k_cache, v_cache) = kv_cache
        rows = meta.slot_ids[:, None]  # [B, 1]
        k_cache[rows, :, meta.positions] = k_rope.transpose(
            1, 2
        )  # [B, T, kv_heads, head_dim]
        v_cache[rows, :, meta.positions] = v_head.transpose(
            1, 2
        )  # [B, T, kv_heads, head_dim]

        valid = torch.arange(nt, device=x.device) < meta.lengths[:, None]
        out = prefill_wrapper.run(
            q_rope.transpose(1, 2)[valid], (k_cache, v_cache)
        )  # (N, qh, hd)

        # unpack back to (nb, nt, ...); padding rows stay zero
        o = x.new_zeros(nb, nt, self.num_q_heads, self.head_dim)
        o[valid] = out
        return self.o_proj(o.reshape(nb, nt, self.out_dim))
