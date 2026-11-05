from model.feedforward import FeedForward
from model.attention import GroupQueryAttention
from model.rmsnorm import RMSNorm
from torch import nn


class Block(nn.Module):
    def __init__(
        self, embd_dim, head_dim, num_q_heads, num_kv_heads, hidden_dim, dtype
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(embd_dim, dtype)
        self.self_attn = GroupQueryAttention(
            embd_dim, head_dim, num_q_heads, num_kv_heads, dtype
        )
        self.post_attention_layernorm = RMSNorm(embd_dim, dtype)
        self.mlp = FeedForward(embd_dim, hidden_dim, dtype)

    def forward(self, x, mask, sin, cos):
        res = self.input_layernorm(x)
        res = self.self_attn(res, mask, sin, cos)
        x += res

        res = self.post_attention_layernorm(x)
        res = self.mlp(res)
        x += res

        return x
