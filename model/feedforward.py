from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embd_dim, hidden_dim, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(embd_dim, hidden_dim, dtype=dtype, bias=False)
        self.up_proj = nn.Linear(embd_dim, hidden_dim, dtype=dtype, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embd_dim, dtype=dtype, bias=False)

    def forward(self, x):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        x3 = nn.functional.silu(x1) * x2
        return self.down_proj(x3)
