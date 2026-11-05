import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, embd_dim, dtype, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embd_dim, dtype=dtype))

    def forward(self, x: torch.Tensor):
        x_f32 = x.to(torch.float32)
        mean_squared = x_f32.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x_f32 * torch.rsqrt(mean_squared + self.eps)
        return self.weight * norm_x.to(x.dtype)
