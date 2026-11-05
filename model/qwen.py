import dataclasses
from safetensors import safe_open
import torch
from model.rmsnorm import RMSNorm
from model.block import Block
from model.attention import rope_tables
from model.config import Qwen3Config
from torch import nn


class Qwen3(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, dtype=cfg.torch_dtype
        )

        self.layers = nn.ModuleList(
            [
                Block(
                    cfg.hidden_size,
                    cfg.head_dim,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.intermediate_size,
                    cfg.torch_dtype,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(cfg.hidden_size, cfg.torch_dtype)
        self.lm_head = nn.Linear(
            cfg.hidden_size, cfg.vocab_size, bias=False, dtype=cfg.torch_dtype
        )

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        sin, cos = rope_tables(cfg.head_dim)
        self.register_buffer("sin", sin.to(dtype=cfg.torch_dtype), persistent=False)
        self.register_buffer("cos", cos.to(dtype=cfg.torch_dtype), persistent=False)

    def forward(self, x):
        nb, nt = x.shape

        x = self.embed_tokens(x)
        mask = torch.triu(
            torch.ones(nt, nt, device=x.device, dtype=torch.bool), diagonal=1
        )

        for blk in self.layers:
            x = blk(x, mask, self.sin, self.cos)

        x = self.norm(x)
        x = self.lm_head(x)

        return x

    @classmethod
    def load(cls, path, device="cpu", dtype=None):

        with open(path / "config.json") as f:
            config_raw = f.read()

        config = Qwen3Config.from_json(config_raw)

        if dtype is not None:
            config = dataclasses.replace(config, torch_dtype=dtype)

        with torch.device(device):
            self = cls(config)

        params = dict(self.named_parameters())
        loaded = set()

        with safe_open(path / "model.safetensors", framework="pt", device=device) as f:
            for name in f.keys():
                param_name = name.removeprefix("model.")

                if param_name == "lm_head.weight" and self.cfg.tie_word_embeddings:
                    continue  # tied: same tensor as embed_tokens

                if param_name not in params:
                    raise KeyError(f"Weights {name} is not part of our model!")

                t = f.get_tensor(name)
                p = params[param_name]

                if p.shape != t.shape:
                    raise ValueError(
                        f"{name}: model {tuple(p.shape)} vs checkpoint {tuple(t.shape)}"
                    )
                with torch.no_grad():
                    p.copy_(t)

                loaded.add(param_name)

        leftover = set(params) - loaded
        if leftover:
            raise KeyError(f"not in checkpoint: {sorted(leftover)}")
        return self
