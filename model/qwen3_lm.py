import json
from pathlib import Path
from typing import Self

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from model.qwen import Qwen3


class Qwen3LM:
    """Our Qwen3 network behind the Model protocol, with Qwen's tokenizer and stop tokens."""

    def __init__(self, net: Qwen3, tokenizer: Tokenizer, eos_token_ids: frozenset[int]):
        self.net = net
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.vocab_size = net.cfg.vocab_size

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> Self:
        path = Path(
            snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
        )
        net = Qwen3.load(path, device=device, dtype=dtype).eval()
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        # generation_config.json lists every stop token; config.json only the chat one.
        eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
        return cls(net, tokenizer, frozenset(eos if isinstance(eos, list) else [eos]))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logits = self.net(input_ids.to(self.net.embed_tokens.weight.device))  # [B, T, vocab]
        rows = torch.arange(len(lengths), device=logits.device)
        return logits[rows, lengths.to(logits.device) - 1]
