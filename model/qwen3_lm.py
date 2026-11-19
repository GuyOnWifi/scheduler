from bisect import bisect_left
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from math import ceil
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from model.kv_cache import PagedKVCache
from model.batch_meta import BatchMeta
from model.kv_cache import KVCache
import json
from pathlib import Path
from typing import Self

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from model.qwen import Qwen3


class StaticDecodeData:
    paged_kv_indptr_gpu: torch.Tensor
    paged_kv_indices_gpu: torch.Tensor
    last_page_lens_gpu: torch.Tensor
    slot_mappings_gpu: torch.Tensor
    positions_gpu: torch.Tensor
    indptr_gpu: torch.Tensor
    input_ids_gpu: torch.Tensor

    def __init__(self, max_batch, max_seq_len, page_size, device):
        num_pages_seq = ceil(max_seq_len / page_size)

        # FlashInfer's buffers (written by plan()), int32
        self.paged_kv_indptr_gpu = torch.zeros(
            max_batch + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_indices_gpu = torch.zeros(
            max_batch * num_pages_seq, dtype=torch.int32, device=device
        )
        self.last_page_lens_gpu = torch.zeros(
            max_batch, dtype=torch.int32, device=device
        )

        # read by the graph itself; decode = one token per request, so all [max_batch]
        self.slot_mappings_gpu = torch.zeros(
            max_batch, dtype=torch.int64, device=device
        )
        self.positions_gpu = torch.zeros(max_batch, dtype=torch.int64, device=device)
        self.input_ids_gpu = torch.zeros(max_batch, dtype=torch.int64, device=device)
        # decode: request i's token is row i, so indptr is always arange (never copied)
        self.indptr_gpu = torch.arange(max_batch + 1, dtype=torch.int32, device=device)
        self.device = device

    # takes in cpu made batch meta, returns a gpu batch meta)
    def view(self, meta: BatchMeta, pad_to):
        nt = meta.positions.shape[0]
        self.slot_mappings_gpu[:pad_to] = 0  # dummy slot
        # pinned + non_blocking, so the copy is queued instead of waiting for the GPU
        self.slot_mappings_gpu[:nt].copy_(meta.slot_mappings.pin_memory(), non_blocking=True)
        self.positions_gpu[:nt].copy_(meta.positions.pin_memory(), non_blocking=True)

        return BatchMeta(
            meta.paged_kv_indptr,
            meta.paged_kv_indices,
            self.slot_mappings_gpu[:pad_to],
            self.positions_gpu[:pad_to],
            self.indptr_gpu[: pad_to + 1],
            meta.last_page_lens,
            meta.page_size,
        )


class Qwen3LM:
    """Our Qwen3 network behind the Model protocol, with Qwen's tokenizer and stop tokens."""

    def __init__(self, net: Qwen3, tokenizer: Tokenizer, eos_token_ids: frozenset[int]):
        self.net = net
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.vocab_size = net.cfg.vocab_size
        self.buckets = {}
        self.graphs = {}
        self.ws = torch.empty(
            128 << 20, dtype=torch.uint8, device="cuda"
        )  # scratch space for the kernel
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.ws, kv_layout="HND"
        )
        self.max_buckets = 0

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

    def alloc_cache(self, num_pages: int, page_size: int) -> PagedKVCache:
        cfg = self.net.cfg
        weight = self.net.embed_tokens.weight
        return PagedKVCache.alloc(
            cfg.num_hidden_layers,
            num_pages,
            cfg.num_key_value_heads,
            page_size,
            cfg.head_dim,
            device=weight.device,
            dtype=weight.dtype,
        )

    def build_graphs(self, buckets, page_size, kv_cache, max_seq_len):

        self.max_buckets = max(buckets)
        self.buckets = sorted(buckets)
        self.data = StaticDecodeData(
            self.max_buckets,
            max_seq_len,
            page_size,
            self.net.embed_tokens.weight.device,
        )

        mem_pool = torch.cuda.graph_pool_handle()

        output = {}
        for b in reversed(self.buckets):
            print(f"graphing {b}")
            sample_input_ids = torch.zeros(b, dtype=torch.int64)
            slot_mappings = torch.zeros(b, dtype=torch.int64)
            last_page_lens = torch.ones(
                b, dtype=torch.int32
            )  # 1..page_size, 0 is invalid
            positions = torch.zeros(b, dtype=torch.int64)
            indptr = torch.arange(b + 1, dtype=torch.int32)

            paged_kv_indptr = torch.arange(b + 1, dtype=torch.int32)
            paged_kv_indices = torch.zeros(b, dtype=torch.int32)

            decode = BatchDecodeWithPagedKVCacheWrapper(
                self.ws,  # float_workspace_buffer, e.g. torch.empty(128 MB, uint8, cuda)
                kv_layout="HND",  # default is NHD! your cache is HND
                use_cuda_graph=True,
                use_tensor_cores=True,  # same numerics as your prefill path
                paged_kv_indptr_buffer=self.data.paged_kv_indptr_gpu[: b + 1],
                paged_kv_indices_buffer=self.data.paged_kv_indices_gpu,
                paged_kv_last_page_len_buffer=self.data.last_page_lens_gpu[:b],
            )

            meta = BatchMeta(
                paged_kv_indptr,
                paged_kv_indices,
                slot_mappings,
                positions,
                indptr,
                last_page_lens,
                page_size,
            )

            meta_gpu = self.data.view(meta, b)
            self.data.input_ids_gpu[:b].copy_(sample_input_ids)

            decode.plan(
                meta.paged_kv_indptr,
                meta.paged_kv_indices,
                meta.last_page_lens,
                self.net.cfg.num_attention_heads,
                self.net.cfg.num_key_value_heads,
                self.net.cfg.head_dim,
                meta.page_size,
                q_data_type=torch.bfloat16,
            )

            with torch.inference_mode():
                # warmup:
                print("warming up")
                for _ in range(3):
                    self.net(self.data.input_ids_gpu[:b], meta_gpu, kv_cache, decode)

                print("recording graph")
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=mem_pool):
                    out = self.net(
                        self.data.input_ids_gpu[:b], meta_gpu, kv_cache, decode
                    )

            output[b] = (decode, g, out)
        self.graphs = output

    def replay_graph(self, input_ids, meta, kv_cache, pad_to):
        nt = input_ids.shape[0]
        b = meta.indptr.shape[0] - 1

        self.data.view(meta, pad_to)
        self.data.input_ids_gpu[:nt].copy_(input_ids)

        (decode_wrapper, g, out) = self.graphs[pad_to]

        # pad to fit
        paged_kv_indptr = torch.cat(
            [
                meta.paged_kv_indptr,
                torch.arange(
                    meta.paged_kv_indptr[-1] + 1,
                    meta.paged_kv_indptr[-1] + 1 + pad_to - b,
                    dtype=torch.int32,
                ),
            ]
        )
        paged_kv_indices = torch.nn.functional.pad(
            meta.paged_kv_indices, (0, pad_to - b)
        )
        # on the GPU: from a CPU tensor, plan() would copy it into the graph's buffer
        # blocking (syncs the stream); GPU -> GPU is queued
        paged_kv_indices = paged_kv_indices.pin_memory().to(
            self.data.device, non_blocking=True
        )
        last_page_lens = torch.nn.functional.pad(
            meta.last_page_lens, (0, pad_to - b), value=1
        )

        decode_wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            last_page_lens,
            self.net.cfg.num_attention_heads,
            self.net.cfg.num_key_value_heads,
            self.net.cfg.head_dim,
            meta.page_size,
            q_data_type=torch.bfloat16,
        )

        g.replay()

        return out[:nt]

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, meta: BatchMeta, kv_cache: KVCache
    ) -> torch.Tensor:

        nt = input_ids.shape[0]
        b = meta.indptr.shape[0] - 1

        if nt == b and nt <= self.max_buckets:
            # we can replay a graph

            # find max graph we can calc
            pad_buckets = self.buckets[bisect_left(self.buckets, nt)]
            return self.replay_graph(input_ids, meta, kv_cache, pad_buckets)

        else:
            # eager mode
            self.prefill_wrapper.plan(
                meta.indptr,
                meta.paged_kv_indptr,
                meta.paged_kv_indices,
                meta.last_page_lens,
                self.net.cfg.num_attention_heads,
                self.net.cfg.num_key_value_heads,
                self.net.cfg.head_dim,
                meta.page_size,
                causal=True,
                q_data_type=self.net.embed_tokens.weight.dtype,
            )
            # plan() above wants the CPU meta; the net wants it on the GPU
            device = self.net.embed_tokens.weight.device
            meta = meta.to(device)
            ids_gpu = input_ids.to(device)

            return self.net(ids_gpu, meta, kv_cache, self.prefill_wrapper)  # [B, vocab]
