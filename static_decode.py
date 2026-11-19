from model.qwen3_lm import Qwen3LM
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from model.kv_cache import PagedKVCache
from model.kv_cache import KVCache
from math import ceil
import torch
from model.batch_meta import BatchMeta
from dataclasses import dataclass


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
        self.slot_mappings_gpu[:nt].copy_(meta.slot_mappings)
        self.positions_gpu[:nt].copy_(meta.positions)

        return BatchMeta(
            meta.paged_kv_indptr,
            meta.paged_kv_indices,
            self.slot_mappings_gpu[:pad_to],
            self.positions_gpu[:pad_to],
            self.indptr_gpu[: pad_to + 1],
            meta.last_page_lens,
            meta.page_size,
        )


qwen = Qwen3LM.from_pretrained()

buckets = [1, 2, 4, 8, 16, 32, 64]
B = max(buckets)
page_size = 16
data = StaticDecodeData(B, 512, page_size, "cuda")
kv_cache = qwen.alloc_cache(64, page_size)


workspace_buffer = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")


mem_pool = torch.cuda.graph_pool_handle()

thing = []
for b in reversed(buckets):
    print(f"graphing {b}")
    sample_input_ids = torch.zeros(b, dtype=torch.int64)
    slot_mappings = torch.zeros(b, dtype=torch.int64)
    last_page_lens = torch.ones(b, dtype=torch.int32)  # 1..page_size, 0 is invalid
    positions = torch.zeros(b, dtype=torch.int64)
    indptr = torch.arange(b + 1, dtype=torch.int32)

    paged_kv_indptr = torch.arange(b + 1, dtype=torch.int32)
    paged_kv_indices = torch.zeros(b, dtype=torch.int32)

    decode = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,  # float_workspace_buffer, e.g. torch.empty(128 MB, uint8, cuda)
        kv_layout="HND",  # default is NHD! your cache is HND
        use_cuda_graph=True,
        use_tensor_cores=True,  # same numerics as your prefill path
        paged_kv_indptr_buffer=data.paged_kv_indptr_gpu[: b + 1],
        paged_kv_indices_buffer=data.paged_kv_indices_gpu,
        paged_kv_last_page_len_buffer=data.last_page_lens_gpu[:b],
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

    meta_gpu = data.view(meta, b)
    data.input_ids_gpu[:b].copy_(sample_input_ids)
    ids_gpu = data.input_ids_gpu[:b]

    decode.plan(
        meta.paged_kv_indptr,
        meta.paged_kv_indices,
        meta.last_page_lens,
        qwen.net.cfg.num_attention_heads,
        qwen.net.cfg.num_key_value_heads,
        qwen.net.cfg.head_dim,
        meta.page_size,
        q_data_type=torch.bfloat16,
    )

    with torch.inference_mode():
        # warmup:
        print("warming up")
        for _ in range(3):
            qwen.forward(data.input_ids_gpu[:b], meta_gpu, kv_cache, decode)

        print("recording graph")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=mem_pool):
            qwen.forward(data.input_ids_gpu[:b], meta_gpu, kv_cache, decode)

    thing.append((decode, g))

print(thing)
