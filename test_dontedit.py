import torch
import flashinfer

workspace_buffer = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

DIM = 9999
slots = torch.Tensor([])
kv_cache = torch.empty()
to_decode = torch.Tensor()
seq_len = torch.Tensor()  # which position its in

indptr = torch.arange(len(to_decode) + 1)
indices = slots * DIM + seq_len
prefill_wrapper.plan()
