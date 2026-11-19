from model.qwen import Qwen3
from model.batch_meta import BatchMeta
from model.kv_cache import PagedKVCache
from math import ceil
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = torch.bfloat16  # FlashInfer has no fp32 kernels
NOISE_GAP = 0.125  # one bf16 step at these logit sizes

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ref_qwen = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(device).eval()

SYSTEM = "You are a helpful assistant that answers concisely and accurately. " * 4

prompts = [
    # edge cases
    "Hi",
    "Who are you?",
    "!!! ??? ... ,,, ;;; :::",
    # varied content
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Compute 37 * 412 step by step. First, 37 * 400 =",
    "请用一句话解释什么是注意力机制。",
    "The word 'cache' in French is 'cache', in Chinese it is '缓存', and in Japanese",
    "the the the the the the the the the the the the the the the the",
    '{"model": "qwen3", "layers": 28, "heads": {"q": 16, "kv": 8}, "rope":',
    # long: spans many KV blocks / forces chunked prefill
    "Once upon a time, in a land far away, there lived a curious engineer who " * 60,
    # shared prefix: for prefix caching
    SYSTEM + "What is RoPE?",
    SYSTEM + "What is GQA?",
    SYSTEM + "What is a KV cache?",
]

path = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors"]))

our_qwen = Qwen3.load(path, device="cuda", dtype=DTYPE).eval()
wrapper = BatchPrefillWithPagedKVCacheWrapper(
    torch.empty(128 << 20, dtype=torch.uint8, device=device), kv_layout="HND"
)

for p in prompts:
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        ref = ref_qwen(ids).logits
        # the net takes (x, meta, kv_cache) now: one full prefill into a fresh cache
        n = ids.shape[1]
        cfg = our_qwen.cfg
        ps = 16
        used = ceil(n / ps)  # this prompt owns pages 0..used-1 of its own pool
        cache = PagedKVCache.alloc(
            cfg.num_hidden_layers, used, cfg.num_key_value_heads, ps, cfg.head_dim,
            device=device, dtype=DTYPE,
        )
        i32 = dict(dtype=torch.int32, device=device)
        meta = BatchMeta(
            paged_kv_indptr=torch.tensor([0, used], **i32),
            paged_kv_indices=torch.arange(used, **i32),
            slot_mappings=torch.arange(n, device=device),
            positions=torch.arange(n, device=device),
            indptr=torch.tensor([0, n], **i32),
            last_page_lens=torch.tensor([(n - 1) % ps + 1], **i32),
            page_size=ps,
        )
        # the net takes the attention wrapper too now (Qwen3LM owns and plans it)
        wrapper.plan(
            meta.indptr, meta.paged_kv_indptr, meta.paged_kv_indices,
            meta.last_page_lens, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, ps, causal=True, q_data_type=DTYPE,
        )
        ours = our_qwen(ids[0], meta, cache, wrapper)  # packed [N] in, last-token logits [1, vocab] out
        # bf16 now, so not elementwise close to HF: compare the last token's logits,
        # greedy pick must match unless HF's top-2 are within NOISE_GAP
        ref_last, ours_last = ref[0, -1].float(), ours[0].float()
        top2 = ref_last.topk(2).values
        diff = (ref_last - ours_last).abs().max().item()
        same = int(ref_last.argmax()) == int(ours_last.argmax())
        print(f"{'ok  ' if same else '~ok ' if top2[0] - top2[1] <= NOISE_GAP else 'FAIL'} "
              f"{n:3d} tok, max |logit diff| {diff:.3f}")
        assert same or top2[0] - top2[1] <= NOISE_GAP, p

print("Done testing")
