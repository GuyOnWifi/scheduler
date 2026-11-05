from model.qwen import Qwen3
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = torch.float32

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

for p in prompts:
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        ref = ref_qwen(ids).logits
        ours = our_qwen(ids)
        torch.testing.assert_close(ref, ours, rtol=1e-3, atol=1e-3)

print("Done testing")
