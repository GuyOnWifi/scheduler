"""Check an engine's greedy tokens against Hugging Face transformers, one request at a time.

    uv run python test_engine.py [--dtype bf16|fp32] [key=value ...]

key=value pairs go to the engine's constructor (ints are parsed), e.g.
`test_engine.py --dtype fp32 n_batch=4`.

The reference is transformers' own Qwen3 (AutoModelForCausalLM), not our model, so
the test doesn't change when our forward signature does. It runs first, alone, then
is freed before the engine loads.

Requests arrive staggered (some up front, the rest mid-run), there are more of them
than a small batch holds, and their max_tokens differ, so batches/slots get freed
and refilled at different times. Some prompts share a 48-token prefix, for prefix
caching later.

For engines with a prefix cache (hash_to_blocks), the test wraps engine.py's
RunningReq, the free list and the waiting queue from the outside and prints
admissions with a hit, matched tokens, hits on freed (cached) pages, evictions of
cached pages and preemptions. With the paged engine, tok_budget=16 splits prompts
into chunks, and num_pages=16 or 13 runs out of pages mid-run; with 13 the two
long-output requests get preempted. The largest request needs 12 pages.

Batching and kernel choice change the rounding, most visibly in bf16. A first
mismatch where the reference's top-2 logit gap is <= the dtype's noise gap (a tie,
or one bf16 step) is reported as "~ok" rather than a failure.
"""

import json
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from bench.engine import make
from bench.types import Request, TokenEvent

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
NOISE_GAP = {"bf16": 0.125, "fp32": 1e-2}

# a prompt is text, or (n, text): the first n tokens of SHARED_TEXT, then text's tokens
SHARED_TEXT = (
    "You are a helpful assistant running on a small GPU. Answer briefly and "
    "precisely, in plain English, without lists or markdown. If you do not know an "
    "answer, say so instead of guessing. Keep every reply under fifty words, and "
    "never repeat the question back.\n\n"
)
SHARED_LEN = 48
LONG = "A long prompt about engineers. " * 30

PROMPTS = [
    ((SHARED_LEN, " Write a long story about a dragon."), 48),
    ((SHARED_LEN, " Write a long poem about the sea."), 48),
    ("The capital of France is", 12),
    ("Hi", 1),  # finishes straight out of prefill
    ("def fibonacci(n):", 30),
    ("Once upon a time, in a land far, far away, there lived a", 20),
    ("1, 2, 3, 4,", 8),
    ("Explain the KV cache in one sentence.", 25),
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),  # identical pair, same step
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),
    ("The quick brown fox", 5),
    ("Q: What is 17 * 23?\nA:", 16),
    ("Hello", 2),
    ("Write a haiku about GPUs:\n", 18),
    (LONG, 10),
    ("请用一句话解释什么是注意力机制。", 14),
    ((SHARED_LEN, " Tell me a joke about compilers."), 12),
    ((SHARED_LEN, ""), 6),  # exactly the shared prefix
    ((40, ""), 6),  # a cut of it
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),  # repeat, after the pair finishes
    ("The capital of France is", 4),
    (LONG, 6),  # repeat of a finished request
]
UP_FRONT = 10  # added before the first step; the rest arrive one every ARRIVAL_EVERY steps
ARRIVAL_EVERY = 3
MAX_STEPS = 10_000


@dataclass
class PrefixStats:
    admits: int = 0
    preemptions: int = 0
    readmits: int = 0  # admissions after a preemption
    hits: int = 0  # admissions starting with num_computed > 0
    matched_tokens: int = 0
    prompt_tokens: int = 0
    freed_page_hits: int = 0  # matched pages nobody was using (ref_count was 0)
    evictions: int = 0  # cached pages handed out again, dropping their hash
    matched_by_req: dict[int, list[int]] = field(default_factory=dict)  # per admission


def instrument(engine) -> PrefixStats | None:
    """Count prefix hits from the outside: wrap the engine module's RunningReq to see
    num_computed at admission, and the free list to see cached pages being reused."""
    if not hasattr(engine, "hash_to_blocks"):
        return None
    mod = sys.modules[type(engine).__module__]
    stats = PrefixStats()
    seen = set()
    page_size = engine.page_size

    class CountingReq(mod.RunningReq):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stats.admits += 1
            stats.readmits += self.req_id in seen
            seen.add(self.req_id)
            stats.prompt_tokens += len(self.toks)
            stats.matched_by_req.setdefault(self.req_id, []).append(self.num_computed)
            if self.num_computed > 0:
                stats.hits += 1
                stats.matched_tokens += self.num_computed
                matched = self.pages[: self.num_computed // page_size]
                # ref_count was just bumped, so 1 means it was free (cached) before
                stats.freed_page_hits += sum(b.ref_count == 1 for b in matched)

    class CountingFree(OrderedDict):
        def popitem(self, last=True):
            key, blk = super().popitem(last)
            if blk.hash is not None and engine.hash_to_blocks.get(blk.hash) is blk:
                stats.evictions += 1
            return key, blk

    class CountingQueue(deque):
        def appendleft(self, req):  # only preemption puts a request back in front
            stats.preemptions += 1
            super().appendleft(req)

    mod.RunningReq = CountingReq
    engine.free_kv_slots = CountingFree(engine.free_kv_slots)
    engine.waiting_queue = CountingQueue(engine.waiting_queue)
    return stats


def load_tokenizer() -> tuple[Path, Tokenizer, frozenset[int]]:
    path = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors"]))
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    # generation_config.json lists every stop token; config.json only the chat one
    eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
    return path, tok, frozenset(eos if isinstance(eos, list) else [eos])


def encode(tok: Tokenizer, prompt) -> list[int]:
    if isinstance(prompt, str):
        return tok.encode(prompt).ids
    n, text = prompt
    shared = tok.encode(SHARED_TEXT).ids
    assert len(shared) >= n, len(shared)
    return shared[:n] + (tok.encode(text).ids if text else [])


@torch.inference_mode()
def references(
    path: Path, dtype: torch.dtype, eos: frozenset[int], reqs: list[Request]
) -> dict[int, tuple[list[int], list[float]]]:
    """Greedy tokens per request, alone, plus the top-1/top-2 logit gap at each step."""
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
    out = {}
    for req in reqs:
        ids = torch.tensor([req.prompt_ids], device="cuda")
        past = None
        toks, gaps = [], []
        while True:
            res = hf(input_ids=ids, past_key_values=past, use_cache=True)
            past = res.past_key_values
            logits = res.logits[0, -1].float()
            top2 = logits.topk(2).values
            t = int(logits.argmax())
            toks.append(t)
            gaps.append(float(top2[0] - top2[1]))
            if t in eos or len(toks) >= req.max_tokens:
                break
            ids = torch.tensor([[t]], device="cuda")
        out[req.req_id] = (toks, gaps)
    del hf
    torch.cuda.empty_cache()
    return out


def run_engine(engine, reqs: list[Request]) -> tuple[dict[int, list[TokenEvent]], int]:
    events: dict[int, list[TokenEvent]] = {r.req_id: [] for r in reqs}
    pending = list(reqs)
    for r in pending[:UP_FRONT]:
        engine.add_request(r)
    pending = pending[UP_FRONT:]
    steps = 0
    while pending or engine.has_unfinished():
        if pending and steps % ARRIVAL_EVERY == 0:
            engine.add_request(pending.pop(0))
        for ev in engine.step():
            events[ev.req_id].append(ev)
        steps += 1
        if steps > MAX_STEPS:
            raise SystemExit(f"engine didn't finish after {MAX_STEPS} steps")
    return events, steps


def parse_args(rest: list[str]) -> tuple[str, dict]:
    dtype = "bf16"
    kwargs = {}
    i = 0
    while i < len(rest):
        if rest[i] == "--dtype":
            dtype = rest[i + 1]
            i += 2
            continue
        k, _, v = rest[i].partition("=")
        kwargs[k] = int(v) if v.lstrip("-").isdigit() else v
        i += 1
    if dtype not in DTYPES:
        raise SystemExit(f"--dtype must be one of {list(DTYPES)}")
    return dtype, kwargs


def main():
    dtype_name, kwargs = parse_args(sys.argv[1:])
    dtype = DTYPES[dtype_name]
    path, tok, eos = load_tokenizer()
    reqs = [
        Request(i, encode(tok, prompt), max_tokens)
        for i, (prompt, max_tokens) in enumerate(PROMPTS)
    ]
    prompts_before = {r.req_id: list(r.prompt_ids) for r in reqs}

    refs = references(path, dtype, eos, reqs)
    engine = make(dtype=dtype, **kwargs)
    stats = instrument(engine)
    events, steps = run_engine(engine, reqs)

    gap_ok = NOISE_GAP[dtype_name]
    failures = noisy = 0
    for req in reqs:
        want, gaps = refs[req.req_id]
        evs = events[req.req_id]
        got = [ev.token_id for ev in evs]
        problems = []
        i = 0
        if req.prompt_ids != prompts_before[req.req_id]:
            problems.append("engine mutated req.prompt_ids (aliased list?)")
        if not evs:
            problems.append("no tokens")
        else:
            finished_at = [j for j, ev in enumerate(evs) if ev.finished]
            if finished_at != [len(evs) - 1]:
                problems.append(
                    f"finished=True at event indices {finished_at}, want only the last"
                )
        if got != want:
            i = next(
                (j for j, (a, b) in enumerate(zip(got, want)) if a != b),
                min(len(got), len(want)),
            )
            gap = f", reference top-2 gap {gaps[i]:.4f}" if i < len(gaps) else ""
            problems.append(
                f"tokens differ at index {i} (got {len(got)}, want {len(want)}{gap})"
            )
        noise = len(problems) == 1 and got != want and i < len(gaps) and gaps[i] <= gap_ok
        status = "ok  " if not problems else "~ok " if noise else "FAIL"
        print(
            f"{status} req {req.req_id:2d} prompt {len(req.prompt_ids):3d} tok, "
            f"max {req.max_tokens:2d}: "
            + ("; ".join(problems) or repr(tok.decode(got)[:40]))
        )
        if noise:
            noisy += 1
        elif problems:
            print(f"       got  {got}\n       want {want}")
            failures += 1

    print(
        f"\n{dtype_name}, {kwargs or 'defaults'}: "
        f"{len(reqs) - failures - noisy}/{len(reqs)} requests match exactly, "
        f"{noisy} differ at a tie, {failures} fail, {steps} steps"
    )
    if stats:
        print(
            f"prefix cache: {stats.hits}/{stats.admits} admissions hit, "
            f"{stats.preemptions} preemptions ({stats.readmits} readmitted), "
            f"{stats.matched_tokens}/{stats.prompt_tokens} prompt tokens matched, "
            f"{stats.freed_page_hits} matched pages were free (cached), "
            f"{stats.evictions} cached pages evicted"
        )
        shown = {r: m for r, m in stats.matched_by_req.items() if any(m) or len(m) > 1}
        print(
            f"matched tokens per admission (requests with a hit or a readmit): {shown}"
        )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
