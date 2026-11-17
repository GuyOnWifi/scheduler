"""Check an engine's greedy tokens against a one-request-at-a-time reference, in bf16.

    uv run python test_engine.py paged [TOK_BUDGET] [NUM_PAGES]

The model takes a paged KV cache now, so only the paged engine runs (static,
continuous, chunked and ragged still build the old metadata). For paged, a small
TOK_BUDGET (e.g. 16) splits prompts into several chunks, and a small NUM_PAGES
(e.g. 16 or 13 pages of 16 tokens) runs out of memory mid-run; with 13 the two
long-output requests get preempted (the admission reserve avoids it otherwise). The
largest request needs 12 pages, so stay at or above that.

Requests arrive staggered (some up front, the rest mid-run), there are more of them
than slots, and their max_tokens differ, so slots get freed and refilled at
different times.

Some prompts share a 48-token (3-page) prefix to exercise prefix caching: two
identical ones arrive in the same step (both miss), later ones hit it, one is exactly
the prefix (a multiple of 16: only 2 pages may match, the last token is computed),
one is a 40-token cut of it, and repeats arrive late enough to hit pages of finished
requests. A small NUM_PAGES also evicts cached pages. For engines with a prefix
cache (hash_to_blocks), the test wraps RunningReq, the free list and the waiting
queue from the outside and prints admissions with a hit, matched tokens, hits on
freed (cached) pages, evictions of cached pages and preemptions.

FlashInfer has no fp32 kernels, so this runs in bf16, where batching changes the
rounding. A first mismatch where the reference's top-2 logit gap is <= NOISE_GAP
(a tie or one bf16 step) is reported as "~ok" rather than a failure.
"""

import sys
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from math import ceil

import torch

from bench.types import Request, TokenEvent
from model.batch_meta import BatchMeta
from model.qwen3_lm import Qwen3LM

N_SLOTS = 4
PAGE_SIZE = 16

# a prompt is text, or (n, text): the first n tokens of SHARED_TEXT, then text's tokens
SHARED_TEXT = (
    "You are a helpful assistant running on a small GPU. Answer briefly and "
    "precisely, in plain English, without lists or markdown. If you do not know an "
    "answer, say so instead of guessing. Keep every reply under fifty words, and "
    "never repeat the question back.\n\n"
)
SHARED_LEN = 48  # 3 full pages
LONG = "A long prompt about engineers. " * 30

PROMPTS = [
    # long outputs: with a small NUM_PAGES these get preempted and, when readmitted,
    # match their own cached pages (prompt and generated tokens)
    ((SHARED_LEN, " Write a long story about a dragon."), 48),
    ((SHARED_LEN, " Write a long poem about the sea."), 48),
    ("The capital of France is", 12),
    ("Hi", 1),  # finishes straight out of prefill
    ("def fibonacci(n):", 30),
    ("Once upon a time, in a land far, far away, there lived a", 20),
    ("1, 2, 3, 4,", 8),
    ("Explain the KV cache in one sentence.", 25),
    # same step, identical: both miss and compute the same pages
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),
    ("The quick brown fox", 5),
    ("Q: What is 17 * 23?\nA:", 16),
    ("Hello", 2),
    ("Write a haiku about GPUs:\n", 18),
    (LONG, 10),
    ("请用一句话解释什么是注意力机制。", 14),
    ((SHARED_LEN, " Tell me a joke about compilers."), 12),  # hits the 3 shared pages
    ((SHARED_LEN, ""), 6),  # exactly the prefix: 2 pages match, last page computed
    ((40, ""), 6),  # 40 tokens of it: 2 pages match
    ((SHARED_LEN, "Q: Name a colour.\nA:"), 8),  # repeat of the pair, after they finish
    ("The capital of France is", 4),  # under a page: nothing to match
    (LONG, 6),  # repeat: 11 pages of a finished request
]
UP_FRONT = (
    10  # added before the first step; the rest arrive one every ARRIVAL_EVERY steps
)
ARRIVAL_EVERY = 3
NOISE_GAP = 0.125


def make_engine(name: str, lm: Qwen3LM):
    if name == "static":
        from engines.static_batch import StaticBatching

        return StaticBatching(lm, n_batch=N_SLOTS)
    if name == "continuous":
        from engines.continous_batch import ContinousBatching

        return ContinousBatching(lm, n_batch=N_SLOTS)

    if name == "chunked":
        from engines.chunked import ChunkedEngine

        return ChunkedEngine(lm, n_batch=N_SLOTS)

    if name == "ragged":
        from engines.chunked_ragged import ChunkedRagged

        return ChunkedRagged(lm, n_batch=N_SLOTS)

    if name == "paged":
        from engines.chunked_ragged_paged import ChunkedRaggedPaged

        kwargs = {}
        if len(sys.argv) > 2:
            kwargs["tok_budget"] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs["num_pages"] = int(sys.argv[3])
        return ChunkedRaggedPaged(lm, page_size=PAGE_SIZE, **kwargs)

    raise SystemExit(f"unknown engine {name!r}; use paged")


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


def encode(lm: Qwen3LM, prompt) -> list[int]:
    if isinstance(prompt, str):
        return lm.tokenizer.encode(prompt).ids
    n, text = prompt
    shared = lm.tokenizer.encode(SHARED_TEXT).ids
    assert len(shared) >= n, len(shared)
    return shared[:n] + (lm.tokenizer.encode(text).ids if text else [])


def reference(lm: Qwen3LM, req: Request) -> tuple[list[int], list[float]]:
    """Greedy tokens for one request alone, plus the top-1/top-2 logit gap at each step."""
    ps = PAGE_SIZE
    num_pages = ceil((len(req.prompt_ids) + req.max_tokens) / ps)
    cache = lm.alloc_cache(num_pages, ps)
    pages = list(range(num_pages))  # this request owns the whole pool, in order
    ids = torch.tensor(req.prompt_ids)
    toks, gaps = [], []
    pos = 0
    while True:
        n = len(ids)
        kv_len = pos + n  # KV length after this step, new tokens included
        used = ceil(kv_len / ps)
        meta = BatchMeta(
            lengths=torch.tensor([n]),
            paged_kv_indptr=torch.tensor([0, used], dtype=torch.int32),
            paged_kv_indices=torch.tensor(pages[:used], dtype=torch.int32),
            slot_mappings=torch.tensor(
                [pages[p // ps] * ps + p % ps for p in range(pos, kv_len)]
            ),
            positions=torch.arange(pos, kv_len),
            indptr=torch.tensor([0, n], dtype=torch.int32),
            last_page_lens=torch.tensor([(kv_len - 1) % ps + 1], dtype=torch.int32),
            page_size=ps,
        )
        logits = lm.forward(ids, meta, cache)[0].float()
        pos += n
        top2 = logits.topk(2).values
        t = int(logits.argmax())
        toks.append(t)
        gaps.append(float(top2[0] - top2[1]))
        if t in lm.eos_token_ids or len(toks) >= req.max_tokens:
            return toks, gaps
        ids = torch.tensor([t])


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
        if steps > 10_000:
            raise SystemExit("engine didn't finish after 10000 steps")
    return events, steps


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "continuous"
    lm = Qwen3LM.from_pretrained(dtype=torch.bfloat16)
    reqs = [
        Request(i, encode(lm, prompt), max_tokens)
        for i, (prompt, max_tokens) in enumerate(PROMPTS)
    ]

    # references first: they must see the original prompts
    refs = {req.req_id: reference(lm, req) for req in reqs}
    prompts_before = {req.req_id: list(req.prompt_ids) for req in reqs}
    engine = make_engine(name, lm)
    stats = instrument(engine)
    events, steps = run_engine(engine, reqs)

    failures = noisy = 0
    for req in reqs:
        want, gaps = refs[req.req_id]
        evs = events[req.req_id]
        got = [ev.token_id for ev in evs]
        problems = []
        if req.prompt_ids != prompts_before[req.req_id]:
            problems.append("engine mutated req.prompt_ids (aliased list?)")
        if not evs:
            problems.append("no tokens")
        else:
            finished_at = [i for i, ev in enumerate(evs) if ev.finished]
            if finished_at != [len(evs) - 1]:
                problems.append(
                    f"finished=True at event indices {finished_at}, want only the last"
                )
        if got != want:
            i = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                min(len(got), len(want)),
            )
            gap = f", reference top-2 gap {gaps[i]:.4f}" if i < len(gaps) else ""
            problems.append(
                f"tokens differ at index {i} (got {len(got)}, want {len(want)}{gap})"
            )
        noise = (
            len(problems) == 1
            and got != want
            and i < len(gaps)
            and gaps[i] <= NOISE_GAP
        )
        status = "ok  " if not problems else "~ok " if noise else "FAIL"
        print(
            f"{status} req {req.req_id:2d} prompt {len(prompts_before[req.req_id]):3d} tok, max {req.max_tokens:2d}: "
            + ("; ".join(problems) or repr(lm.tokenizer.decode(got)[:40]))
        )
        if noise:
            noisy += 1
        elif problems:
            print(f"       got  {got}\n       want {want}")
            failures += 1

    print(
        f"\n{name}: {len(reqs) - failures - noisy}/{len(reqs)} requests match exactly, "
        f"{noisy} differ at a bf16 tie, {failures} fail, {steps} steps"
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
        print(f"matched tokens per admission (requests with a hit or a readmit): {shown}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
