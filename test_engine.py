"""Check an engine's greedy tokens against a one-request-at-a-time reference, in fp32.

    uv run python test_engine.py static
    uv run python test_engine.py continuous

Requests arrive staggered (some up front, the rest mid-run), there are more of them
than slots, and their max_tokens differ, so slots get freed and refilled at
different times.
"""

import sys

import torch

from bench.types import Request, TokenEvent
from model.qwen3_lm import Qwen3LM

N_SLOTS = 4

PROMPTS = [
    ("The capital of France is", 12),
    ("Hi", 1),  # finishes straight out of prefill
    ("def fibonacci(n):", 30),
    ("Once upon a time, in a land far, far away, there lived a", 20),
    ("1, 2, 3, 4,", 8),
    ("Explain the KV cache in one sentence.", 25),
    ("The quick brown fox", 5),
    ("Q: What is 17 * 23?\nA:", 16),
    ("Hello", 2),
    ("Write a haiku about GPUs:\n", 18),
    ("A long prompt about engineers. " * 30, 10),
    ("请用一句话解释什么是注意力机制。", 14),
]
UP_FRONT = 6  # added before the first step; the rest arrive one every ARRIVAL_EVERY steps
ARRIVAL_EVERY = 3


def make_engine(name: str, lm: Qwen3LM):
    if name == "static":
        from engines.static_batch import StaticBatching

        return StaticBatching(lm, n_batch=N_SLOTS)
    if name == "continuous":
        from engines.continous_batch import ContinousBatching

        return ContinousBatching(lm, n_batch=N_SLOTS)
    raise SystemExit(f"unknown engine {name!r}; use static or continuous")


def reference(lm: Qwen3LM, req: Request) -> tuple[list[int], list[float]]:
    """Greedy tokens for one request alone, plus the top-1/top-2 logit gap at each step."""
    cache = lm.alloc_cache(1, len(req.prompt_ids) + req.max_tokens)
    ids = torch.tensor([req.prompt_ids])
    toks, gaps = [], []
    while True:
        logits = lm.forward(ids, torch.tensor([ids.shape[1]]), cache, torch.tensor([0]))[0].float()
        top2 = logits.topk(2).values
        t = int(logits.argmax())
        toks.append(t)
        gaps.append(float(top2[0] - top2[1]))
        if t in lm.eos_token_ids or len(toks) >= req.max_tokens:
            return toks, gaps
        ids = torch.tensor([[t]])


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
    lm = Qwen3LM.from_pretrained(dtype=torch.float32)
    reqs = [
        Request(i, lm.tokenizer.encode(text).ids, max_tokens)
        for i, (text, max_tokens) in enumerate(PROMPTS)
    ]

    # references first: they must see the original prompts
    refs = {req.req_id: reference(lm, req) for req in reqs}
    prompts_before = {req.req_id: list(req.prompt_ids) for req in reqs}
    events, steps = run_engine(make_engine(name, lm), reqs)

    failures = 0
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
                problems.append(f"finished=True at event indices {finished_at}, want only the last")
        if got != want:
            i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            gap = f", reference top-2 gap {gaps[i]:.4f}" if i < len(gaps) else ""
            problems.append(f"tokens differ at index {i} (got {len(got)}, want {len(want)}{gap})")
        status = "ok  " if not problems else "FAIL"
        print(f"{status} req {req.req_id:2d} prompt {len(prompts_before[req.req_id]):3d} tok, max {req.max_tokens:2d}: "
              + ("; ".join(problems) or repr(lm.tokenizer.decode(got)[:40])))
        if problems:
            print(f"       got  {got}\n       want {want}")
            failures += 1

    print(f"\n{name}: {len(reqs) - failures}/{len(reqs)} requests match, {steps} steps")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
