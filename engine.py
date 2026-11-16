from collections import deque
from model.kv_cache import KVCache
from dataclasses import dataclass

import torch

from bench.engine import Engine
from bench.types import Request, TokenEvent
from model.kv_cache import KVCache
from model.batch_meta import BatchMeta
from model.model import Model


@dataclass
class RunningReq:
    req_id: int
    toks: list[int]
    max_tokens: int
    eos_ids: frozenset[int]
    slot: int
    generated: int = 0
    num_computed: int = 0

    @property
    def done(self) -> bool:
        hit_eos = self.generated > 0 and self.toks[-1] in self.eos_ids
        return hit_eos or self.generated >= self.max_tokens


class ChunkedRagged(Engine):
    def __init__(self, model: Model, n_batch=8, tok_budget=512, max_len=2048):
        self.n_batch = n_batch
        self.max_len = max_len
        self.model = model
        self.waiting_queue: deque[Request] = deque()
        self.running: list[RunningReq] = []
        self.free_kv_slots = [i for i in range(n_batch)]
        self.cache = model.alloc_cache(n_batch, max_len)
        self.prefill = False
        self.tok_budget: int = tok_budget

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self) -> list[TokenEvent]:
        toks = []

        if self.waiting_queue and len(self.running) < self.n_batch:
            while self.waiting_queue and len(self.running) < self.n_batch:
                r = self.waiting_queue.popleft()
                new = RunningReq(
                    r.req_id,
                    list(r.prompt_ids),
                    r.max_tokens,
                    self.model.eos_token_ids,
                    self.free_kv_slots.pop(),
                )
                self.running.append(new)

            # token_list = [torch.tensor(x.toks) for x in to_prefill]
            # lengths = torch.tensor([len(x.toks) for x in to_prefill])
            # slots = torch.tensor([x.slot for x in to_prefill])
            #
            # token_list = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True)
            # out = self.model.forward(token_list, lengths, self.cache, slots)
            # next_toks = out.argmax(-1).tolist()
            # next_toks = zip(to_prefill, next_toks)

        # one forward now
        to_process = [
            (r, 1) for r in self.running if (len(r.toks) - r.num_computed) == 1
        ]
        prefill_list = deque(
            [r for r in self.running if (len(r.toks) - r.num_computed) > 1]
        )
        budget = self.tok_budget - len(to_process)
        while prefill_list and budget > 0:
            req = prefill_list.popleft()
            wants = len(req.toks) - req.num_computed
            b = min(budget, wants)
            to_process.append((req, b))
            budget -= b

        if to_process:
            # prefill forward
            prefill_token_list = [
                torch.tensor(x.toks[x.num_computed : x.num_computed + b])
                for (x, b) in to_process
            ]

            lengths = torch.tensor([b for (x, b) in to_process])
            slots = torch.tensor([x.slot for (x, b) in to_process])
            prefill_token_list = torch.cat(prefill_token_list, dim=0)

            # prefill_token_list = torch.nn.utils.rnn.pad_sequence(
            #     prefill_token_list, batch_first=True
            # )
            starts = torch.tensor([x.num_computed for (x, b) in to_process])
            positions = torch.cat(
                [
                    torch.arange(start, start + ln)
                    for (start, ln) in zip(starts, lengths)
                ]
            )
            prefill_out = self.model.forward(
                prefill_token_list, BatchMeta(lengths, slots, positions), self.cache
            )
            prefill_next_toks = prefill_out.argmax(-1).tolist()

            for (req, b), t in zip(to_process, prefill_next_toks):
                if req.num_computed + b == len(req.toks):
                    req.toks.append(t)
                    req.generated += 1
                    toks.append(TokenEvent(req.req_id, t, req.done))

                req.num_computed += b

                if req.done:
                    self.free_kv_slots.append(req.slot)

        self.running = [r for r in self.running if not r.done]

        return toks

    def has_unfinished(self):
        return self.running_queue_in_progress() or bool(self.waiting_queue)

    def running_queue_in_progress(self):
        return any(not r.done for r in self.running)
