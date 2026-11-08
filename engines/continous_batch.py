from collections import deque
from model.kv_cache import KVCache
from dataclasses import dataclass

import torch

from bench.engine import Engine
from bench.types import Request, TokenEvent
from model.kv_cache import KVCache
from model.model import Model


@dataclass
class RunningReq:
    req_id: int
    toks: list[int]
    max_tokens: int
    eos_ids: frozenset[int]
    slot: int
    generated: int = 0

    @property
    def done(self) -> bool:
        hit_eos = self.generated > 0 and self.toks[-1] in self.eos_ids
        return hit_eos or self.generated >= self.max_tokens


class ContinousBatching(Engine):
    def __init__(self, model: Model, n_batch=8, max_len=2048):
        self.n_batch = n_batch
        self.max_len = max_len
        self.model = model
        self.waiting_queue: deque[Request] = deque()
        self.running: list[RunningReq] = []
        self.free_kv_slots = [i for i in range(n_batch)]
        self.cache = model.alloc_cache(n_batch, max_len)
        self.prefill = False

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self) -> list[TokenEvent]:
        next_toks = []

        if self.waiting_queue and len(self.running) < self.n_batch:
            to_prefill = []

            while self.waiting_queue and len(self.running) < self.n_batch:
                r = self.waiting_queue.popleft()
                new = RunningReq(
                    r.req_id,
                    list(r.prompt_ids),
                    r.max_tokens,
                    self.model.eos_token_ids,
                    self.free_kv_slots.pop(),
                )
                to_prefill.append(new)
                self.running.append(new)

            token_list = [torch.tensor(x.toks) for x in to_prefill]
            lengths = torch.tensor([len(x.toks) for x in to_prefill])
            slots = torch.tensor([x.slot for x in to_prefill])

            token_list = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True)
            out = self.model.forward(token_list, lengths, self.cache, slots)
            next_toks = out.argmax(-1).tolist()
            next_toks = zip(to_prefill, next_toks)
        else:
            token_list = torch.tensor([x.toks[-1] for x in self.running])[:, None]
            lengths = torch.ones(len(self.running), dtype=torch.long)
            slots = torch.tensor([x.slot for x in self.running])

            out = self.model.forward(token_list, lengths, self.cache, slots)
            next_toks = out.argmax(-1).tolist()
            next_toks = zip(self.running, next_toks)

        toks = []
        for req, t in next_toks:
            req.toks.append(t)
            req.generated += 1

            if req.done:
                self.free_kv_slots.append(req.slot)
                self.cache.seq_len[req.slot] = 0
            toks.append(TokenEvent(req.req_id, t, req.done))

        self.running = [r for r in self.running if not r.done]

        return toks

    def has_unfinished(self):
        return self.running_queue_in_progress() or bool(self.waiting_queue)

    def running_queue_in_progress(self):
        return any(not r.done for r in self.running)
