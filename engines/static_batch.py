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
    generated: int = 0

    @property
    def done(self) -> bool:
        hit_eos = self.generated > 0 and self.toks[-1] in self.eos_ids
        return hit_eos or self.generated >= self.max_tokens


class StaticBatching(Engine):
    def __init__(self, model: Model, n_batch=8, max_len=2048):
        self.n_batch = n_batch
        self.max_len = max_len
        self.model = model
        self.waiting_queue: list[Request] = []
        self.running: list[RunningReq] = []
        self.cache = model.alloc_cache(n_batch, max_len)
        self.batch_cache: KVCache | None = None
        self.prefill = False

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self) -> list[TokenEvent]:
        if not self.running_queue_in_progress():
            batch = self.waiting_queue[: self.n_batch]
            self.waiting_queue = self.waiting_queue[self.n_batch :]
            self.running = [
                RunningReq(
                    r.req_id, list(r.prompt_ids), r.max_tokens, self.model.eos_token_ids
                )
                for r in batch
            ]
            for r in batch:
                if len(r.prompt_ids) + r.max_tokens > self.max_len:
                    raise ValueError(
                        f"request {r.req_id} needs more than max_len={self.max_len} tokens"
                    )
            nb = len(batch)
            self.batch_cache = KVCache(
                self.cache.k[:, :nb], self.cache.v[:, :nb], self.cache.seq_len[:nb]
            )
            self.batch_cache.seq_len.zero_()
            self.prefill = True

        if not self.running:
            return []

        next_toks = []
        if self.prefill:
            token_list = [torch.tensor(x.toks) for x in self.running]
            lengths = torch.tensor([len(x.toks) for x in self.running])

            token_list = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True)
            out = self.model.forward(
                token_list, lengths, self.batch_cache, torch.arange(len(self.running))
            )
            next_toks = out.argmax(-1).tolist()

            self.prefill = False
        else:
            token_list = torch.tensor([x.toks[-1] for x in self.running])[:, None]
            lengths = torch.ones(len(self.running), dtype=torch.long)

            out = self.model.forward(
                token_list, lengths, self.batch_cache, torch.arange(len(self.running))
            )
            next_toks = out.argmax(-1).tolist()

        toks = []
        for req, t in zip(self.running, next_toks):
            if req.done:
                continue  # finished rows still ride along in the batch, but stay silent
            req.toks.append(t)
            req.generated += 1
            toks.append(TokenEvent(req.req_id, t, req.done))

        return toks

    def has_unfinished(self):
        return self.running_queue_in_progress() or bool(self.waiting_queue)

    def running_queue_in_progress(self):
        return any(not r.done for r in self.running)
