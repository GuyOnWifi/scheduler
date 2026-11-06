from dataclasses import dataclass

import torch

from bench.engine import Engine
from bench.types import Request, TokenEvent
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
    def __init__(self, model: Model, n_batch=8):
        self.n_batch = n_batch
        self.model = model
        self.waiting_queue: list[Request] = []
        self.running: list[RunningReq] = []

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

        if not self.running:
            return []

        # pack into 2d
        token_list = [torch.tensor(x.toks) for x in self.running]
        lengths = torch.tensor([len(x.toks) for x in self.running])

        token_list = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True)
        out = self.model.forward(token_list, lengths)
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
