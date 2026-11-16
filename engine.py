from math import ceil
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
    pages_idx: list[int]
    generated: int = 0
    num_computed: int = 0

    @property
    def done(self) -> bool:
        hit_eos = self.generated > 0 and self.toks[-1] in self.eos_ids
        return hit_eos or self.generated >= self.max_tokens


class ChunkedRaggedPaged(Engine):
    def __init__(self, model: Model, tok_budget=512, num_pages=2048, page_size=16):
        self.num_pages = num_pages
        self.page_size = page_size
        self.model = model
        self.waiting_queue: deque[Request] = deque()
        self.running: list[RunningReq] = []
        self.free_kv_slots = deque(range(num_pages))
        self.cache = model.alloc_cache(num_pages, page_size)
        self.prefill = False
        self.tok_budget: int = tok_budget

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self) -> list[TokenEvent]:
        toks = []

        while self.waiting_queue:
            r = self.waiting_queue[0]
            # check if it has space even
            num_pages_needed = ceil((len(r.prompt_ids)) / self.page_size)
            # keep one free page per running request, so each can cross its next
            # page boundary before anyone needs preempting (stops a just-preempted
            # request from being readmitted straight into the same shortage)
            reserve = len(self.running)
            if num_pages_needed + reserve <= len(self.free_kv_slots):
                # admit request
                new = RunningReq(
                    r.req_id,
                    list(r.prompt_ids),
                    r.max_tokens,
                    self.model.eos_token_ids,
                    [self.free_kv_slots.popleft() for _ in range(num_pages_needed)],
                )
                self.running.append(new)
                self.waiting_queue.popleft()
            else:
                break

            # token_list = [torch.tensor(x.toks) for x in to_prefill]
            # lengths = torch.tensor([len(x.toks) for x in to_prefill])
            # slots = torch.tensor([x.slot for x in to_prefill])
            #
            # token_list = torch.nn.utils.rnn.pad_sequence(token_list, batch_first=True)
            # out = self.model.forward(token_list, lengths, self.cache, slots)
            # next_toks = out.argmax(-1).tolist()
            # next_toks = zip(to_prefill, next_toks)

        # one forward now
        to_process = []
        preempted = set()

        # process all the decodes

        for r in self.running:
            if r.req_id in preempted:
                break
            # decode
            if (len(r.toks) - r.num_computed) == 1:
                # needs new page?
                if len(r.toks) > self.page_size * len(r.pages_idx):
                    # do we have a apge to give?
                    if self.free_kv_slots:
                        r.pages_idx.append(self.free_kv_slots.popleft())
                    else:
                        # preempt

                        # find the last
                        idx = -1
                        while self.running[idx].req_id in preempted:
                            idx -= 1
                        to_preempt = self.running[idx]

                        self.free_kv_slots.extend(to_preempt.pages_idx)
                        self.waiting_queue.appendleft(
                            Request(
                                to_preempt.req_id,
                                to_preempt.toks,
                                to_preempt.max_tokens - to_preempt.generated,
                                -1,
                            )
                        )
                        preempted.add(to_preempt.req_id)
                        if to_preempt.req_id == r.req_id:
                            # self preemption
                            break
                        else:
                            r.pages_idx.append(self.free_kv_slots.popleft())
                to_process.append((r, 1))

        budget = self.tok_budget - len(to_process)
        for r in self.running:
            if r.req_id in preempted:
                break
            if (len(r.toks) - r.num_computed) <= 1:
                continue
            # deocdes only, they should have enough kv cache allocated
            if budget > 0:
                wants = len(r.toks) - r.num_computed
                b = min(budget, wants)
                to_process.append((r, b))
                budget -= b

        self.running = [r for r in self.running if r.req_id not in preempted]

        if to_process:
            # prefill forward
            prefill_token_list = [
                torch.tensor(x.toks[x.num_computed : x.num_computed + b])
                for (x, b) in to_process
            ]

            lengths = torch.tensor([b for (x, b) in to_process])
            # slots = torch.tensor([x.slot for (x, b) in to_process])
            paged_kv_indptr = torch.nn.functional.pad(
                torch.tensor(
                    [
                        len(x.pages_idx[: ceil((x.num_computed + b) / self.page_size)])
                        for (x, b) in to_process
                    ]
                ).cumsum(0),
                (1, 0),
            ).int()
            paged_kv_indices = torch.cat(
                [
                    torch.tensor(
                        x.pages_idx[: ceil((x.num_computed + b) / self.page_size)]
                    )
                    for (x, b) in to_process
                ]
            ).int()

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
            slot_mappings = torch.cat(
                [
                    torch.tensor(
                        [
                            x.pages_idx[(x.num_computed + i) // self.page_size]
                            * self.page_size
                            + (x.num_computed + i) % self.page_size
                            for i in range(b)
                        ]
                    )
                    for (x, b) in to_process
                ]
            )
            indptr = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).int()
            last_page_lens = (positions[indptr[1:] - 1] % self.page_size + 1).int()

            prefill_out = self.model.forward(
                prefill_token_list,
                BatchMeta(
                    lengths=lengths,
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    slot_mappings=slot_mappings,
                    positions=positions,
                    indptr=indptr,
                    last_page_lens=last_page_lens,
                    page_size=self.page_size,
                ),
                self.cache,
            )
            prefill_next_toks = prefill_out.argmax(-1).tolist()

            for (req, b), t in zip(to_process, prefill_next_toks):
                if req.num_computed + b == len(req.toks):
                    req.toks.append(t)
                    req.generated += 1
                    toks.append(TokenEvent(req.req_id, t, req.done))

                req.num_computed += b

                if req.done:
                    self.free_kv_slots.extend(req.pages_idx)

        self.running = [r for r in self.running if not r.done]

        return toks

    def has_unfinished(self):
        return self.running_queue_in_progress() or bool(self.waiting_queue)

    def running_queue_in_progress(self):
        return any(not r.done for r in self.running)
