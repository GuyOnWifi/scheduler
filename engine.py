from collections import OrderedDict
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
class Block:
    hash: int
    block_idx: int
    ref_count: int


@dataclass
class RunningReq:
    req_id: int
    toks: list[int]
    max_tokens: int
    eos_ids: frozenset[int]
    pages: list[Block]
    generated: int = 0
    num_computed: int = 0
    num_scheduled: int = 0

    @property
    def done(self) -> bool:
        hit_eos = self.generated > 0 and self.toks[-1] in self.eos_ids
        return hit_eos or self.generated >= self.max_tokens


class ChunkedRaggedPaged(Engine):
    def __init__(
        self,
        model: Model,
        tok_budget=512,
        num_pages=2048,
        page_size=16,
        max_seq_len=4096,
        graph_buckets=(1, 2, 4, 8, 16, 32, 64),
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self.model = model
        self.waiting_queue: deque[Request] = deque()
        self.running: list[RunningReq] = []
        self.free_kv_slots = OrderedDict(
            # page 0 is the dummy page the CUDA graphs' padding rows write into
            {i: Block(None, i, 0) for i in range(1, num_pages)}
        )
        self.cache = model.alloc_cache(num_pages, page_size)
        self.device = self.cache.kv.device
        # after alloc_cache: the graphs bake in this cache's address
        model.build_graphs(graph_buckets, page_size, self.cache, max_seq_len)
        self.prefill = False
        self.tok_budget: int = tok_budget
        self.hash_to_blocks = {}
        self.in_flight = None

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def schedule(self):
        # admission
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

                pages = []
                num_computed = 0

                # greedy match
                prev_hash = None
                for i in range((len(r.prompt_ids) - 1) // self.page_size):
                    t = r.prompt_ids[i * self.page_size : (i + 1) * self.page_size]
                    h = hash((prev_hash, tuple(t)))
                    prev_hash = h
                    if h in self.hash_to_blocks:
                        b = self.hash_to_blocks[h]
                        pages.append(b)
                        if b.ref_count == 0:
                            del self.free_kv_slots[b.block_idx]
                        num_pages_needed -= 1
                        num_computed += self.page_size
                        b.ref_count += 1

                    else:
                        break

                for _ in range(num_pages_needed):
                    _, blk = self.free_kv_slots.popitem(last=False)
                    # only drop the map entry if it points to this block (a duplicate copy
                    # of the same prefix shares the hash but isn't the registered one)
                    if self.hash_to_blocks.get(blk.hash) is blk:
                        del self.hash_to_blocks[blk.hash]
                    blk.hash = None
                    blk.ref_count += 1
                    pages.append(blk)

                new = RunningReq(
                    r.req_id,
                    list(r.prompt_ids),
                    r.max_tokens,
                    self.model.eos_token_ids,
                    pages,
                    0,
                    num_computed,
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
            if (len(r.toks) - r.num_computed - r.num_scheduled) <= 1:
                # needs new page?
                if r.num_computed + r.num_scheduled + 1 > self.page_size * len(r.pages):
                    # do we have a apge to give?
                    if self.free_kv_slots:
                        _, blk = self.free_kv_slots.popitem(last=False)
                        # only drop the map entry if it points to this block (a duplicate copy
                        # of the same prefix shares the hash but isn't the registered one)
                        if self.hash_to_blocks.get(blk.hash) is blk:
                            del self.hash_to_blocks[blk.hash]
                        blk.hash = None
                        blk.ref_count += 1
                        r.pages.append(blk)
                    else:
                        # preempt

                        # find the last
                        idx = -1
                        while self.running[idx].req_id in preempted:
                            idx -= 1
                        to_preempt = self.running[idx]

                        for blk in reversed(to_preempt.pages):
                            blk.ref_count -= 1
                            if blk.ref_count == 0:
                                self.free_kv_slots[blk.block_idx] = blk

                        self.waiting_queue.appendleft(
                            Request(
                                to_preempt.req_id,
                                list(
                                    to_preempt.toks
                                ),  # copy: a late process() must not grow the queued prompt
                                to_preempt.max_tokens - to_preempt.generated,
                                -1,
                            )
                        )
                        preempted.add(to_preempt.req_id)
                        if to_preempt.req_id == r.req_id:
                            # self preemption
                            break
                        else:
                            _, blk = self.free_kv_slots.popitem(last=False)
                            # only drop the map entry if it points to this block (a duplicate copy
                            # of the same prefix shares the hash but isn't the registered one)
                            if self.hash_to_blocks.get(blk.hash) is blk:
                                del self.hash_to_blocks[blk.hash]
                            blk.hash = None
                            blk.ref_count += 1
                            r.pages.append(blk)
                to_process.append((r, 1))
                r.num_scheduled += 1

        budget = self.tok_budget - len(to_process)
        for r in self.running:
            if r.req_id in preempted:
                break
            if (len(r.toks) - r.num_computed - r.num_scheduled) <= 1:
                continue
            # deocdes only, they should have enough kv cache allocated
            if budget > 0:
                wants = len(r.toks) - r.num_computed - r.num_scheduled
                b = min(budget, wants)
                to_process.append((r, b))
                r.num_scheduled += b
                budget -= b

        self.running = [r for r in self.running if r.req_id not in preempted]

        row_of = {}
        if self.in_flight is not None:
            prev_batch = self.in_flight[0]  # to_process
            row_of = {r.req_id: j for j, (r, b) in enumerate(prev_batch)}

        if to_process:
            # prefill forward

            ids, dst, src = [], [], []
            for x, b in to_process:
                start = x.num_computed + x.num_scheduled - b
                if start < len(x.toks):
                    ids.extend(x.toks[start : start + b])  # known, from the CPU
                else:
                    dst.append(len(ids))  # flat position in the packed input
                    src.append(row_of[x.req_id])  # r's row in N
                    ids.append(0)  # placeholder

            # pinned + non_blocking: a blocking copy would wait for step N to finish
            ids_gpu = torch.tensor(ids).pin_memory().to(self.device, non_blocking=True)
            if dst:
                dst_gpu = (
                    torch.tensor(dst).pin_memory().to(self.device, non_blocking=True)
                )
                src_gpu = (
                    torch.tensor(src).pin_memory().to(self.device, non_blocking=True)
                )
                ids_gpu[dst_gpu] = self.in_flight[1][src_gpu]

            lengths = torch.tensor([b for (x, b) in to_process])
            # slots = torch.tensor([x.slot for (x, b) in to_process])
            paged_kv_indptr = torch.nn.functional.pad(
                torch.tensor(
                    [
                        len(
                            x.pages[
                                : ceil(
                                    (x.num_computed + x.num_scheduled) / self.page_size
                                )
                            ]
                        )
                        for (x, b) in to_process
                    ]
                ).cumsum(0),
                (1, 0),
            ).int()
            paged_kv_indices = torch.cat(
                [
                    torch.tensor(
                        [
                            blk.block_idx
                            for blk in x.pages[
                                : ceil(
                                    (x.num_computed + x.num_scheduled) / self.page_size
                                )
                            ]
                        ]
                    )
                    for (x, b) in to_process
                ]
            ).int()

            # prefill_token_list = torch.nn.utils.rnn.pad_sequence(
            #     prefill_token_list, batch_first=True
            # )
            starts = torch.tensor(
                [x.num_computed + x.num_scheduled - b for (x, b) in to_process]
            )
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
                            x.pages[
                                (x.num_computed + x.num_scheduled - b + i)
                                // self.page_size
                            ].block_idx
                            * self.page_size
                            + (x.num_computed + x.num_scheduled - b + i)
                            % self.page_size
                            for i in range(b)
                        ]
                    )
                    for (x, b) in to_process
                ]
            )
            indptr = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).int()
            last_page_lens = (positions[indptr[1:] - 1] % self.page_size + 1).int()

            prefill_out = self.model.forward(
                ids_gpu,
                BatchMeta(
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

            next_gpu = prefill_out.argmax(-1)  # stays on the GPU: N+1's decode inputs
            next_cpu = next_gpu.to("cpu", non_blocking=True)  # pinned, filled later
            done = torch.cuda.Event()
            done.record()  # fires once N's forward + argmax + copy have run
            return to_process, next_gpu, next_cpu, done

    def process(self, to_process, next_gpu, next_cpu, done):
        toks = []
        done.synchronize()  # waits for N's copy only, not for N+1
        prefill_next_toks = next_cpu.tolist()

        # finished in the previous process() or preempted since: drop the stale row
        alive = {r.req_id for r in self.running}
        for (req, b), t in zip(to_process, prefill_next_toks):
            if req.req_id not in alive:
                continue

            if req.num_computed + b == len(req.toks):
                req.toks.append(t)
                req.generated += 1
                toks.append(TokenEvent(req.req_id, t, req.done))

            # hash the blocks
            for i in range(
                req.num_computed // self.page_size,
                (req.num_computed + b) // self.page_size,
            ):
                # starting block
                prev_hash = None
                if i > 0:
                    prev_hash = req.pages[i - 1].hash

                h = hash(
                    (
                        prev_hash,
                        tuple(req.toks[i * self.page_size : (i + 1) * self.page_size]),
                    )
                )
                req.pages[i].hash = h

                # keep the first block registered for h; a later duplicate stays private
                self.hash_to_blocks.setdefault(h, req.pages[i])

            req.num_computed += b
            req.num_scheduled -= b

            if req.done:
                for blk in reversed(req.pages):
                    blk.ref_count -= 1
                    if blk.ref_count == 0:
                        self.free_kv_slots[blk.block_idx] = blk

        self.running = [r for r in self.running if not r.done]
        return toks

    def step(self) -> list[TokenEvent]:
        new = self.schedule()

        out = []
        if self.in_flight is not None:
            out = self.process(*self.in_flight)

        self.in_flight = new

        return out

    def has_unfinished(self):
        return (
            self.running_queue_in_progress()
            or bool(self.waiting_queue)
            or self.in_flight is not None
        )

    def running_queue_in_progress(self):
        return any(not r.done for r in self.running)
