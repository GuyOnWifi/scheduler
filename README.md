# scheduler

an llm inference scheduler built from scratch, one technique at a time. each "rung"
is one commit (`git log --oneline`) and adds one idea on top of the last.

- model: Qwen3-0.6B, own implementation (weights loaded from the HF safetensors), bf16
- hardware: one RTX 3060 (12 GB)
- stack: Python 3.14, torch, FlashInfer (attention kernels), uv

## rungs

| # | commit | idea |
|---|--------|------|
| 00 | `naive` | static batches of 8, no KV cache: every step re-runs the whole sequence |
| 01 | `kv caching` | contiguous per-request KV cache, so decode steps only process the new token |
| 02 | `continous batching` | 8 slots refilled every step as requests finish; prefill-first (a new request's prefill runs before decodes) |
| 03 | `chunked batching` | vLLM-like policy: each step runs every running decode, then fills the rest of a 512-token budget with prefill chunks |
| 04 | `flash attention` | attention through FlashInfer kernels instead of hand-written torch attention |
| 04b | `ragged tensors` | the chunked schedule as one packed (ragged) forward per step, no padding; requests back to back with `indptr` offsets |
| 05 | `paged kv cache` | shared pool of 2048 pages x 16 tokens instead of 8 fixed slots; admission by free pages (keeping one spare page per running request), newest-first preemption by recompute when pages run out |
| 06 | `prefix caching` | full pages are hashed (chained over the prefix); new requests greedily reuse matching pages, refcounted; freed pages stay cached until evicted |
| 07 | `cuda graphs` | decode-only steps replay a captured CUDA graph, batch padded up to a bucket (1, 2, 4, 8, 16, 32, 64; padding rows write to a dummy page 0); mixed prefill/decode steps fall back to eager |

## results

all numbers are single runs on the 3060, taken from the raw files in `results/<rung>/`
(gitignored; each `.txt` starts with the exact command, `.json` has args + summary).
configs changed between rungs, so each table below is one config only.

for rung 07, the runs went through a small wrapper around `bench.py` that counts
graph vs eager steps; `baseline-eager` is the same HEAD code with graph capture
disabled.

### decode step time (latency mode)

`bench.py latency --batch-size 8 --input-len 128 --output-len 64` (3 warmup + 10 timed
iterations of the same batch). decode step = one step after the first.

| rung | engine | decode step p50 (ms) | decode step mean (ms) | prefill (ms, p50) | whole batch (ms, p50) |
|------|--------|---:|---:|---:|---:|
| 00 naive | static | 82.86 | 85.22 | 65.32 | 5429 |
| 01 kv cache | static | 10.16 | 10.17 | 67.26 | 708 |
| 02 continuous | continuous | 11.19 | 11.21 | 68.57 | 775 |
| 03 chunked | chunked | 11.88 | 12.52 | 37.75 | 852 |
| 04 flashinfer | chunked | 10.24 | 11.03 | 35.06 | 755 |
| 04 flashinfer | continuous | 10.20 | 10.29 | 62.85 | 713 |
| 04b ragged | ragged | 9.14 | 9.58 | 30.05 | 650 |
| 05 paged | paged | 7.98 | 8.41 | 29.12 | 572 |
| 07 eager (HEAD, graphs off) | paged | 8.69 | 8.90 | 10.84* | 580 |
| 07 graphs | paged | 6.69 | 6.72 | 11.28* | 435 |

\* prefix caching is on from 06, and the latency bench re-runs the same batch, so
prefill mostly hits cached pages; not comparable with the prefill column above it.
06 has no latency run. 03/04/04b/05 have a p99 decode step of ~30-48 ms (a slow step
somewhere in each run); 07 graphs p99 is 7.25 ms.

batch-size-1 runs (same lengths) exist for 00-03 only: decode step p50 15.20 (00),
8.57 (01), 8.55 (02), 9.03 (03) ms.

### online serving, random dataset

`bench.py serve --num-prompts 100 --input-len 512 --output-len 128 --range-ratio 0.8
--request-rate 1.8` (Poisson arrivals, seed 0, 1 warmup request). at 1.8 req/s every
engine keeps up, so output tok/s (~129-132) is set by the arrival rate and isn't a
comparison; the latency columns are.

| rung | engine | TPOT p50 | ITL p99 | TTFT p50 | E2E p50 | out tok/s |
|------|--------|---:|---:|---:|---:|---:|
| 02 continuous | continuous | 16.44 | 70.21 | 57.34 | 1288 | 130.4 |
| 03 chunked | chunked | 16.55 | 62.36 | 67.95 | 1296 | 130.6 |
| 04 flashinfer | continuous | 10.62 | 35.00 | 41.57 | 911 | 129.0 |
| 04 flashinfer | chunked | 10.64 | 40.82 | 52.08 | 926 | 128.9 |
| 04b ragged | ragged | 9.49 | 24.33 | 42.59 | 775 | 129.2 |
| 05 paged | paged | 8.23 | 21.50 | 39.36 | 707 | 131.3 |
| 05 paged (re-run as 06 baseline) | paged | 8.37 | 21.46 | 40.55 | 733 | 132.2 |
| 06 prefix | paged | 8.30 | 21.61 | 39.78 | 726 | 131.2 |
| 07 eager (HEAD, graphs off) | paged | 9.37 | 23.03 | 41.13 | 836 | 131.0 |
| 07 graphs | paged | 6.62 | 17.13 | 39.97 | 577 | 131.8 |

all times in ms. no prefix sharing in this dataset, so 06 ~= 05 is expected. in the 07
graphs run, 6039 of 6187 steps replayed a graph (step p50 6.36 ms) and 148 ran eager
(mixed steps, p50 23.04 ms); eager baseline step p50 was 8.72 ms.

### shared-prefix dataset (06, 07)

every prompt = the same 384-token prefix + its own random suffix of 128 tokens
(+-80%), output 128 (+-80%), 200 prompts.

serve: `bench.py serve --dataset shared-prefix --prefix-len 384 --num-prompts 200
--input-len 128 --output-len 128 --range-ratio 0.8 --request-rate 8`

| rung | TPOT p50 | ITL p99 | TTFT p50 | TTFT p99 | E2E p50 | out tok/s |
|------|---:|---:|---:|---:|---:|---:|
| 05 paged (no prefix cache, `06-prefix/baseline-05`) | 10.72 | 32.07 | 44.91 | 98.83 | 854 | 606.9 |
| 06 prefix | 8.64 | 16.04 | 17.06 | 28.97 | 680 | 615.7 |
| 07 eager (graphs off) | 9.61 | 16.51 | 18.18 | 30.58 | 735 | 611.2 |
| 07 graphs | 7.22 | 16.30 | 17.04 | 28.71 | 565 | 625.6 |

throughput (all 200 requests at t=0): `bench.py throughput --dataset shared-prefix
--prefix-len 384 --num-prompts 200 --input-len 128 --output-len 128 --range-ratio 0.8`

| rung | duration (s) | req/s | out tok/s | total tok/s |
|------|---:|---:|---:|---:|
| 05 paged (no prefix cache) | 11.59 | 17.25 | 1442 | 10214 |
| 06 prefix | 5.61 | 35.66 | 3048 | 21182 |
| 07 eager (graphs off) | 5.85 | 34.18 | 2921 | 20299 |
| 07 graphs | 5.58 | 35.85 | 3064 | 21295 |

graphs barely help here: in the graphs throughput run only 136 of 284 steps were
decode-only (graph), the rest were mixed and ran eager.

### early stress runs (01 vs 02)

`bench.py serve --num-prompts 300 --input-len 512 --output-len 128 --range-ratio 0.8`
at three rates; only 01 (static + KV cache) and 02 (continuous) were run with this.

| rate | rung | TPOT p50 | TTFT p50 | E2E p50 | out tok/s |
|------|------|---:|---:|---:|---:|
| 0.9 | 01 static | 13.48 | 1083 | 2494 | 85.3 |
| 0.9 | 02 continuous | 12.59 | 56.25 | 1269 | 86.9 |
| 1.8 | 01 static | 28.25 | 28878 | 31434 | 117.9 |
| 1.8 | 02 continuous | 20.01 | 65.35 | 1904 | 169.0 |
| inf | 01 static | 28.42 | 117624 | 122051 | 117.3 |
| inf | 02 continuous | 31.92 | 55860 | 58463 | 242.8 |

at 1.8 req/s static batching can't keep up (queue grows, TTFT in tens of seconds);
`inf` is a capacity run (everything at t=0). there are also smoke runs (16 prompts,
in 128, out 32, rate 4) for 00-02 in `results/0{0,1,2}-*/serve_smoke.json`; 00 has no
other serve results.

## how to run

setup (uv; needs a CUDA GPU for FlashInfer):

```sh
uv sync
```

the model weights come from the HF hub (`Qwen/Qwen3-0.6B`) on first use.

at HEAD the model only takes a paged KV cache, so only the `paged` engine runs; the
older engines (`static`, `continuous`, `chunked`, `ragged`) still build the old
metadata. to run an older rung, check out its commit. note `bench.py`'s default
`--engine` is `static`, so pass `--engine paged`.

correctness test (engine greedy tokens vs a one-request-at-a-time reference, bf16;
a first mismatch where the reference's top-2 logits are tied within bf16 noise is
reported as `~ok`):

```sh
uv run python test_engine.py paged             # defaults
uv run python test_engine.py paged 16          # small token budget: prompts split into chunks
uv run python test_engine.py paged 16 13       # also 13 pages: forces preemption + cache eviction
```

benchmarks (`uv run python bench.py {latency,throughput,serve,all} --help`):

```sh
# latency floor: one fixed batch
uv run python bench.py latency --engine paged --batch-size 8 --input-len 128 --output-len 64

# online serving, random prompts
uv run python bench.py serve --engine paged --num-prompts 100 --input-len 512 \
  --output-len 128 --range-ratio 0.8 --request-rate 1.8 --percentiles 50,90,99 \
  --output-json results/serve.json

# shared-prefix serving / offline throughput
uv run python bench.py serve --engine paged --dataset shared-prefix --prefix-len 384 \
  --num-prompts 200 --input-len 128 --output-len 128 --range-ratio 0.8 --request-rate 8 \
  --percentiles 50,90,99
uv run python bench.py throughput --engine paged --dataset shared-prefix --prefix-len 384 \
  --num-prompts 200 --input-len 128 --output-len 128 --range-ratio 0.8
```

benchmarks force output length (`max_tokens`) and use random token ids; `serve` also
takes `--burstiness` (gamma arrivals) and `--goodput ttft:MS tpot:MS` SLOs.

## layout

```
engines/                  one engine per rung
  static_batch.py         00/01: static batches of 8
  continous_batch.py      02: continuous batching, 8 slots
  chunked.py              03/04: chunked prefill, 512-token budget
  chunked_ragged.py       04b: same schedule, packed forward
  chunked_ragged_paged.py 05+: paged KV pool, preemption, prefix cache (bench name `paged`)
model/
  qwen.py                 Qwen3 network + safetensors loading
  block.py, feedforward.py, rmsnorm.py, config.py
  attention.py            GQA with RoPE + q/k norm; writes K/V into the paged cache, FlashInfer attention
  slow_attention.py       older padded attention, kept for reference
  qwen3_lm.py             Qwen3LM: tokenizer, stop tokens, cache alloc, CUDA-graph capture/replay, eager fallback
  kv_cache.py             KVCache (contiguous, old) and PagedKVCache
  batch_meta.py           per-step packed batch description (indptr, page tables, slot mapping, positions)
  model.py                Model protocol the engines code against
bench/                    harness: datasets, arrival driver, latency/throughput/serve modes, metrics
bench.py                  benchmark CLI
test_engine.py            token-for-token check against a single-request reference
results/                  raw bench output per rung (gitignored)
```

`test.py`, `test_dontedit.py` and `static_decode.py` at the root are scratch files
(model-vs-HF check, FlashInfer notes, a CUDA-graph decode prototype).
