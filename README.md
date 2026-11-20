# scheduler

An LLM inference engine for Qwen3-0.6B, built from scratch one optimization at a time.

Writeup: [Building an LLM Inference Engine](https://easonhuang.dev/blog/llm_inference_engine)

Each rung is a git tag, so you can browse the code at any step or diff it against the previous one:

| rung | tag |
|---|---|
| 00 Qwen3 implementation | [`rung-00-qwen`](https://github.com/GuyOnWifi/scheduler/tree/rung-00-qwen) |
| 01 Static batching | [`rung-01-naive`](https://github.com/GuyOnWifi/scheduler/tree/rung-01-naive) |
| 02 KV caching | [`rung-02-kv-cache`](https://github.com/GuyOnWifi/scheduler/tree/rung-02-kv-cache) |
| 03 Continuous batching | [`rung-03-continuous`](https://github.com/GuyOnWifi/scheduler/tree/rung-03-continuous) |
| 04 Chunked prefill | [`rung-04-chunked`](https://github.com/GuyOnWifi/scheduler/tree/rung-04-chunked) |
| 05 Flash attention + ragged tensors | [`rung-05-flash-attn`](https://github.com/GuyOnWifi/scheduler/tree/rung-05-flash-attn) |
| 06 Paged KV cache | [`rung-06-paged`](https://github.com/GuyOnWifi/scheduler/tree/rung-06-paged) |
| 07 Prefix caching | [`rung-07-prefix`](https://github.com/GuyOnWifi/scheduler/tree/rung-07-prefix) |
| 08 CUDA graphs | [`rung-08-graphs`](https://github.com/GuyOnWifi/scheduler/tree/rung-08-graphs) |
| 09 Overlap scheduling | [`rung-09-overlap`](https://github.com/GuyOnWifi/scheduler/tree/rung-09-overlap) |

## Results & Benchmarking

We're benchmarking a Qwen3-0.6b in bf16 on an RTX 3060 12GB. We check for correctness against HuggingFace's implementation.

We collect the following metrics:
- **TTFT** - time to first token (how long it takes for the first generation)
- **TPOT** - time per output token
- **ITL** - inter-token latency (the time between tokens)
- **E2E** - the whole generation, end to end

We record the p50/p90/p99 of it, because different optimizations will affect different percentiles.

To find this, we run 3 benchmarks:
- **serve**, which simulates users arriving at 1.8 req/s (Poisson). **This is the main benchmark**, as it simulates real-world behavior and tests the engine under uneven arrivals
- **latency**, which times how long a step of the engine takes. This is used to find any kernel / engine-side improvements
- **throughput**, which sends a bunch of requests all at once and sees the TPS that the engine achieves.

We mainly focus on the **serve** benchmark, as it is the most relevant to the real world.

We always do a warmup before benchmarking, for all of the JIT code to compile, kernels to be selected, etc.

**Serve, random prompts at 1.8 req/s, plus max TPS** (ms, p50 / p90 / p99)

| rung          | TTFT               | TPOT               | ITL                | E2E                | max TPS |
| ------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------- |
| 01 naive      | (too slow to count) | (too slow to count) | (too slow to count) | (too slow to count) | (too slow to count) |
| 02 kv cache   | 2241 / 4767 / 7589 | 18.1 / 21.5 / 29.1 | 18.1 / 21.5 / 22.8 | 3740 / 7422 / 9642 | 150     |
| 03 continuous | 57.2 / 97.6 / 184  | 17.9 / 23.9 / 28.9 | 16.1 / 24.4 / 75.9 | 1622 / 3852 / 4640 | 224     |
| 04 chunked    | 75.1 / 143 / 191   | 18.1 / 24.2 / 29.4 | 16.4 / 25.4 / 65.7 | 1616 / 3863 / 4582 | 222     |
| 05 flash-attn | 42.1 / 61.1 / 72.6 | 10.6 / 11.3 / 12.1 | 10.2 / 11.6 / 26.4 | 884 / 2169 / 2454  | 577     |
| 06 paged      | 41.0 / 59.7 / 70.2 | 9.6 / 10.8 / 12.0  | 9.0 / 10.7 / 23.1  | 808 / 1924 / 2144  | 1315    |
| 07 prefix     | 41.4 / 59.9 / 70.7 | 9.6 / 10.3 / 11.2  | 9.2 / 10.7 / 23.3  | 858 / 1974 / 2232  | 1296    |
| 08 graphs     | 40.4 / 56.9 / 66.7 | 6.7 / 7.4 / 8.7    | 6.5 / 6.9 / 17.1   | 588 / 1403 / 1635  | 1380    |
| 09 overlap    | 39.1 / 61.0 / 75.8 | 6.1 / 6.8 / 7.2    | 6.0 / 6.3 / 16.3   | 550 / 1299 / 1506  | 1451    |

**Serve, shared prefix (384 shared + ~128 own tokens) at 8 req/s** (ms, p50 / p90 / p99)

| rung                       | TTFT               | TPOT               | ITL               | E2E               | out tok/s |
| -------------------------- | ------------------ | ------------------ | ----------------- | ----------------- | --------- |
| 06 paged (no prefix cache) | 47.0 / 70.8 / 97.9 | 11.6 / 13.7 / 31.7 | 9.9 / 12.4 / 32.5 | 926 / 2219 / 2796 | 601       |
| 07 prefix                  | 17.8 / 23.4 / 30.8 | 9.2 / 10.1 / 11.9  | 8.7 / 10.5 / 16.3 | 757 / 1709 / 2090 | 619       |
| 08 graphs                  | 17.0 / 21.4 / 28.2 | 7.2 / 7.9 / 9.2    | 6.9 / 7.6 / 16.2  | 607 / 1352 / 1689 | 636       |
| 09 overlap                 | 22.4 / 27.0 / 33.8 | 6.6 / 7.1 / 8.3    | 6.4 / 7.1 / 12.3  | 538 / 1236 / 1522 | 633       |

**Latency, closed loop: batch 8, 128 in / 64 out** (ms, p50 / p90 / p99)

| rung | decode step | prefill | batch |
|---|---|---|---|
| 01 naive | 83.6 / 99.9 / 103 | 66.0 / 66.1 / 66.1 | 5482 / 5483 / 5484 |
| 02 kv cache | 11.2 / 12.3 / 12.8 | 68.0 / 68.9 / 69.1 | 793 / 796 / 797 |
| 03 continuous | 12.3 / 13.0 / 13.9 | 70.1 / 70.1 / 70.1 | 846 / 847 / 847 |
| 04 chunked | 12.8 / 13.5 / 49.0 | 38.6 / 38.6 / 38.6 | 910 / 920 / 922 |
| 05 flash-attn | 9.6 / 10.8 / 30.9 | 30.6 / 30.7 / 30.8 | 684 / 712 / 718 |
| 06 paged | 9.1 / 10.6 / 29.9 | 29.4 / 29.5 / 29.5 | 661 / 664 / 665 |
| 07 prefix | 9.4 / 10.4 / 11.2 | 11.0 / 11.1 / 11.1\* | 606 / 615 / 617 |
| 08 graphs | 6.7 / 6.9 / 7.1 | 12.0 / 12.5 / 12.6 | 435 / 435 / 436 |
| 09 overlap | 6.1 / 6.2 / 6.3 | 9.3 / 9.7 / 9.8 | 405 / 405 / 405 |

\*Prefix cache hits: the bench reruns the same prompts.
