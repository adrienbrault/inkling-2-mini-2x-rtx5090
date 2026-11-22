# Inkling-2-Mini on two RTX 5090s

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 26B total / 3.4B active
looped MoE, text and image in) served with vLLM on two RTX 5090 (32 GB, PCIe, no NVLink): both cards as one model
(tensor parallel 2), the checkpoint's own DSpark drafter, and the full 262,144-token context. A single 5090 cannot hold
the drafter and a useful context; two give an 818,388-token KV pool with an fp8 cache.

vLLM's day-0 image for this model does not run its drafter on consumer Blackwell (sm_120). The overlay in
[`docker/`](docker/README.md) makes it run and adds five changes, none of which changes the output of a request served alone (bitwise). Two products in
the decode step move off cuBLAS onto Triton kernels. A scheduler budget lets long prefills interleave with decoding
streams. Sampling, for the model and for the drafter, is sharded across the two cards. The short-conv state is stored
in pages of its own size, which enlarges the KV pool by 17 %. Everything else is the official checkpoint and vLLM's
recipe flags.

## Numbers

Measured 2026-11-22 on one boot of the served configuration (`scripts/serve.sh`, image `sm120-r10` from
`scripts/build-image.sh`), stock power limits. **Decode rate** is one stream's output tokens per second from its first
to its last token, from the server's per-chunk token counts. **Aggregate** is the total decode rate of the streams
running together. Time to first token (TTFT) is separate.

![Decode: aggregate and per-stream rate at 1, 4 and 8 streams for code at depth, Spec-Bench, and Spec-Bench without the drafter](docs/img/decode.svg)

Blue: code with a 2K append, averaged over the 1K / 32K / 64K cached depths. Brown: Spec-Bench. Grey dashed: Spec-Bench
without the drafter (an earlier image with NCCL all-reduce, [docs/CONFIGURATION.md](docs/CONFIGURATION.md)). Drawn by
`bench/plot.py`.

**Spec-Bench** (120-prompt stratified subset per concurrency, temperature 0, 256 forced output tokens, `vllm bench serve`;
[results](bench/results/20261122T053112Z-inkling2-specbench-served-r10), `bench/specbench.sh`,
`bench/specbench_summary.py`):

| Streams | Decode rate per stream (median) | Aggregate | TTFT (median) | Drafter acceptance length |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 376 tok/s | 413 tok/s | 32 ms | 2.97 |
| 4 | 242 tok/s | 1,018 tok/s | 54 ms | 2.98 |
| 8 | 188 tok/s | 1,588 tok/s | 67 ms | 3.00 |

Here the aggregate is the stream count times the mean request decode rate (the benchmark keeps every stream busy).

**Code at depth** (a shared code context cached at 1K / 32K / 64K tokens, a fresh tool-output append and a HumanEval task
per stream, 400 forced output tokens, temperature 0, no thinking; the cached context is verified as a cache hit on every
stream; [results](bench/results/20261122T053347Z-inkling2-code-r10), `bench/headline_inkling.py code`). Aggregate = tokens
decoded by all streams during the window in which every stream is decoding, divided by its length:

| Cached context | Append | 1 stream | 4 streams: per stream / aggregate | 8 streams: per stream / aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 2K | 423 tok/s | 265 / 1,134 tok/s | 207 / 1,724 tok/s |
| 32K | 2K | 446 tok/s | 253 / 1,079 tok/s | 181 / 1,619 tok/s |
| 64K | 2K | 451 tok/s | 220 / 918 tok/s | 174 / 1,557 tok/s |
| 1K | 8K | 559 tok/s | 213 / 1,005 tok/s | 144 / 1,681 tok/s |
| 32K | 8K | 476 tok/s | 198 / 1,095 tok/s | 126 / 1,606 tok/s |
| 64K | 8K | 619 tok/s | 181 / 974 tok/s | 108 / 1,420 tok/s |

One stream's rate varies with the task and the appended code: each cell is one request, and the drafter accepts more
on predictable code, so the six one-stream cells range from 423 to 619 tok/s on this boot. Median TTFT is 0.13–0.78 s
for one stream. With eight streams, 64K context and 8K appends it reaches 3.6 s, because the prefills queue together
behind the decoding streams.

**Cold prefill** (fresh salted prompts with no cached prefix, verified, one at a time; end to end: prompt tokens divided
by the client's TTFT; [results](bench/results/20261122T053516Z-inkling2-prefill-r10a), `bench/headline_inkling.py prefill`):

| Prompt tokens | 2,045 | 8,187 | 30,715 | 61,435 | 122,876 | 204,797 | 245,758 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 0.11 s | 0.45 s | 1.85 s | 4.18 s | 10.3 s | 21.5 s | 28.3 s |
| Prefill rate | 18,712 | 18,345 | 16,626 | 14,693 | 11,947 | 9,544 | 8,675 tok/s |

![Cold prefill rate and time to first token versus prompt length](docs/img/prefill.svg)

A second pass on the same boot is within 2 % on every cell ([results](bench/results/20261122T053630Z-inkling2-prefill-r10b)).

**Tool calling** (same boot; tool-eval-bench 2.1.0, 69 scenarios, 4 trials, temperature 1.0 (the checkpoint's
default), 8 in parallel; [results](bench/results/20261122T053020Z-inkling2-quality-r10)): final score 83 / 100, mean
over the trials 82.5 (standard deviation 1.9).

## Served configuration

```bash
bash scripts/build-image.sh
MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh     # OpenAI-compatible API on :8000, model "inkling-2-mini"
```

| Setting | Value |
| --- | --- |
| Checkpoint | `thinkingmachines/Inkling-2-Mini`, official NVFP4/MXFP8 (19.5 GB) + DSpark drafter `mtp.safetensors` (636 MB) |
| Engine | `vllm/vllm-openai:inkling-2-mini` (vLLM `0.1.dev25855+g7cb5dae22`, pinned by digest) + the `docker/` overlay (`inkling2-rtx5090:sm120-r10`) |
| Parallelism | tensor parallel 2; all-reduce = FlashInfer PCIe IPC (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`), NCCL fallback; vLLM's custom all-reduce off |
| Drafter | DSpark, 7 draft tokens per step (the only value the checkpoint accepts); its KV cache in bf16; draft sampling sharded across the cards (`VLLM_INKLING_DRAFT_SHARD=1`, overlay) |
| Context | 262,144 tokens; KV pool 818,388 tokens (`--kv-cache-dtype fp8`, unit-scale E4M3; short-conv state in its own pages, `VLLM_INKLING_CONV_PAGE=1 --block-size 64`, overlay), `--gpu-memory-utilization 0.92`, 16 sequences |
| Scheduler | `VLLM_DECODE_STEP_BUDGET=1024` (overlay): at most 1,024 tokens per step while a request is decoding; 2,048 otherwise |
| Sampling | `--enable-batch-sharded-sampling` (overlay): each request sampled on one card from both cards' vocabulary shards |
| API | `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice`; `reasoning_effort` none … max |
| Warmup | FlashInfer autotune off (`--kernel-config`) |
| Usage | `--enable-prompt-tokens-details` (cached prompt tokens in `usage`) |

## What the stack is

- **Model**: 24 decoder layers whose middle 12 run three times with shared weights (48 layer applications per token),
  128-expert MoE with 8 active, mostly sliding-window attention with one full-attention layer in six. Experts are NVFP4
  (layers 0–19) and MXFP8 (layers 20–23).
- **Drafter**: DSpark, a 3-layer block drafter that reads target hidden states and proposes 7 tokens per step.
- **Overlay**: the day-0 image's compiled core library has no sm_120 code. Inkling's own kernels (Triton/Gluon) run
  on the 5090. The overlay replaces the two compiled kernels the drafter and the tensor-parallel embedding reach. It
  also adds the decode, scheduler, sampling and cache-layout changes above. Details and the guard checks:
  [`docker/README.md`](docker/README.md).

Alternatives that were measured (no drafter, one model copy per card, bf16 KV, scheduler and sampling settings) and the
output checks are in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).
