# Inkling-2-Mini on two RTX 5090s

![Recipe card: decode at 1 and 8 agents, cold prefill, KV pool and memory on two RTX 5090s, 2026-11-22](docs/img/recipe-card-2b26f1e5.png)

Serving configuration, vLLM overlay, launch scripts and measurements for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on two RTX 5090 cards.

**Goal**: run many coding agents from one desktop box. Several agent sessions share the server, their prompts grow with every tool call up to the 262K window and mostly hit the prefix cache, and new prompts arrive while the others decode. The 16 sequences, the KV pool, the scheduler step budget and the benchmarks below follow from that workload.

- **Model**: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines; 26B total, 3.4B active; text and images in), the official checkpoint and its own DSpark drafter (7 draft tokens).
- **Engine**: vLLM's Inkling day-0 image plus a small overlay ([`docker/`](docker/README.md)) that makes it run on the RTX 5090 and speeds up serving. A request served alone gets bitwise the same output with or without the overlay.
- **Window and KV pool**: 262,144 tokens per request; an 818,388-token fp8 KV pool shared by 16 sequences, tensor parallel over both cards.
- **Enabled**: images, reasoning, tool calls.

Every number was measured on one machine on the date given and links its raw results under [`bench/results/`](bench/results/).

## Numbers

Measured 2026-11-22 on the served configuration ([`scripts/serve.sh`](scripts/serve.sh), image `sm120-r10`), one boot, stock power limits, temperature 0. Decode rate is one stream's output tokens per second from its first to its last token; the aggregate is the sum over the streams running together ([how the numbers are measured](#how-the-numbers-are-measured)).

![Decode: aggregate and per-stream rate at 1, 4 and 8 streams for code at depth, Spec-Bench, and Spec-Bench without the drafter](docs/img/decode.svg)

Blue: code with a 2K append, averaged over the 1K / 32K / 64K cached depths. Brown: Spec-Bench. Grey dashed: Spec-Bench without the drafter, on an earlier image ([`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)).

- **1 agent**: 376 tok/s on Spec-Bench, 423 on code with a cached context; first token in 32 ms.
- **8 agents**: 1,588 tok/s summed on Spec-Bench (188 each, first token 67 ms); 1,724 on cached code (207 each).
- **8 agents at 64K cached context**: 174 tok/s each, 1,557 summed.

![Cold prefill rate and time to first token versus prompt length](docs/img/prefill.svg)

- **Cold prefill**: 18,712 tok/s at 2K prompt tokens, 14,693 at 61K and 8,675 at 246K (28.3 s to the first token).
- **Quality**: tool-eval 82.5 ± 1.9 over 4 trials of 69 scenarios; GSM8K 95.5 % (5-shot, 200 problems).

Sources: Spec-Bench [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10), code [`code-r10`](bench/results/20261122T053347Z-inkling2-code-r10), prefill [`prefill-r10a`](bench/results/20261122T053516Z-inkling2-prefill-r10a), tool-eval [`quality-r10`](bench/results/20261122T053020Z-inkling2-quality-r10), GSM8K [`gsm8k-r10`](bench/results/20261122T060316Z-inkling2-gsm8k-r10), memory [`memory-r10`](bench/results/20261122T053020Z-inkling2-memory-r10). Full tables: [details](#all-numbers).

## Quick start

Requirements: two RTX 5090 GPUs (32 GB each), Docker with GPU support (`--gpus`), and the official checkpoint directory (19.5 GB weights plus 636 MB `mtp.safetensors`). Everything in [`scripts/serve.sh`](scripts/serve.sh) beyond the overlay is the official checkpoint and vLLM's recipe flags; `--disable-custom-all-reduce`, FlashInfer autotune off and `"kv_cache_dtype":"auto"` for the drafter are required on `sm_120` with this image.

```bash
# 1. download the official checkpoint (19.5 GB + 636 MB DSpark drafter mtp.safetensors)
hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/Inkling-2-Mini

# 2. build the SM120 overlay image once (base digest pinned in docker/Dockerfile)
bash scripts/build-image.sh

# 3. start the OpenAI-compatible server on :8000 (model id "inkling-2-mini")
MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh
```

`MODEL_DIR` is the only required variable. Optional overrides in [`scripts/serve.sh`](scripts/serve.sh): `PORT` (default `8000`), `IMAGE` (default `inkling2-rtx5090:sm120-r10`) and `CACHE_DIR` (default `$HOME/.cache/inkling2-vllm`, mounted to `/root/.cache` so Inkling's Triton kernels are cached across boots). The container uses GPUs `0,1` with `--ipc=host`.

The endpoint is OpenAI-compatible at `http://localhost:8000/v1`, model name `inkling-2-mini`:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}]
}'
```

Reasoning (`reasoning_effort` from `none` to `max`), tool calls, image inputs and `prompt_tokens_details.cached_tokens` in `usage` are enabled by default. The model reasons briefly even at `reasoning_effort: "none"`, so evaluations with tight generation budgets or stop strings should account for reasoning tokens (2026-11-22, [`gsm8k-r10`](bench/results/20261122T060316Z-inkling2-gsm8k-r10)).

## All numbers

| | value | source |
| --- | --- | --- |
| context window | 262,144 tokens | [`scripts/serve.sh`](scripts/serve.sh) |
| KV pool on the GPUs | 818,388 tokens, fp8 target KV (unit-scale E4M3), bf16 drafter KV, 16 sequences | 2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined) |
| Spec-Bench decode, 1 / 4 / 8 streams (median per stream) | 376 / 242 / 188 tok/s | 2026-11-22, [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10) |
| Spec-Bench decode aggregate, 1 / 4 / 8 streams | 413 / 1,018 / 1,588 tok/s | 2026-11-22, [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10) |
| Spec-Bench wall-clock output, 1 / 4 / 8 streams | 363.46 / 896.16 / 1,373.87 tok/s | 2026-11-22, [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10) |
| Spec-Bench TTFT (median), 1 / 4 / 8 streams | 32 / 54 / 67 ms | 2026-11-22, [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10) |
| DSpark acceptance length, Spec-Bench, 1 / 4 / 8 streams | 2.97 / 2.98 / 3.00 | 2026-11-22, [`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10) |
| code at depth, 2K append, 1 stream (1K / 32K / 64K cached) | 423 / 446 / 451 tok/s | 2026-11-22, [`code-r10`](bench/results/20261122T053347Z-inkling2-code-r10) |
| code at depth, 2K append, 8 streams aggregate (1K / 32K / 64K cached) | 1,724 / 1,619 / 1,557 tok/s (207 / 181 / 174 per stream) | 2026-11-22, [`code-r10`](bench/results/20261122T053347Z-inkling2-code-r10) |
| cold prefill TTFT, end to end | 0.11 / 0.45 / 4.18 / 28.3 s at 2,045 / 8,187 / 61,435 / 245,758 tokens | 2026-11-22, [`prefill-r10a`](bench/results/20261122T053516Z-inkling2-prefill-r10a) |
| tool-eval-bench 2.1.0, 69 scenarios, 4 trials, T=1.0, 8 in parallel | final score 83 / 100; trial mean 82.5 (sd 1.9) | 2026-11-22, [`quality-r10`](bench/results/20261122T053020Z-inkling2-quality-r10) |
| GSM8K, lm-eval 0.4.12, 5-shot, n=200, T=0, `reasoning_effort` none | 95.5 % flexible extract (se 1.5); 63.5 % strict match | 2026-11-22, [`gsm8k-r10`](bench/results/20261122T060316Z-inkling2-gsm8k-r10) |

Served-path smoke checks on the same boot passed a tool call, an image input and four sampled generations at temperature 0.8 with the drafter (2026-11-22, [`quality-r10`](bench/results/20261122T053020Z-inkling2-quality-r10)).

Memory per card (2026-11-22, [`memory-r10`](bench/results/20261122T053020Z-inkling2-memory-r10)): 10.27 GiB weights and non-torch memory, 15.38 GiB KV cache, 2.42 GiB kept for activations and 0.58 GiB CUDA graphs, within the 28.88 GiB budget of `--gpu-memory-utilization 0.92`. The 818,388-token figure is vLLM's admission estimate (3.12 requests at the full window); it charges every windowed and short-conv layer for its window plus the tokens in flight.

## Served configuration

- Weights: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), official NVFP4/MXFP8 checkpoint (19.5 GB) plus the bundled DSpark drafter `mtp.safetensors` (636 MB).
- Engine and image: `vllm/vllm-openai:inkling-2-mini` (vLLM `0.1.dev25855+g7cb5dae22`, pinned by digest in [`docker/Dockerfile`](docker/Dockerfile)) plus the [`docker/`](docker/README.md) overlay (`inkling2-rtx5090:sm120-r10`), built by [`scripts/build-image.sh`](scripts/build-image.sh) and started by [`scripts/serve.sh`](scripts/serve.sh) on port 8000 as `inkling-2-mini`.
- Parallelism and all-reduce: tensor parallel 2 (`--tensor-parallel-size 2`), FlashInfer PCIe IPC all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) with NCCL fallback, and vLLM's custom all-reduce off (`--disable-custom-all-reduce`).
- Drafter: DSpark (`"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto"`), 7 draft tokens per step (the only value the checkpoint accepts), bf16 drafter KV cache, and draft sampling sharded across both cards (`VLLM_INKLING_DRAFT_SHARD=1`, overlay).
- Context and KV pool: 262,144 tokens (`--max-model-len 262144`), 16 concurrent sequences (`--max-num-seqs 16`), `--gpu-memory-utilization 0.92`, fp8 target KV cache in unit-scale E4M3 (`--kv-cache-dtype fp8`), and exact-size short-conv pages (`VLLM_INKLING_CONV_PAGE=1 --block-size 64`, overlay), giving an 818,388-token KV pool (2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined)).
- Scheduler: `VLLM_DECODE_STEP_BUDGET=1024` (overlay): at most 1,024 tokens per step while any request is decoding, and 2,048 otherwise.
- Sampling: `--enable-batch-sharded-sampling` (overlay): each request is sampled on one card from both cards' vocabulary shards.
- API, warmup and caches: `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` (`reasoning_effort` `none` to `max`), `--enable-prompt-tokens-details` (reports cached prompt tokens in `usage`), FlashInfer warmup autotune off (`--kernel-config '{"enable_flashinfer_autotune":false}'`), and persisted Triton cache (`TRITON_CACHE_DIR=/root/.cache/triton`).

## What the stack is

- **Model**: 24 decoder layers whose middle 12 run three times with shared weights (48 layer applications per token), 128-expert MoE with 8 active, and mostly sliding-window attention with one full-attention layer in six. Experts are NVFP4 in layers 0–19 and MXFP8 in layers 20–23.
- **Drafter**: DSpark, a 3-layer block drafter that reads target hidden states and proposes 7 tokens per step.
- **Overlay**: the day-0 image compiles its core library (`_C_stable_libtorch`) for `sm_80`, `sm_89`, `sm_90` and `sm_100` only, with no PTX fallback. Inkling's own Triton and Gluon kernels run on the RTX 5090 (`sm_120`). The overlay in [`docker/`](docker/README.md) checks the SHA-256 of every file it patches at build time and makes seven changes (two SM120 compatibility fixes and five serving changes):
  - Triton KV-cache scatter for DSpark on SM12x with an unquantized drafter cache (`docker/kv_write_sm120.py`, `docker/patch_dspark.py`) and vLLM's unfused lookup-then-all-reduce path for `VocabParallelEmbedding` at tensor parallel 2 (`docker/patch_embedding.py`).
  - Two small bf16 decode products (loop controllers' down projection and short-conv dynamic taps) moved off cuBLAS onto bitwise-identical Triton kernels on the decode shapes where cuBLAS rounding matches (`docker/decode_sm120.py`, `docker/patch_decode.py`).
  - Decode-aware scheduler step budget (`VLLM_DECODE_STEP_BUDGET=1024`, `docker/patch_sched.py`): caps steps to 1,024 tokens while any stream is decoding and keeps 2,048-token chunks when no stream is decoding.
  - Local logits computation (`compute_logits_local`, `docker/patch_bss.py`), enabling `--enable-batch-sharded-sampling` without gathering full target logits on both cards.
  - Exact-size 8-token short-conv pages in vLLM's `BLNHC` block-outer layout (`VLLM_INKLING_CONV_PAGE=1` with `--block-size 64`, `docker/patch_convpage.py`), enlarging the KV pool by 17 % (from 699,885 to 818,388 tokens as served, 2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined)).
  - Sharded DSpark draft sampling (`VLLM_INKLING_DRAFT_SHARD=1`, `docker/patch_draftshard.py`): each card samples from its vocabulary shard with Gumbel noise keyed by global token id and exchanges one small winner packet per draft position instead of all-gathering full-vocabulary draft logits seven times per step.
  - Patch details, guard checks, microbenchmarks and credits: [`docker/README.md`](docker/README.md).

## How the numbers are measured

- **Boot and configuration**: Spec-Bench, code at depth, cold prefill, tool calling and smoke checks ran on one boot of [`scripts/serve.sh`](scripts/serve.sh) (image `inkling2-rtx5090:sm120-r10`) at stock power limits on 2026-11-22; GSM8K ran on another boot of the same configuration on 2026-11-22.
- **Figures**: [`bench/plot.py`](bench/plot.py) (`uv run bench/plot.py`) generates [`docs/img/decode.svg`](docs/img/decode.svg) and [`docs/img/prefill.svg`](docs/img/prefill.svg) directly from the raw files under [`bench/results/`](bench/results/).
- **Tool calling**: tool-eval-bench 2.1.0, 69 scenarios, 4 trials, temperature 1.0 (the checkpoint's default), 8 in parallel (2026-11-22, [`quality-r10`](bench/results/20261122T053020Z-inkling2-quality-r10)): final score 83 / 100, trial mean 82.5 (standard deviation 1.9).
- **GSM8K**: lm-eval 0.4.12 (`local-chat-completions`), 5-shot, first 200 problems, temperature 0, `reasoning_effort` `none`, concurrency 4, 4,096-token generation budget and a non-matching stop string so brief reasoning does not truncate answers (2026-11-22, [`gsm8k-r10`](bench/results/20261122T060316Z-inkling2-gsm8k-r10)): 95.5 % flexible extract (standard error 1.5), 63.5 % strict match.

### Spec-Bench (`vllm bench serve`)

Measured 2026-11-22 ([`specbench-served-r10`](bench/results/20261122T053112Z-inkling2-specbench-served-r10), driver [`bench/specbench.sh`](bench/specbench.sh), summary [`bench/specbench_summary.py`](bench/specbench_summary.py)): 120-prompt stratified subsets of [Spec-Bench](https://github.com/hemingkx/Spec-Bench) per concurrency ([`bench/inputs/SPEC-BENCH.md`](bench/inputs/SPEC-BENCH.md)), temperature 0, 256 forced output tokens (`--ignore-eos`), 0 failed requests at every concurrency (38,963 input tokens, 30,720 generated tokens per cell). Decode aggregate is stream count times mean request decode rate (the benchmark keeps every stream busy); wall-clock output throughput is vLLM's total generated tokens divided by benchmark duration, including prefill, TTFT and request turnover.

| Streams | Decode per stream (median) | Decode aggregate | Wall-clock output | Req/s | TTFT p50 / p99 | TPOT p50 / p99 | E2E p50 | Acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 376 tok/s | 413 tok/s | 363.46 tok/s | 1.42 | 31.78 / 105.07 ms | 2.66 / 4.10 ms | 717.21 ms | 2.97 |
| 4 | 242 tok/s | 1,018 tok/s | 896.16 tok/s | 3.50 | 53.81 / 157.16 ms | 4.14 / 6.25 ms | 1,124.34 ms | 2.98 |
| 8 | 188 tok/s | 1,588 tok/s | 1,373.87 tok/s | 5.37 | 67.48 / 216.68 ms | 5.31 / 8.37 ms | 1,429.32 ms | 3.00 |

### Code at depth

Measured 2026-11-22 ([`code-r10`](bench/results/20261122T053347Z-inkling2-code-r10), [`bench/headline_inkling.py`](bench/headline_inkling.py) `code`): a shared code context from [ExLlamaV3](https://github.com/turboderp-org/exllamav3) sources cached at 1K / 32K / 64K tokens, a fresh salted tool-output append and a frozen HumanEval task per stream, 400 forced output tokens, temperature 0, `reasoning_effort` `none`. The shared context is verified as a cache hit on every stream. Aggregate is the tokens decoded by all streams during the window in which every stream is decoding, divided by that window's length.

| Cached context | Append | 1 stream | 4 streams: per stream / aggregate | 8 streams: per stream / aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 2K | 423 tok/s | 265 / 1,134 tok/s | 207 / 1,724 tok/s |
| 32K | 2K | 446 tok/s | 253 / 1,079 tok/s | 181 / 1,619 tok/s |
| 64K | 2K | 451 tok/s | 220 / 918 tok/s | 174 / 1,557 tok/s |
| 1K | 8K | 559 tok/s | 213 / 1,005 tok/s | 144 / 1,681 tok/s |
| 32K | 8K | 476 tok/s | 198 / 1,095 tok/s | 126 / 1,606 tok/s |
| 64K | 8K | 619 tok/s | 181 / 974 tok/s | 108 / 1,420 tok/s |

One stream's rate varies with the task and the appended code: each cell is one request, and the drafter accepts more on predictable code, so the six one-stream cells range from 423 to 619 tok/s on this boot. Median TTFT is 0.13–0.78 s for one stream; with eight streams, 64K context and 8K appends it reaches 3.6 s, because the prefills queue together behind the decoding streams.

### Cold prefill

Measured 2026-11-22 ([`prefill-r10a`](bench/results/20261122T053516Z-inkling2-prefill-r10a), [`bench/headline_inkling.py`](bench/headline_inkling.py) `prefill`): fresh salted prompts with zero cached prefix (verified), one at a time, 1 output token; end-to-end rate is prompt tokens divided by client TTFT. A second pass on the same boot is within 2 % on every cell ([`prefill-r10b`](bench/results/20261122T053630Z-inkling2-prefill-r10b)).

| Prompt tokens | 2,045 | 8,187 | 30,715 | 61,435 | 122,876 | 204,797 | 245,758 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 0.11 s | 0.45 s | 1.85 s | 4.18 s | 10.3 s | 21.5 s | 28.3 s |
| Prefill rate | 18,712 | 18,345 | 16,626 | 14,693 | 11,947 | 9,544 | 8,675 tok/s |

## Output checks and caveats

- **Overlay changes are lossless for a lone request**: every serving change in the overlay (Triton decode GEMVs, decode-aware step budget, batch-sharded sampling, exact-size short-conv pages, and DSpark draft sharding) leaves the output of a request served alone bitwise identical. Paired boots of r8 against r10 matched on all 11 prompts of [`bench/greedy.py`](bench/greedy.py) (`--long`, 8 short and 3 long) at temperature 0 and at temperature 1.0 with a fixed seed per prompt (2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined); earlier component checks on 2026-11-21 in [`stepbudget-identity`](bench/results/20261121T192820Z-inkling2-stepbudget-identity) and [`seeded-identity`](bench/results/20261121T231722Z-inkling2-seeded-identity), and 2026-11-22 in [`draftshard`](bench/results/20261122T012901Z-inkling2-draftshard)), alongside 20-trial GPU gates on the conv-page and draft-sharding kernels.
- **Batched output varies run to run**: with several requests in flight, output is not reproducible on this engine because which requests share a step changes batch-variant kernels' arithmetic, as on the stock image. Two passes of the same 22 concurrent requests ([`bench/batched.py`](bench/batched.py)) on one boot produced 0 of 22 identical outputs in every configuration tried on image r8, and batch-sharded sampling diverged from the unsharded configuration no sooner than either configuration diverged from itself (2026-11-21, [`batched-parity`](bench/results/20261121T221734Z-inkling2-batched-parity)).
- **fp8 target KV cache**: `--kv-cache-dtype fp8` changes arithmetic relative to bf16 KV (greedy text diverges after 0–204 characters). On image r5, paired NLL difference (fp8 minus bf16) was −0.00099 nats/token (se 0.00148) on an 8,191-token code text and +0.00109 (se 0.00073) on a 32,767-token code text (2026-11-21, [`fp8kv`](bench/results/20261121T131647Z-inkling2-fp8kv)).
- **Drafter versus no drafter**: greedy output with the drafter is deterministic for a lone request across runs and restarts, but diverges from drafter-off greedy output after a few hundred characters because decode and prefill take different kernel paths. Replaying greedy generations as prompts ([`bench/teacher_check.py`](bench/teacher_check.py)) on image r3 gave the same non-top-1 rate with and without the drafter: 136 of 3,781 tokens (3.6 %, median / max log-prob gap 0.20 / 1.16) with the drafter on against 131 of 3,602 (3.6 %, 0.17 / 1.24) with it off (2026-11-21, [`teacher-check`](bench/results/20261121T0022Z-inkling2-teacher-check)).

## Other configurations and rejected changes

Full progression from r4 to r10, paired A/B boots and all-reduce comparisons are in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md); kernel microbenchmarks and rejected kernel changes are in [`docker/README.md`](docker/README.md).

- **One card / data parallel 2**: tensor parallel 1 leaves 7.65 GiB of KV per 32 GB card without the drafter (fits a 94,208-token context) and 5.2 GiB with it (even a 45,056-token context needs 5.96 GiB; [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#one-model-copy-per-card)). Drafter-off TP1 at 32K context measured 177 tok/s per stream at 1 stream and 418 tok/s aggregate at 4 streams on Spec-Bench (2026-11-21, [`specbench-tp1-nospec-32k`](bench/results/20261121T014555Z-inkling2-specbench-tp1-nospec-32k)).
- **Drafter off**: tensor parallel 2 with NCCL all-reduce measured 206 / 128 / 114 tok/s per stream and 203 / 492 / 905 tok/s aggregate at 1 / 4 / 8 streams on Spec-Bench, with 2.1 s median TTFT at 8 streams (2026-11-21, [`specbench-nospec-r3`](bench/results/20261121T012150Z-inkling2-specbench-nospec-r3)).
- **bf16 target KV cache**: 481,529-token pool against 699,259 with fp8 KV on image r5, at 373 / 225 / 171 vs 369 / 226 / 171 tok/s per stream on Spec-Bench and 18,577 / 8,680 vs 18,160 / 8,559 tok/s cold prefill at 8K / 246K (2026-11-21, [`fp8kv`](bench/results/20261121T131647Z-inkling2-fp8kv)).
- **Global `--max-num-batched-tokens 1024`**: changed greedy output on all three long prompts and cost 5–10 % cold prefill on image r5 (2026-11-21, [`global-batch-1024`](bench/results/20261121T173337Z-inkling2-global-batch-1024)); replaced by `VLLM_DECODE_STEP_BUDGET=1024`.
- **Fused one-pass logit soft-cap and shorter DSpark drafts (3–6 tokens)**: not adopted; the soft-cap is not bitwise equal to PyTorch's three passes, and shorter drafts lost 3–5 % at one stream and changed greedy text (2026-11-21, [`docker/README.md`](docker/README.md#decode-products-moved-off-cublas)).

## Repository map

| path | contents |
| --- | --- |
| [`scripts/build-image.sh`](scripts/build-image.sh) | builds `inkling2-rtx5090:sm120-r10` from the pinned day-0 vLLM base image and [`docker/`](docker/README.md) |
| [`scripts/serve.sh`](scripts/serve.sh) | launches the served TP2 + DSpark configuration on `:8000` |
| [`docker/`](docker/README.md) | [`Dockerfile`](docker/Dockerfile), Triton kernels (`kv_write_sm120.py`, `decode_sm120.py`), SHA-checked patch scripts, design notes and credits |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | r4-to-r10 progression, paired comparisons, all-reduce and one-card measurements, and SM120 day-0 failure modes |
| [`bench/specbench.sh`](bench/specbench.sh), [`bench/specbench_summary.py`](bench/specbench_summary.py) | Spec-Bench `vllm bench serve` driver and per-stream / aggregate summary script |
| [`bench/headline_inkling.py`](bench/headline_inkling.py) | code-at-depth and cold-prefill benchmark driver |
| [`bench/greedy.py`](bench/greedy.py), [`bench/batched.py`](bench/batched.py), [`bench/teacher_check.py`](bench/teacher_check.py) | single-request greedy/seeded identity, concurrent batched parity, and drafter teacher-replay checks |
| [`bench/plot.py`](bench/plot.py) | draws [`docs/img/decode.svg`](docs/img/decode.svg) and [`docs/img/prefill.svg`](docs/img/prefill.svg) from committed raw results |
| [`bench/inputs/`](bench/inputs/SPEC-BENCH.md) | stratified Spec-Bench subsets, frozen code corpus and HumanEval task prompts |
| [`bench/results/`](bench/results/) | raw records for every dated measurement cited in the README and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| [`LICENSE`](LICENSE) | MIT license for original work and Apache-2.0 notice for vLLM-derived parts |

## License

MIT ([`LICENSE`](LICENSE)) for the original work in this repository (documentation, benchmarks, scripts and patch logic). Modifications to vLLM remain under Apache-2.0. Credits for ideas and upstream code are in [`docker/README.md`](docker/README.md#credits).
