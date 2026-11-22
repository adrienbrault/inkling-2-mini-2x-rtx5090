# Inkling-2-Mini on two RTX 5090s

![Recipe card: decode at 1 to 16 agents, cold prefill, KV pool and memory on two RTX 5090s, 2026-11-22](docs/img/recipe-card-1efbaa6c.png)

Serving configuration, vLLM overlay, launch scripts and measurements for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on two RTX 5090 cards.

**Goal**: run many coding agents from one desktop box. Several agent sessions share the server, their prompts grow with every tool call up to the 262K window and mostly hit the prefix cache, and new prompts arrive while the others decode. The 16 sequences, the KV pool, the scheduler settings and the benchmarks below follow from that workload.

- **Model**: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines; 26B total, 3.4B active; text and images in), the official checkpoint and its own DSpark drafter (7 draft tokens).
- **Engine**: vLLM's Inkling day-0 image plus a small overlay ([`docker/`](docker/README.md)) that makes it run on the RTX 5090 and speeds up serving. A request served alone gets bitwise the same output with or without the overlay and the scheduler settings.
- **Window and KV pool**: 262,144 tokens per request; the fp8 KV pool holds about 1.49M tokens of context at once, measured, shared by 16 sequences (vLLM logs 818,388, a worst-case estimate), tensor parallel over both cards.
- **Enabled**: images, reasoning, tool calls.

Every number was measured on one machine on the date given and links its raw results under [`bench/results/`](bench/results/).

## Numbers

Measured 2026-11-22 on the served configuration ([`scripts/serve.sh`](scripts/serve.sh), image `sm120-r10`, fair prefill, Rust frontend), one boot, stock power limits. Decode rate is one stream's output tokens per second from its first to its last token, averaged over the requests (the median is in the tables below); the aggregate is the sum over the streams running together ([how the numbers are measured](#how-the-numbers-are-measured)).

![Decode: aggregate and per-stream rate at 1, 4, 8 and 16 streams for code at depth, Spec-Bench, and Spec-Bench without the drafter](docs/img/decode.svg)

Blue: code with a 2K append, averaged over the 1K / 32K / 64K cached depths. Brown: Spec-Bench. Grey dashed: Spec-Bench without the drafter, on an earlier image ([`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)).

- **1 agent**: 416 tok/s on Spec-Bench, 532 on code with a cached context; first token in 32 ms.
- **16 agents**: 2,419 tok/s summed on Spec-Bench (151 each, first token 106 ms); 2,717 on cached code (147 each). At 64K cached context: 112 each, 2,227 summed.
- **Agent sessions**: 711 tok/s summed with 10 coding-agent sessions replayed at once (prompts of about 39K–202K tokens, p10–p90, mostly cached), first token 0.23 s median and 0.72 s p90; 511 tok/s with 6 sessions, first token 0.15 s median.
- **Newcomers**: a short request arriving behind a long cold prompt gets its first token in 0.5 s instead of 20 s (fair prefill).

![Cold prefill rate and time to first token versus prompt length](docs/img/prefill.svg)

- **Cold prefill**: 19,115 tok/s at 2K prompt tokens, 14,835 at 61K and 8,716 at 246K (28.2 s to the first token).
- **Quality**: tool-eval 83.5 ± 3.0 over 4 trials of 69 scenarios; GSM8K 94.5 % (5-shot, 200 problems).

- **KV pool**: 14 agents with 80K-token prompts decoding together used 88 % of it, nothing preempted: about 1.49M tokens of context fit at once.

Sources: KV pool [`kv-capacity`](bench/results/20261122T124729Z-inkling2-kv-capacity), Spec-Bench [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe), code [`code-rustfe`](bench/results/20261122T143647Z-inkling2-code-rustfe), prefill [`prefill-rustfe-a`](bench/results/20261122T143852Z-inkling2-prefill-rustfe-a), agent sessions [`pi-rustfe`](bench/results/20261122T143310Z-inkling2-pi-rustfe), newcomers [`fair-prefill-768`](bench/results/20261122T092118Z-inkling2-fair-prefill-768), tool-eval and GSM8K [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe), memory [`memory-r10`](bench/results/20261122T053020Z-inkling2-memory-r10). Full tables: [details](#all-numbers).

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

Reasoning (`reasoning_effort` from `none` to `max`), tool calls, image inputs and `prompt_tokens_details.cached_tokens` in `usage` are enabled by default. The model reasons briefly even at `reasoning_effort: "none"`, so evaluations with tight generation budgets or stop strings should account for reasoning tokens (2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe)).

## All numbers

| | value | source |
| --- | --- | --- |
| context window | 262,144 tokens | [`scripts/serve.sh`](scripts/serve.sh) |
| KV pool on the GPUs, measured | about 1.49–1.51M tokens held at once (8, 12 and 14 long distinct requests, 0 preemptions); vLLM logs 818,388 | 2026-11-22, [`kv-capacity`](bench/results/20261122T124729Z-inkling2-kv-capacity) |
| KV cache | fp8 target KV (unit-scale E4M3), bf16 drafter KV, 16 sequences | [`scripts/serve.sh`](scripts/serve.sh) |
| Spec-Bench decode, 1 / 4 / 8 / 16 streams (mean per stream; median) | 416 / 258 / 200 / 151 tok/s; 379 / 248 / 185 / 142 tok/s | 2026-11-22, [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| Spec-Bench decode aggregate, 1 / 4 / 8 / 16 streams | 416 / 1,034 / 1,601 / 2,419 tok/s | 2026-11-22, [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| Spec-Bench wall-clock output, 1 / 4 / 8 / 16 streams | 367.37 / 904.38 / 1,379.87 / 2,020.57 tok/s | 2026-11-22, [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| Spec-Bench TTFT (median), 1 / 4 / 8 / 16 streams | 32 / 54 / 69 / 106 ms | 2026-11-22, [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| DSpark acceptance length, Spec-Bench, 1 / 4 / 8 / 16 streams | 2.97 / 3.01 / 2.99 / 2.97 | 2026-11-22, [`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| code at depth, 2K append, 1 stream (1K / 32K / 64K cached) | 532 / 463 / 548 tok/s | 2026-11-22, [`code-rustfe`](bench/results/20261122T143647Z-inkling2-code-rustfe) |
| code at depth, 2K append, 8 streams aggregate (1K / 32K / 64K cached) | 1,871 / 1,658 / 1,582 tok/s (206 / 188 / 177 per stream) | 2026-11-22, [`code-rustfe`](bench/results/20261122T143647Z-inkling2-code-rustfe) |
| code at depth, 2K append, 16 streams aggregate (1K / 32K / 64K cached) | 2,717 / 2,714 / 2,227 tok/s (147 / 132 / 112 per stream) | 2026-11-22, [`code-rustfe`](bench/results/20261122T143647Z-inkling2-code-rustfe) |
| agent replay, 6 / 10 sessions: aggregate decode | 511 / 711 tok/s (211 / 147 per stream) | 2026-11-22, [`pi-rustfe`](bench/results/20261122T143310Z-inkling2-pi-rustfe) |
| agent replay, 6 / 10 sessions: first token p50 / p90 | 0.15 / 0.69 s ; 0.23 / 0.72 s | 2026-11-22, [`pi-rustfe`](bench/results/20261122T143310Z-inkling2-pi-rustfe) |
| short / 16K newcomer behind a cold 196K prompt, first token (Python frontend) | 0.51–0.52 s / 6.66–6.67 s (20.4–20.5 s / 21.5 s without fair prefill) | 2026-11-22, [`fair-prefill-768`](bench/results/20261122T092118Z-inkling2-fair-prefill-768) |
| cold prefill TTFT, end to end | 0.11 / 0.44 / 4.14 / 28.2 s at 2,046 / 8,192 / 61,433 / 245,758 tokens | 2026-11-22, [`prefill-rustfe-a`](bench/results/20261122T143852Z-inkling2-prefill-rustfe-a) |
| tool-eval-bench 2.1.0, 69 scenarios, 4 trials, T=1.0, 8 in parallel | trial mean 83.5 (sd 3.0), median 84.0 | 2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe) |
| GSM8K, lm-eval 0.4.12, 5-shot, n=200, T=0, `reasoning_effort` none | 94.5 % flexible extract (se 1.6); 68.0 % strict match | 2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe) |

Served-path smoke checks on the same boot passed a tool call, an image input and four sampled generations at temperature 0.8 with the drafter (2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe)).

Memory per card (2026-11-22, boot without the fair-prefill flags and with the Python frontend, neither of which changes allocation, [`memory-r10`](bench/results/20261122T053020Z-inkling2-memory-r10)): 10.27 GiB weights and non-torch memory, 15.38 GiB KV cache, 2.42 GiB kept for activations and 0.58 GiB CUDA graphs, within the 28.88 GiB budget of `--gpu-memory-utilization 0.92`. About 3.9 GiB per card stays free on purpose: vLLM sizes the KV cache from the memory measured after warmup and keeps free the activation reserve (2.42 GiB) plus a margin of 75 % of the measured activation peak (1.46 GiB) against out-of-memory errors. That margin, not `--gpu-memory-utilization`, bounds the pool: 0.96 gave 25,270 blocks instead of 25,200 (+0.3 %). The 818,388-token figure vLLM logs is its admission estimate for full-window requests (3.12 requests at 262,144 tokens); it charges every windowed and short-conv cache for its window plus the tokens in flight. At run time only the full-attention history grows with the context: the 25,200 blocks hold 64 tokens of it each (1,612,800 tokens), less about 10K tokens of window state per running request, and the pool measured 1,486,835–1,508,118 tokens held at once ([`kv-capacity`](bench/results/20261122T124729Z-inkling2-kv-capacity)).

## Served configuration

- Weights: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), official NVFP4/MXFP8 checkpoint (19.5 GB) plus the bundled DSpark drafter `mtp.safetensors` (636 MB).
- Engine and image: `vllm/vllm-openai:inkling-2-mini` (vLLM `0.1.dev25855+g7cb5dae22`, pinned by digest in [`docker/Dockerfile`](docker/Dockerfile)) plus the [`docker/`](docker/README.md) overlay (`inkling2-rtx5090:sm120-r10`), built by [`scripts/build-image.sh`](scripts/build-image.sh) and started by [`scripts/serve.sh`](scripts/serve.sh) on port 8000 as `inkling-2-mini`.
- Parallelism and all-reduce: tensor parallel 2 (`--tensor-parallel-size 2`), FlashInfer PCIe IPC all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) with NCCL fallback, and vLLM's custom all-reduce off (`--disable-custom-all-reduce`).
- Drafter: DSpark (`"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto"`), 7 draft tokens per step (the only value the checkpoint accepts), bf16 drafter KV cache, and draft sampling sharded across both cards (`VLLM_INKLING_DRAFT_SHARD=1`, overlay).
- Context and KV pool: 262,144 tokens (`--max-model-len 262144`), 16 concurrent sequences (`--max-num-seqs 16`), `--gpu-memory-utilization 0.92`, fp8 target KV cache in unit-scale E4M3 (`--kv-cache-dtype fp8`), and exact-size short-conv pages (`VLLM_INKLING_CONV_PAGE=1 --block-size 64`, overlay), giving a KV pool that holds about 1.49M tokens of context at once (vLLM logs 818,388; 2026-11-22, [`kv-capacity`](bench/results/20261122T124729Z-inkling2-kv-capacity)).
- Scheduler: `VLLM_DECODE_STEP_BUDGET=1024` (overlay): at most 1,024 tokens per step while any request is decoding, and 2,048 otherwise. Fair prefill (`--long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive`, stock vLLM): while several requests prefill, each gets at most max(768, 2,048 / their number) tokens per step.
- Sampling: `--enable-batch-sharded-sampling` (overlay): each request is sampled on one card from both cards' vocabulary shards.
- Frontend: vLLM's Rust API server (`VLLM_USE_RUST_FRONTEND=1`, stock vLLM) in place of the Python one: the same output text, token counts and throughput, and the first token about 0.1 s sooner on long agent prompts (median 0.20–0.23 s against 0.30–0.32 s with 10 sessions, two boots each; [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#rust-frontend)).
- API, warmup and caches: `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` (`reasoning_effort` `none` to `max`), `--enable-prompt-tokens-details` (reports cached prompt tokens in `usage`), FlashInfer warmup autotune off (`--kernel-config '{"enable_flashinfer_autotune":false}'`), and persisted Triton cache (`TRITON_CACHE_DIR=/root/.cache/triton`).

## What the stack is

- **Model**: 24 decoder layers whose middle 12 run three times with shared weights (48 layer applications per token), 128-expert MoE with 8 active, and mostly sliding-window attention with one full-attention layer in six. Experts are NVFP4 in layers 0–19 and MXFP8 in layers 20–23.
- **Drafter**: DSpark, a 3-layer block drafter that reads target hidden states and proposes 7 tokens per step.
- **Overlay**: the day-0 image compiles its core library (`_C_stable_libtorch`) for `sm_80`, `sm_89`, `sm_90` and `sm_100` only, with no PTX fallback. Inkling's own Triton and Gluon kernels run on the RTX 5090 (`sm_120`). The overlay in [`docker/`](docker/README.md) checks the SHA-256 of every file it patches at build time and makes seven changes (two SM120 compatibility fixes and five serving changes):
  - Triton KV-cache scatter for DSpark on SM12x with an unquantized drafter cache (`docker/kv_write_sm120.py`, `docker/patch_dspark.py`) and vLLM's unfused lookup-then-all-reduce path for `VocabParallelEmbedding` at tensor parallel 2 (`docker/patch_embedding.py`).
  - Two small bf16 decode products (loop controllers' down projection and short-conv dynamic taps) moved off cuBLAS onto bitwise-identical Triton kernels on the decode shapes where cuBLAS rounding matches (`docker/decode_sm120.py`, `docker/patch_decode.py`).
  - Decode-aware scheduler step budget (`VLLM_DECODE_STEP_BUDGET=1024`, `docker/patch_sched.py`): caps steps to 1,024 tokens while any stream is decoding and keeps 2,048-token chunks when no stream is decoding.
  - Local logits computation (`compute_logits_local`, `docker/patch_bss.py`), enabling `--enable-batch-sharded-sampling` without gathering full target logits on both cards.
  - Exact-size 8-token short-conv pages in vLLM's `BLNHC` block-outer layout (`VLLM_INKLING_CONV_PAGE=1` with `--block-size 64`, `docker/patch_convpage.py`), enlarging the pool by 17 % in vLLM's logged estimate (from 699,885 to 818,388 tokens as served, 2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined)).
  - Sharded DSpark draft sampling (`VLLM_INKLING_DRAFT_SHARD=1`, `docker/patch_draftshard.py`): each card samples from its vocabulary shard with Gumbel noise keyed by global token id and exchanges one small winner packet per draft position instead of all-gathering full-vocabulary draft logits seven times per step.
  - Patch details, guard checks, microbenchmarks and credits: [`docker/README.md`](docker/README.md).

## How the numbers are measured

- **Boot and configuration**: Spec-Bench, code at depth, cold prefill, the agent replay, tool calling, GSM8K and smoke checks ran on one boot of [`scripts/serve.sh`](scripts/serve.sh) (image `inkling2-rtx5090:sm120-r10`, fair prefill, Rust frontend) at stock power limits on 2026-11-22.
- **Figures**: [`bench/plot.py`](bench/plot.py) (`uv run bench/plot.py`) generates [`docs/img/decode.svg`](docs/img/decode.svg) and [`docs/img/prefill.svg`](docs/img/prefill.svg) directly from the raw files under [`bench/results/`](bench/results/).
- **Agent replay**: [`bench/inkling_replay.py`](bench/inkling_replay.py) `--preset pi` replays coding-agent sessions: long-lived conversations that grow with every tool call (preset prompt p10 / p50 / p90 of about 39K / 115K / 202K tokens, mostly prefix-cached), replies of 282 tokens median and short gaps between steps, temperature 1.0, 6 and 10 sessions at once, a 150 s scored window after 150 s of warmup (2026-11-22, [`pi-rustfe`](bench/results/20261122T143310Z-inkling2-pi-rustfe)). Aggregate decode is the tokens decoded inside the window divided by its length.
- **Newcomers**: [`bench/newcomer.py`](bench/newcomer.py): one stream decoding in the background, a cold 196,608-token prompt, and 2 s later a cold ~1K and ~16K prompt; medians of three trials per boot, four boots alternating with and without fair prefill (2026-11-22, [`fair-prefill-768`](bench/results/20261122T092118Z-inkling2-fair-prefill-768)).
- **Tool calling**: tool-eval-bench 2.1.0, 69 scenarios, 4 trials, temperature 1.0 (the checkpoint's default), 8 in parallel (2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe)): trial mean 83.5 (standard deviation 3.0), median 84.0.
- **GSM8K**: lm-eval 0.4.12 (`local-chat-completions`), 5-shot, first 200 problems, temperature 0, `reasoning_effort` `none`, 4,096-token generation budget and a non-matching stop string so brief reasoning does not truncate answers (2026-11-22, [`quality-rustfe`](bench/results/20261122T143310Z-inkling2-quality-rustfe)): 94.5 % flexible extract (standard error 1.6), 68.0 % strict match.

### Spec-Bench (`vllm bench serve`)

Measured 2026-11-22 ([`specbench-served-rustfe`](bench/results/20261122T143353Z-inkling2-specbench-served-rustfe), driver [`bench/specbench.sh`](bench/specbench.sh), summary [`bench/specbench_summary.py`](bench/specbench_summary.py)): 120-prompt stratified subsets of [Spec-Bench](https://github.com/hemingkx/Spec-Bench) per concurrency ([`bench/inputs/SPEC-BENCH.md`](bench/inputs/SPEC-BENCH.md)), temperature 0, 256 forced output tokens (`--ignore-eos`), 0 failed requests at every concurrency (38,963 input tokens, 30,720 generated tokens per cell). Per-request decode rates are skewed by the drafter's acceptance (at 1 stream, p10 / p90 296 / 580 tok/s), so the mean exceeds the median. Decode aggregate is stream count times the mean request decode rate (the benchmark keeps every stream busy); wall-clock output throughput is vLLM's total generated tokens divided by benchmark duration, including prefill, TTFT and request turnover.

| Streams | Decode per stream (median / mean) | Decode aggregate | Wall-clock output | Req/s | TTFT p50 / p99 | TPOT p50 / p99 | E2E p50 | Acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 379 / 416 tok/s | 416 tok/s | 367.37 tok/s | 1.44 | 31.70 / 107.65 ms | 2.64 / 4.08 ms | 707.75 ms | 2.97 |
| 4 | 248 / 258 tok/s | 1,034 tok/s | 904.38 tok/s | 3.53 | 53.97 / 137.70 ms | 4.04 / 6.20 ms | 1,101.68 ms | 3.01 |
| 8 | 185 / 200 tok/s | 1,601 tok/s | 1,379.87 tok/s | 5.39 | 69.32 / 211.58 ms | 5.41 / 8.54 ms | 1,469.98 ms | 2.99 |
| 16 | 142 / 151 tok/s | 2,419 tok/s | 2,020.57 tok/s | 7.89 | 105.91 / 488.88 ms | 7.06 / 10.78 ms | 1,918.64 ms | 2.97 |

### Code at depth

Measured 2026-11-22 ([`code-rustfe`](bench/results/20261122T143647Z-inkling2-code-rustfe), [`bench/headline_inkling.py`](bench/headline_inkling.py) `code`): a shared code context from [ExLlamaV3](https://github.com/turboderp-org/exllamav3) sources cached at 1K / 32K / 64K tokens, a fresh salted tool-output append and a frozen HumanEval task per stream, 400 forced output tokens, temperature 0, `reasoning_effort` `none`. The shared context is verified as a cache hit on every stream. Aggregate is the tokens decoded by all streams during the window in which every stream is decoding, divided by that window's length.

| Cached context | Append | 1 stream | 4 streams: per stream / aggregate | 8 streams: per stream / aggregate | 16 streams: per stream / aggregate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 2K | 532 tok/s | 264 / 1,100 tok/s | 206 / 1,871 tok/s | 147 / 2,717 tok/s |
| 32K | 2K | 463 tok/s | 254 / 1,037 tok/s | 188 / 1,658 tok/s | 132 / 2,714 tok/s |
| 64K | 2K | 548 tok/s | 230 / 996 tok/s | 177 / 1,582 tok/s | 112 / 2,227 tok/s |
| 1K | 8K | 588 tok/s | 240 / 1,128 tok/s | 145 / 1,606 tok/s | 105 / 1,680 tok/s\* |
| 32K | 8K | 501 tok/s | 213 / 1,192 tok/s | 130 / 1,648 tok/s | 85 / 1,360 tok/s\* |
| 64K | 8K | 560 tok/s | 189 / 1,061 tok/s | 114 / 1,632 tok/s | 73 / 1,163 tok/s\* |

\* With 16 streams and 8K appends, the first streams finish before the last prefill does, so no window has every stream decoding; the aggregate is the sum of the per-stream rates.

One stream's rate varies with the task and the appended code: each cell is one request, and the drafter accepts more on predictable code, so the six one-stream cells range from 463 to 588 tok/s on this boot (449 to 500 on the previous boot of the same image). Median TTFT is 0.13–0.72 s for one stream; with 16 streams it is 1.1–1.8 s with 2K appends and 4.5–7.1 s with 8K appends, because the prefills queue together behind the decoding streams.

### Cold prefill

Measured 2026-11-22 ([`prefill-rustfe-a`](bench/results/20261122T143852Z-inkling2-prefill-rustfe-a), [`bench/headline_inkling.py`](bench/headline_inkling.py) `prefill`): fresh salted prompts with zero cached prefix (verified), one at a time, 1 output token; end-to-end rate is prompt tokens divided by client TTFT. A second pass on the same boot is within 0.4 % on every cell ([`prefill-rustfe-b`](bench/results/20261122T143959Z-inkling2-prefill-rustfe-b)).

| Prompt tokens | 2,046 | 8,192 | 30,720 | 61,433 | 122,876 | 204,800 | 245,758 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 0.11 s | 0.44 s | 1.83 s | 4.14 s | 10.2 s | 21.4 s | 28.2 s |
| Prefill rate | 19,115 | 18,638 | 16,808 | 14,835 | 12,023 | 9,592 | 8,716 tok/s |

## Output checks and caveats

- **Serving changes are lossless for a lone request**: every serving change (Triton decode GEMVs, decode-aware step budget, batch-sharded sampling, exact-size short-conv pages, DSpark draft sharding, and fair prefill) leaves the output of a request served alone bitwise identical. Paired boots of r8 against r10 matched on all 11 prompts of [`bench/greedy.py`](bench/greedy.py) (`--long`, 8 short and 3 long) at temperature 0 and at temperature 1.0 with a fixed seed per prompt (2026-11-22, [`r10-combined`](bench/results/20261122T032748Z-inkling2-r10-combined); earlier component checks on 2026-11-21 in [`stepbudget-identity`](bench/results/20261121T192820Z-inkling2-stepbudget-identity) and [`seeded-identity`](bench/results/20261121T231722Z-inkling2-seeded-identity), and 2026-11-22 in [`draftshard`](bench/results/20261122T012901Z-inkling2-draftshard)), alongside 20-trial GPU gates on the conv-page and draft-sharding kernels. Fair prefill kept the same 11 outputs on four alternating boots (2026-11-22, [`fair-prefill-768`](bench/results/20261122T092118Z-inkling2-fair-prefill-768)).
- **Rust frontend**: at temperature 0 the 11 [`bench/greedy.py`](bench/greedy.py) outputs have the same text and token counts as with the Python frontend, except that the streamed reasoning keeps its last whitespace characters (for example a final `"\n  "`) that the Python frontend drops, and `usage` omits the empty `multimodal_tokens` field (2026-11-22, [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#rust-frontend)).
- **Admission**: vLLM admits a request only when its whole prompt fits (`--scheduler-reserve-full-isl`, the default). With 16 distinct 96K prompts it ran 12 and queued the rest without preempting anything; with `--no-scheduler-reserve-full-isl` it admitted all 16 and then preempted requests 326 times as their outputs outgrew the pool (2026-11-22, [`kv-capacity`](bench/results/20261122T124729Z-inkling2-kv-capacity)).
- **Batched output varies run to run**: with several requests in flight, output is not reproducible on this engine because which requests share a step changes batch-variant kernels' arithmetic, as on the stock image. Two passes of the same 22 concurrent requests ([`bench/batched.py`](bench/batched.py)) on one boot produced 0 of 22 identical outputs in every configuration tried on image r8, and batch-sharded sampling diverged from the unsharded configuration no sooner than either configuration diverged from itself (2026-11-21, [`batched-parity`](bench/results/20261121T221734Z-inkling2-batched-parity)).
- **fp8 target KV cache**: `--kv-cache-dtype fp8` changes arithmetic relative to bf16 KV (greedy text diverges after 0–204 characters). On image r5, paired NLL difference (fp8 minus bf16) was −0.00099 nats/token (se 0.00148) on an 8,191-token code text and +0.00109 (se 0.00073) on a 32,767-token code text (2026-11-21, [`fp8kv`](bench/results/20261121T131647Z-inkling2-fp8kv)).
- **Drafter versus no drafter**: greedy output with the drafter is deterministic for a lone request across runs and restarts, but diverges from drafter-off greedy output after a few hundred characters because decode and prefill take different kernel paths. Replaying greedy generations as prompts ([`bench/teacher_check.py`](bench/teacher_check.py)) on image r3 gave the same non-top-1 rate with and without the drafter: 136 of 3,781 tokens (3.6 %, median / max log-prob gap 0.20 / 1.16) with the drafter on against 131 of 3,602 (3.6 %, 0.17 / 1.24) with it off (2026-11-21, [`teacher-check`](bench/results/20261121T0022Z-inkling2-teacher-check)).

## Other configurations and rejected changes

Full progression from r4 to r10, paired A/B boots and all-reduce comparisons are in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md); kernel microbenchmarks and rejected kernel changes are in [`docker/README.md`](docker/README.md).

- **One card / data parallel 2**: tensor parallel 1 leaves 7.65 GiB of KV per 32 GB card without the drafter (fits a 94,208-token context) and 5.2 GiB with it (even a 45,056-token context needs 5.96 GiB; [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#one-model-copy-per-card)). Drafter-off TP1 at 32K context measured 177 tok/s per stream at 1 stream and 418 tok/s aggregate at 4 streams on Spec-Bench (2026-11-21, [`specbench-tp1-nospec-32k`](bench/results/20261121T014555Z-inkling2-specbench-tp1-nospec-32k)).
- **Drafter off**: tensor parallel 2 with NCCL all-reduce measured 203 / 123 / 113 tok/s per stream (mean; median 206 / 128 / 114) and 203 / 492 / 905 tok/s aggregate at 1 / 4 / 8 streams on Spec-Bench, with 2.1 s median TTFT at 8 streams (2026-11-21, [`specbench-nospec-r3`](bench/results/20261121T012150Z-inkling2-specbench-nospec-r3)).
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
| [`bench/kv_capacity.py`](bench/kv_capacity.py), [`bench/kv_probe.py`](bench/kv_probe.py) | measures the KV tokens actually held at once against vLLM's own usage metric |
| [`bench/inkling_replay.py`](bench/inkling_replay.py), [`bench/newcomer.py`](bench/newcomer.py) | coding-agent session replay and the newcomer-behind-a-long-prefill probe |
| [`bench/greedy.py`](bench/greedy.py), [`bench/batched.py`](bench/batched.py), [`bench/teacher_check.py`](bench/teacher_check.py) | single-request greedy/seeded identity, concurrent batched parity, and drafter teacher-replay checks |
| [`bench/plot.py`](bench/plot.py) | draws [`docs/img/decode.svg`](docs/img/decode.svg) and [`docs/img/prefill.svg`](docs/img/prefill.svg) from committed raw results |
| [`bench/inputs/`](bench/inputs/SPEC-BENCH.md) | stratified Spec-Bench subsets, frozen code corpus and HumanEval task prompts |
| [`bench/results/`](bench/results/) | raw records for every dated measurement cited in the README and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| [`LICENSE`](LICENSE) | MIT license for original work and Apache-2.0 notice for vLLM-derived parts |

## License

MIT ([`LICENSE`](LICENSE)) for the original work in this repository (documentation, benchmarks, scripts and patch logic). Modifications to vLLM remain under Apache-2.0. Credits for ideas and upstream code are in [`docker/README.md`](docker/README.md#credits).
