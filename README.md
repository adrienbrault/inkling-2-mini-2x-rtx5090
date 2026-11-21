# Inkling-2-Mini on two RTX 5090s

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 26B total / 3.4B active
looped MoE, text and image in) served with vLLM on two RTX 5090 (32 GB, PCIe, no NVLink): both cards as one model
(tensor parallel 2), the checkpoint's own DSpark drafter, and the full 262,144-token context. A single 5090 cannot hold
the drafter and a useful context; two give a 312,417-token KV pool.

vLLM's day-0 image for this model does not run its drafter on consumer Blackwell (sm_120). The small overlay in
[`docker/`](docker/README.md) makes it run; everything else is the official checkpoint and vLLM's recipe flags.

## Numbers

Measured 2026-11-21 on one boot of the served configuration (`scripts/serve.sh`, image from `scripts/build-image.sh`),
stock power limits. **Decode rate** is one stream's output tokens per second from its first to its last token, from the
server's per-chunk token counts; **aggregate** is the total decode rate of the streams running together. Time to first
token (TTFT) is separate.

**Spec-Bench** (120-prompt stratified subset per concurrency, temperature 0, 256 forced output tokens, `vllm bench serve`;
[results](bench/results/20261121T010840Z-inkling2-specbench-served-r3), `bench/specbench.sh`,
`bench/specbench_summary.py`):

| Streams | Decode rate per stream (median) | Aggregate | TTFT (median) | Drafter acceptance length |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 319 tok/s | 347 tok/s | 37 ms | 2.96 |
| 4 | 184 tok/s | 719 tok/s | 62 ms | 2.95 |
| 8 | 115 tok/s | 997 tok/s | 81 ms | 2.96 |

Here the aggregate is the stream count times the mean request decode rate (the benchmark keeps every stream busy).

**Code at depth** (a shared code context cached at 1K / 32K / 64K tokens, a fresh tool-output append and a HumanEval task
per stream, 400 forced output tokens, temperature 0, no thinking; the cached context is verified as a cache hit on every
stream; [results](bench/results/20261121T011319Z-inkling2-code-r3), `bench/headline_inkling.py code`). Aggregate = tokens
decoded by all streams during the window in which every stream is decoding, divided by its length:

| Cached context | Append | 1 stream | 4 streams: per stream / aggregate | 8 streams: per stream / aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 2K | 498 tok/s | 224 / 873 tok/s | 178 / 1,459 tok/s |
| 32K | 2K | 383 tok/s | 189 / 957 tok/s | 178 / 1,339 tok/s |
| 64K | 2K | 523 tok/s | 169 / 849 tok/s | 162 / 1,379 tok/s |
| 1K | 8K | 314 tok/s | 202 / 951 tok/s | 134 / 1,453 tok/s |
| 32K | 8K | 418 tok/s | 173 / 913 tok/s | 119 / 1,428 tok/s |
| 64K | 8K | 364 tok/s | 157 / 830 tok/s | 106 / 1,290 tok/s |

One stream's rate varies with the task and the appended code (the drafter accepts more on predictable code). Median
TTFT is 0.13–0.77 s for one stream and up to 3.8 s for eight streams with 8K appends, whose prefills queue together.

**Cold prefill** (fresh salted prompts with no cached prefix, verified, one at a time; end to end: prompt tokens divided
by the client's TTFT; [results](bench/results/20261121T011457Z-inkling2-prefill-r3a), `bench/headline_inkling.py prefill`):

| Prompt tokens | 2,043 | 8,192 | 30,718 | 61,437 | 122,880 | 204,798 | 245,759 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 0.11 s | 0.44 s | 1.82 s | 4.13 s | 10.2 s | 21.4 s | 28.3 s |
| Prefill rate | 18,844 | 18,627 | 16,857 | 14,880 | 12,041 | 9,586 | 8,696 tok/s |

A second pass on the same boot is within 1.3 % on every cell ([results](bench/results/20261121T011611Z-inkling2-prefill-r3b)).

## Served configuration

```bash
bash scripts/build-image.sh
MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh     # OpenAI-compatible API on :8000, model "inkling-2-mini"
```

| Setting | Value |
| --- | --- |
| Checkpoint | `thinkingmachines/Inkling-2-Mini`, official NVFP4/MXFP8 (19.5 GB) + DSpark drafter `mtp.safetensors` (636 MB) |
| Engine | `vllm/vllm-openai:inkling-2-mini` (vLLM `0.1.dev25855+g7cb5dae22`, pinned by digest) + the `docker/` overlay (`inkling2-rtx5090:sm120-r3`) |
| Parallelism | tensor parallel 2, NCCL all-reduce (`--disable-custom-all-reduce`) |
| Drafter | DSpark, 7 draft tokens per step (the only value the checkpoint accepts) |
| Context | 262,144 tokens; KV pool 312,417 tokens (bf16), `--gpu-memory-utilization 0.92` |
| API | `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice`; `reasoning_effort` none … max |
| Warmup | FlashInfer autotune off (`--kernel-config`) |
| Usage | `--enable-prompt-tokens-details` (cached prompt tokens in `usage`) |

## What the stack is

- **Model**: 24 decoder layers whose middle 12 run three times with shared weights (48 layer applications per token),
  128-expert MoE with 8 active, mostly sliding-window attention with one full-attention layer in six. Experts are NVFP4
  (layers 0–19) and MXFP8 (layers 20–23).
- **Drafter**: DSpark, a 3-layer block drafter that reads target hidden states and proposes 7 tokens per step.
- **Overlay**: the day-0 image's compiled core library has no sm_120 code. Inkling's own kernels (Triton/Gluon) run
  on the 5090; the overlay replaces the two compiled kernels the drafter and the tensor-parallel embedding reach.
  Details and the guard checks: [`docker/README.md`](docker/README.md).

Alternatives that were measured (no drafter, one model copy per card) and the drafter's correctness check are in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).
