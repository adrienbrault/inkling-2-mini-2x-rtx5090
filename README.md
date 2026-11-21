# Inkling-2-Mini on two RTX 5090s

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines, 26B total / 3.4B active
looped MoE, text and image in) served with vLLM on two RTX 5090 (32 GB, PCIe, no NVLink): both cards as one model
(tensor parallel 2), the checkpoint's own DSpark drafter, and the full 262,144-token context. A single 5090 cannot hold
the drafter and a useful context; two give a 312,417-token KV pool.

vLLM's day-0 image for this model does not run its drafter on consumer Blackwell (sm_120). The small overlay in
[`docker/`](docker/README.md) makes it run; everything else is the official checkpoint and vLLM's recipe flags.

## Numbers

Measured 2026-11-21 on the served configuration (`scripts/serve.sh`, image built by `scripts/build-image.sh`), stock
power limits. **Decode rate** is one stream's output tokens per second from its first to its last token;
**aggregate** is the sum of the rates of the streams running together. Time to first token (TTFT) is separate.

**Spec-Bench** (120-prompt stratified subset of each concurrency's set, temperature 0, 256 output tokens, `vllm bench
serve`; [results](bench/results/20261121T003545Z-inkling2-specbench-served-tp2-dspark), summary by
`bench/specbench_summary.py`):

| Streams | Decode rate per stream (median) | Aggregate | TTFT (median) |
| ---: | ---: | ---: | ---: |
| 1 | 319 tok/s | 343 tok/s | 37 ms |
| 4 | 187 tok/s | 732 tok/s | 63 ms |
| 8 | 136 tok/s | 1,060 tok/s | 80 ms |

The drafter's mean acceptance length on this set is 2.88 tokens per verification step.

**Code at depth** (a shared code context cached at 1K / 32K / 64K tokens, a fresh 2K tool-output append per stream, a
HumanEval task, 400 forced output tokens, temperature 0, no thinking; [results](bench/results/20261121T005515Z-inkling2-code),
client `bench/headline_inkling.py code`):

| Cached context | 1 stream | 4 streams: per stream / aggregate | 8 streams: per stream / aggregate |
| ---: | ---: | ---: | ---: |
| 1K | 450 tok/s | 169 / 676 tok/s | 130 / 1,040 tok/s |
| 32K | 457 tok/s | 213 / 852 tok/s | 168 / 1,345 tok/s |
| 64K | 495 tok/s | 170 / 681 tok/s | 154 / 1,233 tok/s |

With an 8K append instead of 2K, one stream decodes at 387–481 tok/s and 4 and 8 streams reach 643–812 tok/s
aggregate: the appends' prefills overlap the other streams' decode. All cells are in `code-summary.json`.

**Cold prefill** (fresh prompts with no cached prefix, one at a time; [results](bench/results/20261121T005354Z-inkling2-prefill-pass2),
client `bench/headline_inkling.py prefill`):

| Prompt tokens | 2,046 | 8,190 | 30,718 | 61,436 | 122,876 | 204,797 | 245,758 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 0.11 s | 0.44 s | 1.82 s | 4.13 s | 10.2 s | 21.3 s | 28.2 s |
| Prefill rate | 18,611 | 18,668 | 16,896 | 14,894 | 12,063 | 9,608 | 8,717 tok/s |

A first pass measured the same within 1 % except its 8K cell, which paid a one-time kernel warmup
([results](bench/results/20261121T005235Z-inkling2-prefill)).

## Served configuration

```bash
bash scripts/build-image.sh
MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh     # OpenAI-compatible API on :8000, model "inkling-2-mini"
```

| Setting | Value |
| --- | --- |
| Checkpoint | `thinkingmachines/Inkling-2-Mini`, official NVFP4/MXFP8 (19.5 GB) + DSpark drafter `mtp.safetensors` (636 MB) |
| Engine | `vllm/vllm-openai:inkling-2-mini` (vLLM `0.1.dev25855+g7cb5dae22`, pinned by digest) + the `docker/` overlay |
| Parallelism | tensor parallel 2, NCCL all-reduce (`--disable-custom-all-reduce`) |
| Drafter | DSpark, 7 draft tokens per step (the only value the checkpoint accepts) |
| Context | 262,144 tokens; KV pool 312,417 tokens (bf16), `--gpu-memory-utilization 0.92` |
| API | `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice`; `reasoning_effort` none … max |
| Warmup | FlashInfer autotune off (`--kernel-config`) |

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
