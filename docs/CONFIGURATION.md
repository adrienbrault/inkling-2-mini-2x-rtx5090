# Configuration choices

All measured 2026-11-21 on two RTX 5090s with the same checkpoint. Sections below name the image when it is not the
served one (`sm120-r8`).

## How the served configuration got here

Spec-Bench per stream (median), one boot each, README protocol:

| Configuration | 1 stream | 4 streams | 8 streams | KV pool | Results |
| --- | ---: | ---: | ---: | ---: | --- |
| r4: first published (bf16 KV, vLLM's default sequence limit) | 351 | 205 | 138 | 312,417 | [r4](../bench/results/20261121T021530Z-inkling2-specbench-served-r4) |
| r5: decode products off cuBLAS, 16 sequences, bf16 KV | 373 | 225 | 171 | 481,529 | [bf16 boot](../bench/results/20261121T131647Z-inkling2-fp8kv) |
| r5 with the fp8 KV cache | 369 | 226 | 171 | 699,259 | [fp8 boot](../bench/results/20261121T131647Z-inkling2-fp8kv) |
| r8: + step budget + batch-sharded sampling (served) | 372 | 237 | 181 | 699,885 | [r8](../bench/results/20261121T222403Z-inkling2-specbench-served-r8) |

The r8 changes target concurrent traffic with long prompts. Median TTFT at four and eight streams went from 56 / 74 ms
(r5 fp8) to 55 / 70 ms on Spec-Bench, whose prompts are short. The scheduler budget matters most when a long prefill
arrives while other streams decode, as in agent sessions.

## fp8 KV cache

`--kv-cache-dtype fp8` stores the target's KV cache in unit-scale E4M3, which Inkling's SM120 attention reads natively.
The drafter's own cache stays bf16 (the overlay's drafter KV write covers the unquantized cache only).
[Results](../bench/results/20261121T131647Z-inkling2-fp8kv), two boots of image r5:

| | bf16 | fp8 |
| --- | ---: | ---: |
| KV pool | 481,529 tokens | 699,259 tokens |
| Spec-Bench per stream, 1 / 4 / 8 streams | 373 / 225 / 171 tok/s | 369 / 226 / 171 tok/s |
| Cold prefill at 8K / 246K | 18,577 / 8,680 tok/s | 18,160 / 8,559 tok/s |
| Mean NLL, 8,191-token code text | 0.4035 | 0.4025 |
| Mean NLL, 32,767-token code text | 0.3186 | 0.3197 |

Paired over the same tokens, fp8 minus bf16 is −0.00099 nats/token (standard error 0.00148) at 8K and +0.00109
(0.00073) at 32K. fp8 is not bitwise identical to bf16: greedy text diverges after 0–204 characters.

## Decode-aware step budget (`VLLM_DECODE_STEP_BUDGET`)

vLLM schedules up to 2,048 tokens per step here, so a long prefill runs in 2,048-token chunks. Every decoding stream
waits for each chunk. The overlay's `patch_sched.py` caps a step at `VLLM_DECODE_STEP_BUDGET` tokens while any running
request is decoding. A prefill with nobody decoding keeps the 2,048-token chunks, so a request served alone is
scheduled exactly as without the patch. Both alternatives cost something:

- **vLLM's global `--max-num-batched-tokens 1024`** caps every step. It changed greedy output on all three long prompts,
  because their prefill is cut differently, and cost 5–10 % cold prefill
  ([results](../bench/results/20261121T173337Z-inkling2-global-batch-1024)).
- **`--long-prefill-token-threshold`** caps each request's chunk. It kept output and prefill, but several prefilling
  requests still filled 2,048-token steps.

With the budget, greedy output on the same 11 prompts stays identical and cold prefill stays within 1 %
([results](../bench/results/20261121T192820Z-inkling2-stepbudget-identity)). The value 1,024 was chosen on a replay of
multi-turn agent sessions (6 and 10 concurrent sessions with long, mostly cached prompts). That replay is not in this
repository. At 512 per step, prefills took too long under load there.

## Batch-sharded sampling (`--enable-batch-sharded-sampling`)

At tensor parallel 2, vLLM gathers the full logits on both cards and samples every request on both. With batch-sharded
sampling, each request is sampled on one card from the two cards' vocabulary shards. The day-0 image disables it for
Inkling ("does not implement compute_logits_local"). The overlay's `patch_bss.py` adds that method: the same head and
soft-cap without the gather. The drafter's logits cache stays replicated. Spec-Bench at eight streams, paired boots, both
with the step budget: 167 / 169 → 182 / 180 tok/s per stream (+7.6 %), TTFT unchanged
([results](../bench/results/20261121T212103Z-inkling2-sharded-sampling-specbench)). Greedy output one request at a time is
identical.

## Output under concurrency

With several requests in flight, output is not reproducible on this engine. Which requests share a step changes the
kernels' arithmetic. Two passes of the same 22 concurrent requests on the same boot gave 0 of 22 identical outputs, in
every configuration tried. Measured by how long two passes' texts stay identical, sharded sampling diverges no sooner
than the configuration without it diverges from itself
([results](../bench/results/20261121T221734Z-inkling2-batched-parity), `bench/batched.py`). One request at a time,
greedy output is deterministic (`bench/greedy.py`).

## Drafter on or off

Spec-Bench with the README's protocol (forced 256 tokens), tensor parallel 2 with NCCL all-reduce both ways, the same
image: [drafter off](../bench/results/20261121T012150Z-inkling2-specbench-nospec-r3); drafter on gave 319 / 184 / 115
tok/s per stream and 347 / 719 / 997 aggregate at 1 / 4 / 8 streams on the same boot setup (before the PCIe all-reduce
below, which adds about 10 % to both).

| Streams | Off: per stream / aggregate / TTFT | On: per stream / aggregate / TTFT |
| ---: | ---: | ---: |
| 1 | 206 / 203 tok/s / 34 ms | 319 / 347 tok/s / 37 ms |
| 4 | 128 / 492 tok/s / 99 ms | 184 / 719 tok/s / 62 ms |
| 8 | 114 / 905 tok/s / 2.1 s | 115 / 997 tok/s / 81 ms |

The drafter helps most at low concurrency. The drafter-off run's median TTFT at eight streams was 2.1 s on its boot.

## All-reduce and one card versus two

Inkling's fused all-reduce needs NVLink multicast, which the PCIe-only RTX 5090 lacks, and vLLM's custom all-reduce has
no sm_120 code in the day-0 image, so tensor parallel 2 falls back to NCCL. Drafter off, Spec-Bench c1 / c4:

| Configuration | Per stream, 1 stream | Aggregate, 4 streams | Mean time per output token, 1 stream |
| --- | ---: | ---: | ---: |
| One card (tensor parallel 1, 32K context) | 177 tok/s | 418 tok/s | 5.67 ms |
| Two cards, NCCL | 206 tok/s | 492 tok/s | 4.92 ms |
| Two cards, FlashInfer PCIe IPC all-reduce | 222 tok/s | 546 tok/s | 4.52 ms |

Results: [one card](../bench/results/20261121T014555Z-inkling2-specbench-tp1-nospec-32k),
[PCIe IPC](../bench/results/20261121T015459Z-inkling2-specbench-tp2-nospec-pcieipc). Communication is about 0.4 ms of the
per-token time; the rest is spent inside each card (48 layer applications per token on the sm_120 Triton paths). With
the drafter, the PCIe all-reduce raises Spec-Bench from 319 / 184 / 115 to 351 / 205 / 138 tok/s per stream, with the
drafter's drafted and accepted token counts unchanged at one and four streams (the same token streams).

DSpark only accepts 7 draft tokens: the checkpoint's block size is 8, and vLLM refuses any other
`num_speculative_tokens`.

## Correctness of drafted output

Speculative decoding is lossless in exact arithmetic. On this engine, greedy output with the drafter is deterministic
for a request served alone (identical across runs and across server restarts) but is not token-identical to greedy output without it: the two
diverge after a few hundred characters. To check that the drafter is not the cause, `bench/teacher_check.py`
replays each greedy generation as a prompt, where prefill never uses the drafter,
and checks whether each generated token is the target's top choice:

| Server | Generated tokens | Not the prefill top choice | Median / max log-prob gap |
| --- | ---: | ---: | ---: |
| Drafter on | 3,781 | 136 (3.6 %) | 0.20 / 1.16 |
| Drafter off | 3,602 | 131 (3.6 %) | 0.17 / 1.24 |

The rate and the gap distribution are the same with and without the drafter: decode and prefill take different kernel
paths in this engine (NVFP4/MXFP8 GEMMs at different batch shapes), and near-ties flip either way. The drafter adds
nothing measurable on top. Its acceptance length (2.95 on Spec-Bench, 3.16 on long code output) also matches the
model card's. Raw: [teacher check](../bench/results/20261121T0022Z-inkling2-teacher-check).

The teacher check ran on image r3, before its patch guards were hardened (same kernels and patched code paths).

## One model copy per card

Data parallel 2 (one copy per card, tensor parallel 1) leaves 7.65 GiB of KV per card without the drafter, which fits a
94,208-token context, and 5.2 GiB with it. With the drafter, even a 45,056-token context needs 5.96 GiB (vLLM's startup
check), so one copy per card with the drafter does not fit a useful context on 32 GB cards. Tensor parallel 2 with the
drafter has 16.98 GiB of KV per card and serves the full 262,144 tokens.

## Unsupported on sm_120 in the day-0 image

Without the overlay and flags in `scripts/serve.sh`, these fail with `cudaErrorNoKernelImageForDevice`:

| Path | Where | Handling |
| --- | --- | --- |
| FlashInfer warmup autotune | `kernel_warmup.flashinfer_autotune` | `--kernel-config '{"enable_flashinfer_autotune":false}'` |
| DSpark KV-cache write | `reshape_and_cache_flash` from the drafter's `_write_kv` | Triton scatter (`docker/kv_write_sm120.py`) |
| Fused vocab-parallel embedding (TP 2) | `ops.vocab_parallel_embedding` | unfused path on SM12x (`docker/patch_embedding.py`) |
| Custom all-reduce (TP 2) | vLLM `custom_all_reduce` | `--disable-custom-all-reduce` |
