# Configuration choices

All measured 2026-11-21 on two RTX 5090s with the same checkpoint and image.

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
(identical across runs and across server restarts) but is not token-identical to greedy output without it: the two
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

The teacher check ran on the image before its patch guards were hardened (same kernels and patched code paths).

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
