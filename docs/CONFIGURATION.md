# Configuration choices

All measured 2026-11-21 on two RTX 5090s with the same checkpoint and image.

## Drafter on or off

Spec-Bench with the README's protocol (forced 256 tokens), tensor parallel 2 both ways, the same image:
[drafter off](../bench/results/20261121T012150Z-inkling2-specbench-nospec-r3),
[drafter on](../bench/results/20261121T010840Z-inkling2-specbench-served-r3).

| Streams | Off: per stream / aggregate / TTFT | On: per stream / aggregate / TTFT |
| ---: | ---: | ---: |
| 1 | 206 / 203 tok/s / 34 ms | 319 / 347 tok/s / 37 ms |
| 4 | 128 / 492 tok/s / 99 ms | 184 / 719 tok/s / 62 ms |
| 8 | 114 / 905 tok/s / 2.1 s | 115 / 997 tok/s / 81 ms |

The drafter helps most at low concurrency. At eight streams per-stream decode is the same; the drafter-off run's
median TTFT at c8 was 2.1 s on this boot.

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
