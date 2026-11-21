# Configuration choices

All measured 2026-11-21 on two RTX 5090s with the same checkpoint and image.

## Drafter on or off

Spec-Bench (same protocol as the README; [drafter off](../bench/results/20261121T002644Z-inkling2-specbench-nospec-tp2),
[drafter on](../bench/results/20261121T003545Z-inkling2-specbench-served-tp2-dspark)), tensor parallel 2 both ways:

| Streams | Off: per stream / aggregate | On: per stream / aggregate |
| ---: | ---: | ---: |
| 1 | 202 / 203 tok/s | 319 / 343 tok/s |
| 4 | 124 / 469 tok/s | 187 / 732 tok/s |
| 8 | 108 / 796 tok/s | 136 / 1,060 tok/s |

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
nothing measurable on top. Its acceptance length (2.88 on Spec-Bench, 3.16 on long code output) also matches the
model card's.

## One model copy per card

Without the drafter, one copy per card (data parallel 2) fits a 94,208-token context and measured 145 / 497 / 742 tok/s
at 1 / 4 / 8 streams on a single coding prompt. With the drafter, each card has 5.2 GiB left for KV, and even a
45,056-token context needs 5.96 GiB, so data parallel with the drafter does not fit a useful context on 32 GB cards.

## Unsupported on sm_120 in the day-0 image

Without the overlay and flags in `scripts/serve.sh`, these fail with `cudaErrorNoKernelImageForDevice`:

| Path | Where | Handling |
| --- | --- | --- |
| FlashInfer warmup autotune | `kernel_warmup.flashinfer_autotune` | `--kernel-config '{"enable_flashinfer_autotune":false}'` |
| DSpark KV-cache write | `reshape_and_cache_flash` from the drafter's `_write_kv` | Triton scatter (`docker/kv_write_sm120.py`) |
| Fused vocab-parallel embedding (TP 2) | `ops.vocab_parallel_embedding` | unfused path on SM12x (`docker/patch_embedding.py`) |
| Custom all-reduce (TP 2) | vLLM `custom_all_reduce` | `--disable-custom-all-reduce` |
