# SM120 overlay

vLLM's Inkling-2-Mini day-0 image (`vllm/vllm-openai:inkling-2-mini`, vLLM `0.1.dev25855+g7cb5dae22`) builds its compiled
core library (`_C_stable_libtorch`) for sm_80, sm_89, sm_90 and sm_100 only, with no PTX to fall back on. The RTX 5090
is sm_120. Inkling-2-Mini's own kernels are Triton and Gluon and run there, but three paths reach the compiled core and
fail with `cudaErrorNoKernelImageForDevice`:

| Path | Fix |
| --- | --- |
| The DSpark drafter writes its KV cache with `reshape_and_cache_flash` | `kv_write_sm120.py`: the same scatter as a Triton kernel; `patch_dspark.py` routes the drafter's `_write_kv` to it on SM12x with the unquantized cache |
| At tensor parallel 2, `VocabParallelEmbedding` uses the fused `ops.vocab_parallel_embedding` | `patch_embedding.py`: on SM12x, vLLM's unfused path (masked lookup, then all-reduce), which computes the same values |
| vLLM's custom all-reduce | `--disable-custom-all-reduce` in `scripts/serve.sh` (NCCL) |

FlashInfer's warmup autotune also profiles kernels with no sm_120 build; `scripts/serve.sh` turns it off.

Each patch script checks its input files' SHA-256 and requires a unique match for each replacement anchor. A base
package with different source files fails the build at these checks. The scatter covers the
unquantized KV cache only: on SM12x an fp8 drafter cache is refused at startup instead of reaching the missing kernel.
The target's cache can be fp8 (Inkling's own kernels write it); `scripts/serve.sh` keeps the drafter's in bf16 with
`"kv_cache_dtype":"auto"` in the speculative config.

## Patch order

Paths below are relative to the installed `vllm` package. `None` in the predecessor column means the patch hashes the clean base file. The Dockerfile copies `kv_write_sm120.py` and `decode_sm120.py` into `models/inkling2/nvidia/ops/` before applying the patches.

| Patch in Dockerfile order | Files edited | Predecessors whose output it hashes | Runtime switch or guard | Required or optional |
| --- | --- | --- | --- | --- |
| `patch_dspark.py` | `models/inkling2/nvidia/dspark.py` | None | SM12x DSpark KV scatter. Uses the unquantized drafter cache. | Required for DSpark on the pinned SM120 base. |
| `patch_bss.py` | `models/inkling2/nvidia/model.py`, `models/inkling2/speculator.py` | None | `--enable-batch-sharded-sampling` uses `compute_logits_local`. | Optional for basic serving. Required for batch-sharded sampling and by the draft-sharding build. |
| `patch_decode.py` | `models/inkling2/nvidia/loop.py`, `models/inkling2/nvidia/sconv.py` | None | Shape-selected SM12x bf16 GEMVs with cuBLAS fallback. No launch flag. | Optional performance patch. |
| `patch_sched.py` | `v1/core/sched/scheduler.py` | None | Positive `VLLM_DECODE_STEP_BUDGET` below the step budget caps chunks during decode. | Optional at runtime. Required by the conv-page build. |
| `patch_convpage.py` | `models/inkling/nvidia/sconv_swa_attn.py`, `v1/core/sched/scheduler.py` | `patch_sched.py` for the scheduler | `VLLM_INKLING_CONV_PAGE=1` with `--block-size 64` and the [cache and TP guards](../docs/CONFIGURATION.md#knobs). `VLLM_INKLING_CONVPAGE_PROBE=1` enables diagnostic logs. | Optional cache and diagnostics patch. |
| `patch_embedding.py` | `model_executor/layers/vocab_parallel_embedding.py` | None | SM12x disables the fused TP embedding path. | Required for the pinned SM120 TP2 path. |
| `patch_draftshard.py` | `models/inkling2/nvidia/dspark.py`, `models/inkling2/speculator.py`, `v1/worker/gpu/model_runner.py`. Hashes `v1/worker/gpu/sample/gumbel.py` and uses it to generate `v1/worker/gpu/sample/inkling_draftshard.py`. | `patch_dspark.py` for DSpark, `patch_bss.py` for the speculator | `VLLM_INKLING_DRAFT_SHARD=1` with a supported TP2 DSpark layout and batch-sharded sampling. | Optional performance patch. |
| `patch_lora.py` | `models/inkling2/nvidia/lora.py`, `models/inkling2/nvidia/model.py`. The build also checks vLLM's Punica metadata keeps its `no_lora_flag_cpu` contract. | `patch_bss.py` (hashes `model.py` after it) | `VLLM_INKLING_LORA_SKIP=1` skips the attention LoRA delta in steps with no adapter request; `VLLM_INKLING_LORA_HEAD=1` keeps the Inkling logits head when `lm_head` is not a LoRA target. Both only matter with `--enable-lora`. | Optional performance patch (r13). Do not combine with `--specialize-active-lora`. |

The guards expect each replacement anchor exactly once in the input. The conv-page patch also looks for the decode-budget code inserted by the scheduler patch. A changed or missing predecessor can stop the build even if its runtime switch is off. The Dockerfile parses selected Python files and checks markers for KV scatter, GEMV routing, local logits, scheduler budget, conv layout and probe telemetry. The final block checks local draft logits, the draft-sharding switch, the winner gather and runner cache materialization.

## Remove one patch

Use the optional `patch_draftshard.py` block for a first experiment. The later `patch_lora.py` block does not depend on it (it edits `lora.py` and `model.py`, which `patch_draftshard.py` leaves alone), so it can stay. Keep the checkpoint, frontend, KV dtype, DSpark and batch-sharded sampling fixed.

1. Boot the untouched r10 image with `scripts/serve.sh`. Run `scripts/check.sh`. Save `python3 bench/greedy.py runs/r10/greedy.json --long` and `python3 bench/greedy.py runs/r10/seeded.json --long --seeded` after creating `runs/r10`. Run Spec-Bench with `bash bench/specbench.sh runs/r10/specbench`. Keep the launch settings and logs.
2. Stop the server. Remove or comment out the final Dockerfile block from `ARG VLLM_DIR` through `rm /tmp/patch_draftshard.py`, including its `COPY`, `RUN` and all marker checks. Build with `IMAGE=inkling2-rtx5090:sm120-test scripts/build-image.sh`.
3. Boot with `IMAGE=inkling2-rtx5090:sm120-test VLLM_INKLING_DRAFT_SHARD=0 scripts/serve.sh`. Repeat the checks into `runs/no-draft-shard`. Compare content, reasoning, finish reason and completion token count for every prompt in both captures. Compare Spec-Bench acceptance and throughput to the baseline. For a runtime-only control, boot the untouched r10 image with `VLLM_INKLING_DRAFT_SHARD=0`.

## Move to a newer base image

1. Record the new base digest, vLLM version, Python package location, entrypoint and SM120 kernel coverage. Extract the clean files in the table and diff them against the pinned base. Check for fixes already upstream.
2. Apply the surviving patches in Dockerfile order. Inspect each patch's input after its predecessors run. Update hashes and anchors against that input. Update the build markers when the checked code changes.
3. Build under a new tag. Boot it and run `scripts/check.sh` and both lone-request captures with the same settings as r10. Use the [reproduction commands](../docs/REPRODUCE.md) for benchmark comparisons. Repeat concurrent comparisons on the baseline to measure normal output and timing variation. Rerun the relevant GPU gates if available. The checkout includes [conv-page gate captures](../bench/results/20261122T032748Z-inkling2-r10-combined) and [draft-sharding gate captures](../bench/results/20261122T012901Z-inkling2-draftshard). Save baseline and candidate flags, results and image digests.

## Decode products moved off cuBLAS

Two small bf16 products in Inkling's decode step run on cuBLAS, which picks sm_80 WMMA kernels on the RTX 5090 and spends
17 to 18 µs on a few kilobytes of work. `patch_decode.py` routes them to Triton kernels whose results are bitwise
identical to cuBLAS on these shapes. The microbenchmark ran 20 random trials per shape (2026-11-21).

| Product (per card at tensor parallel 2) | cuBLAS | Overlay | Where it applies |
| --- | ---: | ---: | --- |
| Loop controllers' down projection, (S, T, 4608) × (S, 64, 4608) | 16.6–18.4 µs | 6.8–7.2 µs, one launch for all streams (`decode_sm120.stream_gemv`) | 2–4 streams, 2–16 tokens |
| Short-conv dynamic taps, (T, 2304) × (32, 2304) | 9.6–10 µs | 4.0–4.1 µs (Inkling's own `sm12x_bf16_gemv`) | 1–16 tokens |

Outside these ranges cuBLAS takes other kernels whose rounding the Triton kernels do not reproduce, so the overlay keeps
cuBLAS there. A decode step at one stream verifies 8 tokens and is inside the ranges. At four or more streams the step
holds 32 or more tokens and keeps cuBLAS.

Paired boots of the old and new images on the served configuration (A B A B, Spec-Bench, 2026-11-21):
- one stream: 352.6 / 351.3 → 374.8 / 374.1 tok/s per stream (+6.4 %);
- four streams: unchanged;
- greedy text identical (8 of 8 prompts on each boot);
- drafter counts identical on all four boots.

Measured and not adopted:
- **A fused one-pass logit soft-cap.** It is not bitwise equal to PyTorch's three passes.
- **Shorter DSpark drafts (3 to 6 tokens).** At one stream they are 3–5 % slower. At four and eight streams they gain 2–10 %, within run-to-run variation, and they change greedy text.

## Scheduler, sampling and cache layout (r7 to r10)

| Patch | What it changes | Enabled by |
| --- | --- | --- |
| `patch_sched.py` (vLLM's V1 scheduler) | while any running request is decoding, a step carries at most `VLLM_DECODE_STEP_BUDGET` tokens; a prefill with nobody decoding keeps the full 2,048 | `-e VLLM_DECODE_STEP_BUDGET=1024`; unset or 0 = off |
| `patch_bss.py` (Inkling's model and drafter) | adds `compute_logits_local` (the same head and soft-cap without the TP gather), so vLLM's batch-sharded sampling engages; the drafter's logits cache stays replicated | `--enable-batch-sharded-sampling` |
| `patch_convpage.py` (Inkling's short-conv layer, vLLM's scheduler) | the short-conv state gets eight-token pages of its own size (4 heads × 8 tokens × 1,408 bf16 values = 88 KiB) in vLLM's block-outer `BLNHC` layout, instead of pages sized like the attention pages; attention pages and prefix-cache granularity go from 128 to 64 tokens, and the KV pool grows from 699,885 to 819,427 tokens (818,388 with draft sharding's buffers, as served) | `-e VLLM_INKLING_CONV_PAGE=1` with `--block-size 64` |
| `patch_draftshard.py` (the DSpark drafter and vLLM's GPU sampler) | each card draws the drafter's sampled token from its own vocabulary shard and the cards exchange one small winner packet per draft position, instead of all-gathering the full-vocabulary draft logits seven times per step | `-e VLLM_INKLING_DRAFT_SHARD=1` (with `--enable-batch-sharded-sampling`) |
| `patch_lora.py` (Inkling's attention LoRA hook and logits head) | with the abliteration adapter loaded, a step with no adapter request skips the LoRA delta (a zero buffer and an add on 44 projections) and keeps Inkling's own logits head, so `inkling-2-mini` runs as without LoRA: same output, speed within noise (2026-11-22, [`r13-ab`](../bench/results/20261122T182215Z-inkling2-r13-ab)) | `-e VLLM_INKLING_LORA_SKIP=1 -e VLLM_INKLING_LORA_HEAD=1` (set by `scripts/serve.sh` with the adapter) |

Each is inert without its setting; with it, output is bitwise identical to the overlay without it (checked on
greedy and seeded sampled text, and by GPU gates on the patched kernels). Measurements: [`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).

## Credits

The idea of moving the loop-controller and short-conv products off cuBLAS on SM120 comes from the RTX PRO 6000 recipe
at [hyudryu/Inkling-2-Mini-2xRTXPro6000](https://github.com/hyudryu/Inkling-2-Mini-2xRTXPro6000), which also reports
short-block DSpark drafting. The kernels and patches here are independent implementations, checked bitwise against
cuBLAS on this hardware.

`patch_bss.py`'s `compute_logits_local` follows vLLM's own implementations for Llama and Qwen3 (Apache-2.0); the
batch-sharded sampler is vLLM's. The decode-aware step budget, the exact-size conv pages and the draft sharding are this repository's own.
