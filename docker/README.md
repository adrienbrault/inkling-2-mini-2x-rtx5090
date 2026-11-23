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
| `patch_ropetable.py` | `models/inkling2/nvidia/dspark.py`, `models/inkling2/nvidia/ops/dspark.py` | `patch_dspark.py` and `patch_draftshard.py` (hashes `dspark.py` after them) | `VLLM_INKLING_DRAFT_ROPE_ONTHEFLY=1` at model load. | Optional memory patch (r14). |

The guards expect each replacement anchor exactly once in the input. The conv-page patch also looks for the decode-budget code inserted by the scheduler patch. A changed or missing predecessor can stop the build even if its runtime switch is off. The Dockerfile parses selected Python files and checks markers for KV scatter, GEMV routing, local logits, scheduler budget, conv layout and probe telemetry. The draft-sharding block checks local draft logits, its runtime switch, the winner gather and runner cache materialization. Subsequent blocks check LoRA and RoPE markers.

## Draft-sharding A/B

Use the same r14 image and leave the checkpoint, adapter, frontend, target KV dtype, DSpark, RoPE and batch-sharded sampling fixed. For the second boot, stop the first and run `VLLM_INKLING_DRAFT_SHARD=0 scripts/serve.sh`. Run `scripts/check.sh`, the lone-request captures in [`bench/greedy.py`](../bench/greedy.py) with `--long` (also with `--seeded`), and [Spec-Bench](../bench/specbench.sh) into separate report directories. Compare final outputs and counters.

### Removing the source patches

Keep the shared `ARG VLLM_DIR` near the top of the Dockerfile. Remove the r10 `COPY` and `RUN` for `patch_draftshard.py`, including its marker checks. The r13 LoRA block can stay. Either remove the r14 `COPY` and `RUN` for `patch_ropetable.py` too, or rebase its expected SHA-256 and anchors for `models/inkling2/nvidia/dspark.py` onto the output of `patch_dspark.py` alone. Its current guard expects the post-draft-shard file. Keep the final copy of the licence and credits.

Build with `IMAGE=inkling2-rtx5090:sm120-test scripts/build-image.sh`. Launch that tag with `VLLM_INKLING_DRAFT_SHARD=0` and, if the r14 block was removed, set `VLLM_INKLING_DRAFT_ROPE_ONTHEFLY=0` on both the test and control boots. This source-removal variant has no archived r14 measurement.

## Move to a newer base image

1. Record the new base digest, vLLM version, Python package location, entrypoint and SM120 kernel coverage. Extract the clean files in the table and diff them against the pinned base. Check for fixes already upstream.
2. Apply the surviving patches in Dockerfile order. Inspect each patch's input after its predecessors run. Update hashes and anchors against that input. Update the build markers when the checked code changes.
3. Build under a new tag. Boot it and run `scripts/check.sh` and both lone-request captures with the same settings as the r14 control. Use the [reproduction commands](../docs/REPRODUCE.md) for benchmark comparisons. Repeat concurrent comparisons on the baseline to measure normal output and timing variation. Rerun the relevant GPU gates if available. The checkout includes [conv-page gate captures](../bench/results/20261122T032748Z-inkling2-r10-combined) and [draft-sharding gate captures](../bench/results/20261122T012901Z-inkling2-draftshard). Save baseline and candidate flags, results and image digests.

## Decode products moved off cuBLAS

The overlay routes the loop controllers' bf16 down projection to a Triton kernel on selected SM12x decode shapes and uses Inkling's own GEMV for short-conv dynamic taps. The code keeps the cuBLAS paths outside the selected shapes. See [`patch_decode.py`](patch_decode.py) and [`decode_sm120.py`](decode_sm120.py).

## Scheduler, sampling and cache layout (r7 to r14)

| Patch | What it changes | Enabled by |
| --- | --- | --- |
| `patch_sched.py` (vLLM's V1 scheduler) | while any running request is decoding, a step carries at most `VLLM_DECODE_STEP_BUDGET` tokens; a prefill with nobody decoding keeps the full 2,048 | `-e VLLM_DECODE_STEP_BUDGET=1024`; unset or 0 = off |
| `patch_bss.py` (Inkling's model and drafter) | adds `compute_logits_local` (the same head and soft-cap without the TP gather), so vLLM's batch-sharded sampling engages; the drafter's logits cache stays replicated | `--enable-batch-sharded-sampling` |
| `patch_convpage.py` (Inkling's short-conv layer, vLLM's scheduler) | the short-conv state gets eight-token pages of its own size (4 heads × 8 tokens × 1,408 bf16 values = 88 KiB) in vLLM's block-outer `BLNHC` layout, instead of pages sized like the attention pages; attention pages and prefix-cache granularity go from 128 to 64 tokens, and the historical r8/r10 A/B logged a rise from 699,885 to 818,388 KV tokens with compact conv pages and draft sharding | `-e VLLM_INKLING_CONV_PAGE=1` with `--block-size 64` |
| `patch_draftshard.py` (the DSpark drafter and vLLM's GPU sampler) | each card draws the drafter's sampled token from its own vocabulary shard and the cards exchange one small winner packet per draft position, instead of all-gathering the full-vocabulary draft logits seven times per step | `-e VLLM_INKLING_DRAFT_SHARD=1` (with `--enable-batch-sharded-sampling`) |
| `patch_lora.py` (Inkling's attention LoRA hook and logits head) | with the abliteration adapter loaded, a step with no adapter request skips the LoRA delta (a zero buffer and an add on 44 projections) and keeps Inkling's own logits head, so `inkling-2-mini` runs as without LoRA: same output, speed within noise (2026-11-22, [`r13-ab`](../bench/results/20261122T182215Z-inkling2-r13-ab)) | `-e VLLM_INKLING_LORA_SKIP=1 -e VLLM_INKLING_LORA_HEAD=1` (set by `scripts/serve.sh` with the adapter) |
| `patch_ropetable.py` (the DSpark drafter's RoPE) | the drafter computes cos/sin from the position in its RoPE kernels from a 256-byte inverse-frequency vector instead of an fp32 `[window, 128]` table: removing a 512 MiB table per card at the 1M window. The A/B recorded 819 more KV blocks per card and matching one-stream draft counters (2026-11-22, [`r14-ab`](../bench/results/20261122T203506Z-inkling2-r14-ab)) | `-e VLLM_INKLING_DRAFT_ROPE_ONTHEFLY=1` |

The historical r8/r10 comparison matched 11 lone-request outputs with greedy and fixed-seed sampling. The r14 comparison matched 11 outputs in each sampling mode and the one-stream counters. The archived metrics do not capture individual draft tokens. [r8/r10 result](../bench/results/20261122T032748Z-inkling2-r10-combined), [r13/r14 result](../bench/results/20261122T203506Z-inkling2-r14-ab/summary.json)

## Credits

The [RTX PRO 6000 recipe](https://github.com/hyudryu/Inkling-2-Mini-2xRTXPro6000) inspired moving the loop-controller and short-conv products off cuBLAS on SM120. The [third-party credits](../THIRD_PARTY.md) identify the vLLM and Inkling code in the overlay, including the Llama and Qwen3 logits path, Inkling GEMV and the generated Gumbel sampler. The image build copies the project's licence texts and credits into `/usr/share/doc/inkling-overlay/`.
