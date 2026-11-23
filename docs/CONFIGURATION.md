# Configuration choices

The served image is `sm120-r14`. The headline measurements are from r13 on two RTX 5090s, and the older comparisons retain their original image revisions. See the [measurement boot](REPRODUCE.md#measurement-boot) and [r14 A/B](../bench/results/20261122T203506Z-inkling2-r14-ab/summary.json).

## Knobs

Set these variables before running [`scripts/serve.sh`](../scripts/serve.sh). It prints every effective value. The defaults below describe the served TP2 profile. A switch set to `0` omits the associated Docker environment variable or serving flag, except for the decode budget, which passes `0` to the scheduler.

| Variable or flag | Default | Effect | Cost | Tested values | What breaks |
| --- | --- | --- | --- | --- | --- |
| `MODEL_DIR` | Required | Mounts the official checkpoint read-only. | Disk for weights and the drafter. | Official Inkling-2-Mini checkpoint. | Needs an absolute path, readable config and tokenizer files, target weight shards and `mtp.safetensors`. |
| `CACHE_DIR` | `$HOME/.cache/inkling2-vllm` | Mounts a writable cache and persists Triton kernels between boots. | Disk and container-written files. | Cache mount used by the historical recipe. | An inaccessible path stops launch. An empty cache recompiles the kernels. |
| `IMAGE` | `inkling2-rtx5090:sm120-r14` | Selects the local serving image. | A new build may pull the base image and recompile kernels at startup. | Headline r13, RoPE A/B r13 and r14. | Build with the same tag before serving. Other builds may lack the required SM120 paths. |
| `PORT` | `8000` | Sets the published host port. The container listens on 8000. | Update client URLs when it changes. | Host ports `8000` and `8022` appear in the recipe and archived launch commands. | Docker fails to publish an occupied port. |
| `BIND` | `127.0.0.1` | Sets the host IPv4 address. Use `0.0.0.0` for all interfaces. | Exposing the port gives reachable clients access to the API. | Untested as a launcher override. | Docker must be able to bind the address. The recipe provides no API key. |
| `GPUS` | `0,1` | Selects devices by index and derives tensor parallelism from their count. | TP2 communicates over PCIe and leaves more room for KV. | TP2 on two RTX 5090s. TP1 at 32K without DSpark on an earlier image. | Unknown device IDs fail in Docker. TP1 requires `DSPARK=0` and disables TP-only switches. |
| `MAX_MODEL_LEN` | `1048576` | Caps the total prompt plus output tokens per request. | A request near 1M slows other traffic while prefilling and decoding. | r10 compared 1M and 262,144. The r14 A/B used 1M. [Window results](../README.md#1m-window) | Startup rejects a window that cannot fit. Prompts beyond the window get HTTP 400. |
| `MAX_NUM_SEQS` | `16` | Caps concurrently scheduled sequences. Admission also needs enough free KV. | Larger limits can change activation and CUDA graph reserves. | 16 on r13 and in the r14 A/B setup. Earlier runs also used the engine default. | If VRAM is insufficient, try a shorter window and a smaller sequence cap. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | Sets the vLLM device memory fraction. | A higher fraction reduces the headroom available to other work. | 0.92 in the [r13 memory capture](../bench/results/20261122T200911Z-inkling2-kv-capacity-r13/boot-memory.txt). | The launcher accepts values above 0 and up to 1. Startup and workload memory depend on the configuration. |
| `ABLIT_DIR` | `MODEL_DIR-ablit` when that directory exists | Mounts the abliteration LoRA read-only and adds the request name `inkling-2-mini-ablit`. Empty turns it off. | With overlay r13 (`VLLM_INKLING_LORA_SKIP`, `VLLM_INKLING_LORA_HEAD`, set automatically), requests to `inkling-2-mini` run as without LoRA: same output, speed within noise. The adapter uses a second set of CUDA graphs. | The pinned OpenUncensored rank-2 adapter on `attn.wo_ud` (`scripts/fetch-adapter.sh`). | `--specialize-active-lora` must stay off: its capture drops the adapter in batches that mix both names. |
| `DSPARK` | `1` | Enables DSpark with 7 draft tokens, probabilistic drafts and `auto` drafter KV. | Drafter weights and KV take memory. Low-concurrency decode was faster with DSpark in the historical comparison. | On in the r13 headline runs and the r14 A/B. Off in earlier TP1 and TP2 boots. | With DSpark off, the defaults also drop conv pages, draft sharding and in-kernel RoPE. The launcher still requires the full checkpoint, including `mtp.safetensors`. |
| `KV_CACHE_DTYPE` | `fp8` | Target KV accepts `fp8` or `auto`. `auto` lets the engine select the dtype and disables conv pages by default. | bf16 held a smaller pool and produced different greedy text in the r5 comparison. | fp8 on r13 and r14 A/B. bf16 on r5. Confirm the resolved dtype in the log for `auto`. | Compact conv pages require fp8 target KV. |
| `VLLM_INKLING_CONV_PAGE` | `1` | Selects compact short-conv pages and passes `--block-size 64`. | Disabling it gives up the pool gain. | On in r9/r10. Off in the r8 baseline. | Requires SM120 TP2, the checkpoint's conv geometry, fp8 target and bf16 DSpark KV, hybrid KV management, BLNHC or automatic layout, and no KV connector, PP or CP. |
| `VLLM_INKLING_DRAFT_SHARD` | `1` | Samples draft tokens from the two local vocabulary shards. | Sharded sampling adds buffers. See [the historical A/B](#conv-pages-and-draft-sharding-overlay-r10). | `0` and `1` in the paired draft-sharding boots. | Needs DSpark, TP2 and batch-sharded sampling. Unsupported communicator, head or draft layouts can fall back to dense drafting. Check for `Inkling2 DSpark vocab-sharded draft sampling enabled (TP2).` |
| `VLLM_INKLING_DRAFT_ROPE_ONTHEFLY` | `1` with DSpark | r14 computes the drafter's RoPE cos/sin in its kernels. | 819 more KV blocks per card at 1M, with matched lone-request outputs and one-stream draft counters. | r13 table and r14 in-kernel RoPE on the same window. [A/B](../bench/results/20261122T203506Z-inkling2-r14-ab/summary.json) | Disabling this switch brings back the full RoPE table. |
| `VLLM_DECODE_STEP_BUDGET` | `1024` | Caps a step while a request decodes. `0` disables the cap. | Smaller budgets can slow prefills under load. | Off and 1024 in the r8 comparison. | The overlay expects an integer. A value at or above the current step budget has no effect. |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` | `1` | Enables FlashInfer PCIe IPC for TP all-reduce with NCCL fallback. | FlashInfer improved decode by about 10% over NCCL in the measured comparison. | Enabled and omitted on two RTX 5090s. See [all-reduce](#all-reduce-and-one-card-versus-two). | Requires TP2 in this launcher. Keep `--disable-custom-all-reduce` on the pinned base. |
| `VLLM_USE_RUST_FRONTEND` | `1` | Selects the Rust frontend. `0` keeps Python. | Python adds about 0.1 s to median TTFT on long replay prompts. The frontends differ in reported trailing reasoning whitespace. | Both in the [paired boots](#rust-frontend). | Keep the frontend fixed when comparing TTFT or exact reasoning text. |
| `FAIR_PREFILL` / `--long-prefill-token-threshold 768` and `--long-prefill-token-threshold-adaptive` | `1` | Caps each prefill chunk when multiple requests prefill. `0` drops both flags. | Faster newcomer TTFT with 2–4% lower aggregate decode at 10 agent sessions in the paired experiment. | The pair on and off. See [fair prefill](#fair-prefill---long-prefill-token-threshold-768---long-prefill-token-threshold-adaptive). | A single flag changes the experiment. The 768 threshold and adaptive mode were tested together. |
| `BATCH_SHARDED_SAMPLING` / `--enable-batch-sharded-sampling` | `1` | Samples each target request on one TP rank from both vocabulary shards. | Turning it off gathers full logits and disables draft sharding. | The flag on and off in the [r8 comparison](#batch-sharded-sampling---enable-batch-sharded-sampling-image-r8). | Needs TP2 and `compute_logits_local` from the overlay. |
| `VLLM_INKLING_CONVPAGE_PROBE` | `0` | Enables diagnostic queue, scheduler and KV layout messages for probe requests. | Adds diagnostic log output. | Used by [`bench/kv_probe.py`](../bench/kv_probe.py) for admission probes. | Admission analysis needs a readable, continuously captured log with the probe events. |
| `TRITON_CACHE_DIR` | `/root/.cache/triton` | Fixed container path inside the `CACHE_DIR` mount. | Cold starts compile kernels. | The launcher persists the cache between boots. | Losing the persisted cache makes subsequent boots recompile. |

For scheduler telemetry, launch with `VLLM_INKLING_CONVPAGE_PROBE=1`, then follow the [admission commands](REPRODUCE.md#admission-diagnostics).

## Compatibility profiles

Start from the repository root with `MODEL_DIR` exported as in [Quick start](../README.md#quick-start). Stop the current container before starting the next profile.

| Profile | Block size and switches | Evidence |
| --- | --- | --- |
| Default TP2 with DSpark | Block size 64 with compact conv pages. | r13 headline and r14 A/B settings when both names are loaded. |
| TP2 without DSpark | vLLM selects the block size. The default conv pages, draft sharding and drafter RoPE switches are off. | Earlier TP2 images. This r14 profile is untested on GPUs. |
| One card, 32K, without DSpark | vLLM selects the block size. Compact conv pages, draft sharding, drafter RoPE, batch-sharded sampling and FlashInfer PCIe IPC are off. | [Historical TP1](../bench/results/20261121T014555Z-inkling2-specbench-tp1-nospec-32k/server.txt) used a 32,768-token window, the default target KV dtype and the base name only. r14 TP1 is untested on GPUs. |

```bash
scripts/serve.sh

DSPARK=0 ABLIT_DIR= scripts/serve.sh

GPUS=0 DSPARK=0 ABLIT_DIR= MAX_MODEL_LEN=32768 KV_CACHE_DTYPE=auto VLLM_USE_RUST_FRONTEND=0 \
  VLLM_DECODE_STEP_BUDGET=0 FAIR_PREFILL=0 scripts/serve.sh
```

For TP2 with the engine's target KV dtype, use `KV_CACHE_DTYPE=auto scripts/serve.sh` and verify the dtype at startup. The bf16 comparison below used r5. This r14 profile is untested on GPUs. All profiles keep custom all-reduce and FlashInfer warmup autotune disabled for SM120.

## Reasoning effort

The recipe records `none` and `max` as the endpoints. It exercises `none` in GSM8K and uses the checkpoint default in agent replay. The archived results do not contain the template's complete accepted-value list or its literal default. Check the selected checkpoint's `chat_template.jinja` or the `chat_template` field in `tokenizer_config.json` for intermediate values. The launcher leaves the default to that template.

Send the effort in `chat_template_kwargs`, as shown in [Quick start](../README.md#quick-start). The [headline benchmark](../bench/headline_inkling.py) also sets it for tokenization.

Set this field in your client for requests that should use the same effort. Even `none` can spend generation tokens on reasoning. Leave enough output budget for the answer. The [r13 GSM8K reference](../bench/results/20261122T192343Z-inkling2-quality-r13) uses 4,096 generation tokens and a non-matching stop string.

## Historical overlay progression

Spec-Bench per stream (median), one boot each, README protocol:

| Configuration | 1 stream | 4 streams | 8 streams | KV pool | Results |
| --- | ---: | ---: | ---: | ---: | --- |
| r4: first published (bf16 KV, vLLM's default sequence limit) | 351 | 205 | 138 | 312,417 | [r4](../bench/results/20261121T021530Z-inkling2-specbench-served-r4) |
| r5: decode products off cuBLAS, 16 sequences, bf16 KV | 373 | 225 | 171 | 481,529 | [bf16 boot](../bench/results/20261121T131647Z-inkling2-fp8kv) |
| r5 with the fp8 KV cache | 369 | 226 | 171 | 699,259 | [fp8 boot](../bench/results/20261121T131647Z-inkling2-fp8kv) |
| r8: + step budget + batch-sharded sampling | 372 | 237 | 181 | 699,885 | [r8](../bench/results/20261121T222403Z-inkling2-specbench-served-r8) |
| r10: + conv pages + draft sharding | 376 | 242 | 188 | 818,388 | [r10](../bench/results/20261122T053112Z-inkling2-specbench-served-r10) |

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

## Decode-aware step budget (`VLLM_DECODE_STEP_BUDGET`, image r8)

vLLM schedules up to 2,048 tokens per step here, so a long prefill runs in 2,048-token chunks. Every decoding stream
waits for each chunk. The overlay's `patch_sched.py` caps a step at `VLLM_DECODE_STEP_BUDGET` tokens while any running
request is decoding. A prefill with nobody decoding keeps the 2,048-token chunks, so a request served alone is
scheduled exactly as without the patch. Both alternatives cost something:

- **vLLM's global `--max-num-batched-tokens 1024`** caps every step. It changed greedy output on all three long prompts,
  because their prefill is cut differently, and cost 5–10 % cold prefill
  ([results](../bench/results/20261121T173337Z-inkling2-global-batch-1024)).
- **`--long-prefill-token-threshold`** caps each request's chunk. It kept output and prefill, but several prefilling
  requests still filled 2,048-token steps.

With the budget, greedy output matched on the same 11 prompts and cold prefill stayed within 1 %. [r8 results](../bench/results/20261121T192820Z-inkling2-stepbudget-identity)

## Fair prefill (`--long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive`)

This is the historical r10 comparison, with a 262,144-token window, the base name and the Python frontend.

With the step budget alone, one long cold prompt takes most of every step until it is done, so a short request that arrives behind it waits for the whole prefill: 20 s behind a 196K-token prompt. With `-adaptive`, vLLM caps each prefilling request's chunk at max(768, 2,048 / number of prefilling requests) only while more than one request is prefilling. A request served alone is scheduled exactly as before, so its output and cold prefill rate do not change.

On four boots in the order A G A G ([results](../bench/results/20261122T092118Z-inkling2-fair-prefill-768)), the short newcomer's first token came after 0.5 s instead of 20.4 s and a 16K newcomer's after 6.7 s instead of 21.5 s. The long prompt took 1.16x its time alone instead of 1.07x. Greedy output on 11 prompts stayed identical and cold prefill stayed within 1 %. The agent replay's aggregate decode at 10 sessions was 2–4 % lower (both G boots below both A boots); at 6 sessions it was within the A-to-A spread.

## Rust frontend

`VLLM_USE_RUST_FRONTEND=1` replaces vLLM's Python API server with its Rust one; the engine is unchanged. On four historical r10 boots in the order A F A F (A: Python frontend, F: Rust frontend; [results](../bench/results/20261122T133833Z-inkling2-rustfe-ab)), the agent replay's median first token was 0.08–0.10 s sooner on both F boots at 6 and 10 sessions, with long prompts (median about 85–90K tokens) whose request handling the Python frontend does per request. Throughput stayed within the A-to-A spread.

| Boot | 6 sessions: aggregate, first token p50 / p90 | 10 sessions: aggregate, first token p50 / p90 | Spec-Bench 8 / 16 streams aggregate |
| --- | ---: | ---: | ---: |
| A1 | 517 tok/s, 0.26 / 0.71 s | 695 tok/s, 0.32 / 0.84 s | 1,586 / 2,420 tok/s |
| F1 | 533 tok/s, 0.16 / 0.58 s | 703 tok/s, 0.23 / 0.73 s | 1,610 / 2,410 tok/s |
| A2 | 514 tok/s, 0.23 / 0.70 s | 708 tok/s, 0.30 / 0.87 s | 1,611 / 2,456 tok/s |
| F2 | 517 tok/s, 0.15 / 0.55 s | 706 tok/s, 0.20 / 0.79 s | 1,604 / 2,417 tok/s |

At temperature 0 the 11 `bench/greedy.py --long` outputs had the same content, reasoning text and completion token counts on all four boots, except one long prompt whose reasoning ends in `"\n  "` on the F boots: the Rust frontend returns the last whitespace characters of the reasoning that the Python frontend drops. Its `usage` also omits the empty `prompt_tokens_details.multimodal_tokens` field. The tool-call, image and sampled smoke checks passed on all four boots, and tool-eval on the historical r10 Rust boot scored 83.5 ± 3.0 (82.8 ± 2.5 with the Python frontend).

## Batch-sharded sampling (`--enable-batch-sharded-sampling`, image r8)

At tensor parallel 2, vLLM gathers the full logits on both cards and samples every request on both. With batch-sharded
sampling, each request is sampled on one card from the two cards' vocabulary shards. The day-0 image disables it for
Inkling ("does not implement compute_logits_local"). The overlay's `patch_bss.py` adds that method: the same head and
soft-cap without the gather. The drafter's logits cache stays replicated. Spec-Bench at eight streams, paired boots, both
with the step budget: 167 / 169 → 182 / 180 tok/s per stream (+7.6 %), TTFT unchanged
([results](../bench/results/20261121T212103Z-inkling2-sharded-sampling-specbench)). One request at a time, output is
identical with and without it: at temperature 0, and at temperature 1.0 with a fixed seed per request, on 11 prompts
over four boots ([results](../bench/results/20261121T231722Z-inkling2-seeded-identity)).

## Conv pages and draft sharding (overlay r10)

The historical r10 A/B covered these two changes. The captured lone-request outputs matched on 11 prompts:

- **Conv pages** (`VLLM_INKLING_CONV_PAGE=1` with `--block-size 64`, `patch_convpage.py`). Inkling's short-conv layers
  keep a small rolling state per request. The day-0 image stores it in pages sized like the attention pages, so most of
  each conv page is padding. The overlay gives it eight-token pages of its own size in vLLM's block-outer layout. The
  GPU gate compares every kernel that reads or writes the state with the unpatched layout: unaligned and chunked
  prefill, block boundaries, decode with verification, prefix copies and CUDA-graph replay, 20 trials each.
- **Draft sharding** (`VLLM_INKLING_DRAFT_SHARD=1`, `patch_draftshard.py`). DSpark proposes 7 tokens per step, and for
  each it gathered the full-vocabulary draft logits from both cards. With the overlay each card samples from its own
  vocabulary shard (the Gumbel noise is keyed by global token id, so it is the same noise), and the cards exchange one
  small packet each with their winner. The two-card gate checks that draft token ids and the verifier's inputs are
  bitwise equal for greedy and probabilistic drafting (fp32 and fp64 noise, mixed batches), eager and in CUDA-graph
  replay, 20 trials each.

Paired boots (A C A C), r8 against r10 with both on
([results](../bench/results/20261122T032748Z-inkling2-r10-combined)):

| | r8 | r10 |
| --- | ---: | ---: |
| KV pool | 699,885 tokens | 818,388 tokens |
| Cold prefill at 2K / 8K / 246K | 17,911–17,949 / 18,253–18,367 / 8,572–8,577 tok/s | 18,702–18,778 / 18,446–18,483 / 8,691–8,692 tok/s |
| Spec-Bench per stream, 1 / 4 / 8 streams | 371 / 238–242 / 176–178 tok/s | 376–378 / 242 / 181–189 tok/s |
| Spec-Bench aggregate at 8 streams | 1,517 / 1,536 tok/s | 1,592 / 1,600 tok/s |
| Text vs the first r8 boot, one request at a time (11 prompts, greedy and seeded) | identical | identical |

Draft sharding alone, paired boots (A S A S): Spec-Bench aggregate at 4 / 8 streams 1,002 / 1,538–1,545 →
1,030–1,033 / 1,587–1,596 tok/s, one stream 372 → 374 tok/s, text identical
([results](../bench/results/20261122T012901Z-inkling2-draftshard)).

## Output under concurrency (image r8)

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
drafter's drafted and accepted token counts unchanged at one and four streams.

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

The historical [TP1 Spec-Bench run](../bench/results/20261121T014555Z-inkling2-specbench-tp1-nospec-32k/server.txt) used one card, a 32K context and no drafter. The current TP2 launcher shares the model over both cards. See [Compatibility profiles](#compatibility-profiles) for the unmeasured r14 TP1 launch shape.

## Unsupported on sm_120 in the day-0 image

Without the overlay and flags in `scripts/serve.sh`, these fail with `cudaErrorNoKernelImageForDevice`:

| Path | Where | Handling |
| --- | --- | --- |
| FlashInfer warmup autotune | `kernel_warmup.flashinfer_autotune` | `--kernel-config '{"enable_flashinfer_autotune":false}'` |
| DSpark KV-cache write | `reshape_and_cache_flash` from the drafter's `_write_kv` | Triton scatter (`docker/kv_write_sm120.py`) |
| Fused vocab-parallel embedding (TP 2) | `ops.vocab_parallel_embedding` | unfused path on SM12x (`docker/patch_embedding.py`) |
| Custom all-reduce (TP 2) | vLLM `custom_all_reduce` | `--disable-custom-all-reduce` |
