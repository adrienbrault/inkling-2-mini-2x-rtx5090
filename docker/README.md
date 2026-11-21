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

Each patch script checks the SHA-256 of the file it edits and replaces exactly one block (explicit failures, not
`assert`), so a different base image fails the build instead of patching the wrong code. The scatter covers the
unquantized KV cache only: on SM12x an fp8 drafter cache is refused at startup instead of reaching the missing kernel.
The target's cache can be fp8 (Inkling's own kernels write it); `scripts/serve.sh` keeps the drafter's in bf16 with
`"kv_cache_dtype":"auto"` in the speculative config.

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

## Scheduler and sampling (r7, r8)

| Patch | What it changes | Enabled by |
| --- | --- | --- |
| `patch_sched.py` (vLLM's V1 scheduler) | while any running request is decoding, a step carries at most `VLLM_DECODE_STEP_BUDGET` tokens; a prefill with nobody decoding keeps the full 2,048 | `-e VLLM_DECODE_STEP_BUDGET=1024`; unset or 0 = off |
| `patch_bss.py` (Inkling's model and drafter) | adds `compute_logits_local` (the same head and soft-cap without the TP gather), so vLLM's batch-sharded sampling engages; the drafter's logits cache stays replicated | `--enable-batch-sharded-sampling` |

Both are inert without their setting. Measurements: [`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).

## Credits

The idea of moving the loop-controller and short-conv products off cuBLAS on SM120 comes from the RTX PRO 6000 recipe
at [hyudryu/Inkling-2-Mini-2xRTXPro6000](https://github.com/hyudryu/Inkling-2-Mini-2xRTXPro6000), which also reports
short-block DSpark drafting. The kernels and patches here are independent implementations, checked bitwise against
cuBLAS on this hardware.

`patch_bss.py`'s `compute_logits_local` follows vLLM's own implementations for Llama and Qwen3 (Apache-2.0); the
batch-sharded sampler is vLLM's. The decode-aware step budget is this repository's own.
