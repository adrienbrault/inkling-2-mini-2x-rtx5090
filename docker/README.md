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

Each patch script checks the SHA-256 of the file it edits and replaces exactly one block, so a different base image
fails the build instead of patching the wrong code.
