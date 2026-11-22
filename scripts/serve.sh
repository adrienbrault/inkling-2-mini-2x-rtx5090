#!/usr/bin/env bash
# Serve Inkling-2-Mini on two RTX 5090s with vLLM: tensor parallel 2, the bundled DSpark drafter, 262,144-token context,
# fp8 KV cache.
#
#   bash scripts/build-image.sh            # once: day-0 vLLM image + the SM120 overlay in docker/
#   MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh
#
# MODEL_DIR holds the official checkpoint (huggingface.co/thinkingmachines/Inkling-2-Mini, NVFP4/MXFP8, ~20 GB with the
# drafter's mtp.safetensors). The server listens on PORT (default 8000) and serves the id "inkling-2-mini".
# Settings and why (docs/CONFIGURATION.md has the measurements behind each):
#   --tensor-parallel-size 2         both cards as one model; one card alone cannot hold the drafter and a useful context
#   --speculative-config dspark      the checkpoint's own drafter; 7 draft tokens per step (the only value it accepts);
#                                    the drafter's own KV cache stays bf16 ("kv_cache_dtype":"auto")
#   --kv-cache-dtype fp8             the target's KV cache in unit-scale E4M3 (481,529-token pool in bf16), decode and
#                                    prefill unchanged
#   VLLM_INKLING_CONV_PAGE=1, --block-size 64  overlay r9: the short-conv state gets small pages of its own size instead
#                                    of attention-sized ones: 819,427 KV tokens instead of 699,885 (818,388 with the draft
#                                    sharding below), same output
#   VLLM_INKLING_DRAFT_SHARD=1       overlay r10: the drafter samples each card's vocabulary shard and the cards swap
#                                    winners instead of gathering full-vocabulary draft logits (same output)
#   --max-num-seqs 16                16 concurrent sequences: a larger KV pool than vLLM's default sequence limit
#                                    and faster decode at 8 streams
#   VLLM_DECODE_STEP_BUDGET=1024     overlay r8: while any request is decoding, a step carries at most 1,024 tokens, so
#                                    long prefills interleave with the decoding streams in smaller slices; a prefill with
#                                    nobody decoding keeps vLLM's 2,048-token chunks (same output, same prefill rate)
#   --long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive  fair prefill: while several requests
#                                    prefill, each gets at most max(768, 2,048 / their number) tokens per step, so a short
#                                    request behind a long cold prompt starts in 0.5 s instead of 20 s; a lone prefill is
#                                    scheduled as before (same output); costs 2-4 % aggregate decode at 10 agent sessions
#   VLLM_USE_RUST_FRONTEND=1        vLLM's Rust API server instead of the Python one: same output text and throughput,
#                                    first token about 0.1 s sooner on long agent prompts; the streamed reasoning keeps
#                                    its trailing whitespace
#   --enable-batch-sharded-sampling  overlay r7: each request is sampled on one card from both cards' vocabulary shards
#                                    instead of gathering the full logits on both
#   TRITON_CACHE_DIR in CACHE_DIR    Inkling's kernels are Triton; without a persisted Triton cache every boot recompiles
#                                    them (about 3 min 15 s to ready instead of about 40 s)
#   VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1  FlashInfer's PCIe all-reduce for the TP group: about 10 % faster decode
#                                    than NCCL (Inkling's own fused all-reduce needs NVLink multicast)
#   --disable-custom-all-reduce      vLLM's custom all-reduce is compiled without sm_120 in the day-0 image
#   --kernel-config autotune off     FlashInfer's warmup autotune profiles kernels that have no sm_120 build in the image
#   --reasoning-parser / --tool-call-parser inkling, --enable-auto-tool-choice: chat, reasoning and tool calls
#   --enable-prompt-tokens-details   usage reports cached prompt tokens
# Required on sm_120 with this image: --disable-custom-all-reduce, the autotune setting and the bf16 drafter cache (the
# overlay's drafter KV write covers the unquantized cache only and refuses an fp8 one).
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the Inkling-2-Mini checkpoint directory}"
IMAGE=${IMAGE:-inkling2-rtx5090:sm120-r10}
PORT=${PORT:-8000}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling2-vllm}
mkdir -p "$CACHE_DIR"
exec docker run --rm --name inkling --gpus '"device=0,1"' --ipc=host --stop-timeout 30 -p "$PORT:8000" \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 -e VLLM_DECODE_STEP_BUDGET=1024 -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_INKLING_CONV_PAGE=1 -e VLLM_INKLING_DRAFT_SHARD=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -v "$MODEL_DIR":/models/inkling-2-mini:ro -v "$CACHE_DIR":/root/.cache \
  "$IMAGE" /models/inkling-2-mini --served-model-name inkling-2-mini --port 8000 \
  --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --kernel-config '{"enable_flashinfer_autotune":false}' --disable-custom-all-reduce --enable-prompt-tokens-details \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto"}' \
  --kv-cache-dtype fp8 --enable-batch-sharded-sampling --block-size 64 \
  --long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive
