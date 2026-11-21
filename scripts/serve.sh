#!/usr/bin/env bash
# Serve Inkling-2-Mini on two RTX 5090s with vLLM: tensor parallel 2, the bundled DSpark drafter, 262,144-token context.
#
#   bash scripts/build-image.sh            # once: day-0 vLLM image + the SM120 overlay in docker/
#   MODEL_DIR=/path/to/Inkling-2-Mini bash scripts/serve.sh
#
# MODEL_DIR holds the official checkpoint (huggingface.co/thinkingmachines/Inkling-2-Mini, NVFP4/MXFP8, ~20 GB with the
# drafter's mtp.safetensors). The server listens on PORT (default 8000) and serves the id "inkling-2-mini".
# Flags and why (docs/CONFIGURATION.md has the measurements behind each):
#   --tensor-parallel-size 2         both cards as one model: 16.98 GiB of KV per card and the full 262K context; one card
#                                    alone leaves 5.2 GiB once the drafter is loaded (about 59K of context)
#   --speculative-config dspark      the checkpoint's own drafter; 7 draft tokens per step (the only value it accepts)
#   --disable-custom-all-reduce      vLLM's custom all-reduce is compiled without sm_120 in the day-0 image; NCCL is used
#   --kernel-config autotune off     FlashInfer's warmup autotune profiles kernels that have no sm_120 build in the image
#   --reasoning-parser / --tool-call-parser inkling, --enable-auto-tool-choice: the recipe's chat and tool settings
#   --enable-prompt-tokens-details   usage reports cached prompt tokens (the benchmarks check cache hits with it)
# Required on sm_120 with this image: keep --disable-custom-all-reduce and the autotune setting; the default KV dtype
# (an fp8 cache is refused by the overlay).
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the Inkling-2-Mini checkpoint directory}"
IMAGE=${IMAGE:-inkling2-rtx5090:sm120-r3}
PORT=${PORT:-8000}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling2-vllm}
mkdir -p "$CACHE_DIR"
exec docker run --rm --name inkling --gpus '"device=0,1"' --ipc=host -p "$PORT:8000" \
  -v "$MODEL_DIR":/models/inkling-2-mini:ro -v "$CACHE_DIR":/root/.cache \
  "$IMAGE" /models/inkling-2-mini --served-model-name inkling-2-mini --port 8000 \
  --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --kernel-config '{"enable_flashinfer_autotune":false}' --disable-custom-all-reduce --enable-prompt-tokens-details \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
