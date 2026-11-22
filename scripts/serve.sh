#!/usr/bin/env bash
# Serve Inkling-2-Mini on two RTX 5090s with vLLM: tensor parallel 2, the bundled DSpark drafter, 1,048,576-token context,
# fp8 KV cache, and the abliterated name inkling-2-mini-ablit beside inkling-2-mini when the adapter is present.
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

die() { printf 'serve.sh: %s\n' "$*" >&2; exit 1; }

readable_file() { [[ -f "$1" && -r "$1" && -s "$1" ]]; }
mount_path() {
  local path=$1
  if [[ $path != /* ]]; then path=./$path; fi
  path=$(cd -P -- "$path" && pwd) || return 1
  [[ $path != *:* && $path != *$'\n'* ]] || return 1
  printf '%s\n' "$path"
}

MODEL_DIR=${MODEL_DIR:-}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling2-vllm}
IMAGE=${IMAGE:-inkling2-rtx5090:sm120-r13}
PORT=${PORT:-8000}
BIND=${BIND:-127.0.0.1}
GPUS=${GPUS:-0,1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
# The abliteration adapter (scripts/fetch-adapter.sh) is served by default when it sits next to the checkpoint, at
# MODEL_DIR-ablit. ABLIT_DIR= (empty) turns it off; ABLIT_DIR=/path selects another directory.
if [[ -z ${ABLIT_DIR+set} && -n $MODEL_DIR && -d ${MODEL_DIR%/}-ablit ]]; then ABLIT_DIR=${MODEL_DIR%/}-ablit; fi
ABLIT_DIR=${ABLIT_DIR:-}
DSPARK=${DSPARK:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
FAIR_PREFILL=${FAIR_PREFILL:-1}
VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
VLLM_DECODE_STEP_BUDGET=${VLLM_DECODE_STEP_BUDGET:-1024}
VLLM_INKLING_CONVPAGE_PROBE=${VLLM_INKLING_CONVPAGE_PROBE:-0}

[[ $PORT =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) ||
  die "PORT must be an integer from 1 to 65535"
[[ $BIND =~ ^(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})$ ]] ||
  die "BIND must be an IPv4 address (for example, 127.0.0.1 or 0.0.0.0)"
for octet in "${BASH_REMATCH[@]:1}"; do
  (( octet <= 255 )) || die "BIND must be an IPv4 address"
done
[[ $GPUS =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))?$ ]] ||
  die "GPUS must be one GPU index or two comma-separated GPU indices"
tp=1
if [[ $GPUS == *,* ]]; then
  [[ ${GPUS%,*} != "${GPUS#*,}" ]] || die "GPUS must contain distinct GPU indices"
  tp=2
fi
[[ $MAX_MODEL_LEN =~ ^[1-9][0-9]*$ ]] || die "MAX_MODEL_LEN must be a positive integer"
[[ $MAX_NUM_SEQS =~ ^[1-9][0-9]*$ ]] || die "MAX_NUM_SEQS must be a positive integer"
[[ $GPU_MEMORY_UTILIZATION =~ ^((0?\.[0-9]+)|1(\.0+)?)$ && ! $GPU_MEMORY_UTILIZATION =~ ^0?\.0+$ ]] ||
  die "GPU_MEMORY_UTILIZATION must be a number greater than 0 and at most 1"
[[ $VLLM_DECODE_STEP_BUDGET =~ ^(0|[1-9][0-9]*)$ ]] || die "VLLM_DECODE_STEP_BUDGET must be a nonnegative integer"
[[ $KV_CACHE_DTYPE == fp8 || $KV_CACHE_DTYPE == auto ]] || die "KV_CACHE_DTYPE must be fp8 or auto"
[[ $IMAGE =~ ^[[:alnum:]][[:alnum:]_.:/@-]*$ ]] || die "IMAGE must be a Docker image reference"

tp2=0
if (( tp == 2 )); then tp2=1; fi
conv_pages=0
if [[ $DSPARK == 1 && $tp2 == 1 && $KV_CACHE_DTYPE == fp8 ]]; then conv_pages=1; fi
BATCH_SHARDED_SAMPLING=${BATCH_SHARDED_SAMPLING:-$tp2}
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-$tp2}
VLLM_INKLING_CONV_PAGE=${VLLM_INKLING_CONV_PAGE:-$conv_pages}
draft_shard=0
if [[ $DSPARK == 1 && $tp2 == 1 && $BATCH_SHARDED_SAMPLING == 1 ]]; then draft_shard=1; fi
VLLM_INKLING_DRAFT_SHARD=${VLLM_INKLING_DRAFT_SHARD:-$draft_shard}
for name in DSPARK FAIR_PREFILL BATCH_SHARDED_SAMPLING VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC VLLM_USE_RUST_FRONTEND \
            VLLM_INKLING_CONV_PAGE VLLM_INKLING_DRAFT_SHARD VLLM_INKLING_CONVPAGE_PROBE; do
  [[ ${!name} =~ ^[01]$ ]] || die "$name must be 0 or 1"
done
if (( tp == 1 )); then
  [[ $DSPARK == 0 ]] || die "set DSPARK=0 when selecting one GPU"
  [[ $BATCH_SHARDED_SAMPLING == 0 && $VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC == 0 ]] ||
    die "batch-sharded sampling and FlashInfer PCIe IPC require two GPUs"
fi
[[ $VLLM_INKLING_CONV_PAGE == 0 || $conv_pages == 1 ]] ||
  die "VLLM_INKLING_CONV_PAGE=1 requires two GPUs, DSpark and KV_CACHE_DTYPE=fp8"
[[ $VLLM_INKLING_DRAFT_SHARD == 0 || $draft_shard == 1 ]] ||
  die "VLLM_INKLING_DRAFT_SHARD=1 requires two GPUs, DSpark and BATCH_SHARDED_SAMPLING=1"

[[ $MODEL_DIR == /* && -d $MODEL_DIR && -r $MODEL_DIR && -x $MODEL_DIR ]] ||
  die "MODEL_DIR must be an absolute path to a readable checkpoint directory"
MODEL_DIR=$(mount_path "$MODEL_DIR") || die "MODEL_DIR cannot be mounted (check access and colons in its path)"
readable_file "$MODEL_DIR/config.json" || die "MODEL_DIR is missing readable config.json"
readable_file "$MODEL_DIR/tokenizer_config.json" || die "MODEL_DIR is missing readable tokenizer_config.json"
if ! readable_file "$MODEL_DIR/tokenizer.json" && ! readable_file "$MODEL_DIR/tokenizer.model" &&
   ! { readable_file "$MODEL_DIR/vocab.json" && readable_file "$MODEL_DIR/merges.txt"; }; then
  die "MODEL_DIR is missing tokenizer data (tokenizer.json, tokenizer.model, or vocab.json and merges.txt)"
fi
readable_file "$MODEL_DIR/mtp.safetensors" || die "MODEL_DIR is missing readable mtp.safetensors"
shopt -s nullglob
weights=0
for shard in "$MODEL_DIR"/*.safetensors; do
  [[ $shard != "$MODEL_DIR/mtp.safetensors" ]] || continue
  readable_file "$shard" || die "MODEL_DIR has an unreadable or empty weight shard: ${shard##*/}"
  weights=1
done
(( weights )) || die "MODEL_DIR needs at least one target weight shard (*.safetensors) in addition to mtp.safetensors"

if [[ -n $ABLIT_DIR ]]; then
  [[ -d $ABLIT_DIR && -r $ABLIT_DIR && -x $ABLIT_DIR ]] || die "ABLIT_DIR must be a readable adapter directory"
  ABLIT_DIR=$(mount_path "$ABLIT_DIR") || die "ABLIT_DIR cannot be mounted (check access and colons in its path)"
  for file in adapter_config.json adapter_model.safetensors; do
    readable_file "$ABLIT_DIR/$file" || die "ABLIT_DIR is missing readable $file"
  done
fi

command -v docker >/dev/null || die "Docker is required"
existing=$(docker container ls --all --filter 'name=^/inkling$' --format '{{.Names}}') ||
  die "cannot query Docker containers. Check daemon access"
[[ -z $existing ]] || die 'a container named inkling already exists. Stop it with docker stop inkling, or remove it if stopped'
docker image inspect "$IMAGE" >/dev/null 2>&1 ||
  die "image $IMAGE is unavailable locally. Build it with the same IMAGE value using scripts/build-image.sh"

mkdir -p -- "$CACHE_DIR" && [[ -r $CACHE_DIR && -w $CACHE_DIR && -x $CACHE_DIR ]] ||
  die "CACHE_DIR must be a readable, writable directory"
CACHE_DIR=$(mount_path "$CACHE_DIR") || die "CACHE_DIR cannot be mounted (check access and colons in its path)"

docker_args=(--rm --name inkling --gpus "\"device=$GPUS\"" --ipc=host --stop-timeout 30 -p "$BIND:$PORT:8000"
             -e TRITON_CACHE_DIR=/root/.cache/triton -e "VLLM_DECODE_STEP_BUDGET=$VLLM_DECODE_STEP_BUDGET"
             -v "$MODEL_DIR:/models/inkling-2-mini:ro" -v "$CACHE_DIR:/root/.cache")
server_args=(/models/inkling-2-mini --served-model-name inkling-2-mini --port 8000
             --tensor-parallel-size "$tp" --max-model-len "$MAX_MODEL_LEN"
             --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-seqs "$MAX_NUM_SEQS"
             --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling
             --kernel-config '{"enable_flashinfer_autotune":false}' --disable-custom-all-reduce
             --enable-prompt-tokens-details --kv-cache-dtype "$KV_CACHE_DTYPE")
for name in VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC VLLM_INKLING_CONV_PAGE VLLM_INKLING_DRAFT_SHARD \
            VLLM_USE_RUST_FRONTEND VLLM_INKLING_CONVPAGE_PROBE; do
  if [[ ${!name} == 1 ]]; then docker_args+=(-e "$name=1"); fi
done
if [[ $DSPARK == 1 ]]; then
  server_args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto"}')
fi
if [[ $VLLM_INKLING_CONV_PAGE == 1 ]]; then server_args+=(--block-size 64); fi
if [[ $BATCH_SHARDED_SAMPLING == 1 ]]; then server_args+=(--enable-batch-sharded-sampling); fi
if [[ $FAIR_PREFILL == 1 ]]; then
  server_args+=(--long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive)
fi
if [[ -n $ABLIT_DIR ]]; then
  # overlay r13: with no adapter request in a step, skip the LoRA delta work entirely, and keep Inkling's logits head
  # (the adapter does not touch lm_head), so inkling-2-mini runs as without LoRA. Do not add --specialize-active-lora:
  # its single-group capture would drop the adapter in batches that mix both names.
  docker_args+=(-v "$ABLIT_DIR:/lora:ro" -e VLLM_INKLING_LORA_SKIP=1 -e VLLM_INKLING_LORA_HEAD=1)
  # Without explicit targets, vLLM assumes the LoRA may target lm_head and refuses --enable-batch-sharded-sampling.
  server_args+=(--enable-lora --max-lora-rank 8 --max-loras 1 --lora-target-modules wo_ud
                --lora-modules inkling-2-mini-ablit=/lora)
fi

printf 'Starting Inkling at http://%s:%s (TP%s)\n' "$BIND" "$PORT" "$tp"
for name in MODEL_DIR CACHE_DIR IMAGE PORT BIND GPUS MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION ABLIT_DIR \
            DSPARK KV_CACHE_DTYPE FAIR_PREFILL BATCH_SHARDED_SAMPLING VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC \
            VLLM_DECODE_STEP_BUDGET VLLM_INKLING_CONV_PAGE VLLM_INKLING_DRAFT_SHARD VLLM_USE_RUST_FRONTEND \
            VLLM_INKLING_CONVPAGE_PROBE; do
  printf '  %s=%s\n' "$name" "${!name}"
done
exec docker run "${docker_args[@]}" "$IMAGE" "${server_args[@]}"
