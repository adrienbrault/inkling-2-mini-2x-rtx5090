#!/usr/bin/env bash
# Spec-Bench against a running server: the 120-prompt subsets in bench/inputs (SPEC-BENCH.md), c1/c4/c8/c16, temperature 0,
# 256 forced output tokens (--ignore-eos), `vllm bench serve` from the serving image plus pandas/datasets.
#   MODEL_DIR=/path/to/Inkling-2-Mini bash bench/specbench.sh OUTDIR
set -euo pipefail

O=${1:?Usage: bench/specbench.sh OUTDIR}
: "${MODEL_DIR:?set MODEL_DIR to the Inkling-2-Mini checkpoint directory}"
[[ $MODEL_DIR == /* && -d $MODEL_DIR ]] || { echo 'MODEL_DIR must be an absolute checkpoint directory' >&2; exit 1; }
mkdir -p -- "$O"
O=$(CDPATH= cd -- "$O" && pwd -P)
bench=$(cd -- "$(dirname -- "$0")" && pwd -P)
DATA=$bench/inputs
URL=${URL:-http://127.0.0.1:${PORT:-8000}}
URL=${URL%/}
IMAGE=${IMAGE:-inkling2-rtx5090:sm120-r14}
docker image inspect "$IMAGE" >/dev/null ||
  { echo "Build $IMAGE first with scripts/build-image.sh and the same IMAGE value." >&2; exit 1; }
# Client build pins. The r13 archive does not record the installed versions.
docker build -q -t inkling2-rtx5090:bench --build-arg "SERVING_IMAGE=$IMAGE" - <<'DF' > /dev/null
ARG SERVING_IMAGE
FROM ${SERVING_IMAGE}
RUN pip install --no-cache-dir --break-system-packages pandas==3.1.1 datasets==5.2.0
DF

spec_metrics() {
  curl -fsS "$URL/metrics" |
    awk '/^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total[{ ]/'
}

for c in 1 4 8 16; do
  spec_metrics > "$O/spec-before-c$c.txt"
  if ! docker run --rm --network host --entrypoint vllm -v "$DATA":/data:ro -v "$MODEL_DIR":/model:ro -v "$O":/out \
    inkling2-rtx5090:bench bench serve --backend openai-chat --base-url "$URL" --endpoint /v1/chat/completions \
    --model inkling-2-mini --tokenizer /model --temperature 0 --request-rate inf --max-concurrency "$c" --num-warmups 0 \
    --num-prompts 120 --no-oversample --dataset-name spec_bench --dataset-path "/data/specbench-subset-c$c.jsonl" \
    --spec-bench-output-len 256 --ignore-eos --seed "$((7310 + c))" --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 --save-result --save-detailed --result-dir /out --result-filename "specbench-c$c.json" \
    > "$O/specbench-c$c.log" 2>&1; then
    echo "Spec-Bench failed at concurrency $c. See $O/specbench-c$c.log" >&2
    exit 1
  fi
  spec_metrics > "$O/spec-after-c$c.txt"
  python3 "$bench/specbench_summary.py" "$O" "$c"
done | tee "$O/specbench-summary.jsonl"
