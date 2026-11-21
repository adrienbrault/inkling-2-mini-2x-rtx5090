#!/usr/bin/env bash
# Spec-Bench against a running server: the 120-prompt subsets in bench/inputs (SPEC-BENCH.md), c1/c4/c8, temperature 0,
# 256 forced output tokens (--ignore-eos), `vllm bench serve` from the serving image plus pandas/datasets.
#   MODEL_DIR=/path/to/Inkling-2-Mini bash bench/specbench.sh OUTDIR
set -euo pipefail
O=$(realpath "${1:?outdir}"); mkdir -p "$O"
: "${MODEL_DIR:?}"
DATA=$(realpath "$(dirname "$0")/inputs")
URL=${URL:-http://127.0.0.1:8000}
docker build -q -t inkling2-rtx5090:bench - <<'DF' > /dev/null
FROM inkling2-rtx5090:sm120-r3
RUN pip install --no-cache-dir --break-system-packages pandas datasets
DF
for c in 1 4 8; do
  docker run --rm --network host --entrypoint vllm -v "$DATA":/data:ro -v "$MODEL_DIR":/model:ro -v "$O":/out \
    inkling2-rtx5090:bench bench serve --backend openai-chat --base-url "$URL" --endpoint /v1/chat/completions \
    --model inkling-2-mini --tokenizer /model --temperature 0 --request-rate inf --max-concurrency "$c" --num-warmups 0 \
    --num-prompts 120 --no-oversample --dataset-name spec_bench --dataset-path "/data/specbench-subset-c$c.jsonl" \
    --spec-bench-output-len 256 --ignore-eos --seed "$((7310 + c))" --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 --save-result --save-detailed --result-dir /out --result-filename "specbench-c$c.json"
done
python3 "$(dirname "$0")/specbench_summary.py" "$O"
