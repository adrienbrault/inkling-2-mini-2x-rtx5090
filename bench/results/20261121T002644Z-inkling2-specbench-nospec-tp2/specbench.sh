#!/usr/bin/env bash
# Spec-Bench headline subset against the running Inkling on :8022 (public protocol, flan/headline-r1: 120 stratified
# prompts per cell, c1/c4/c8 in one boot, temperature 0, 256 forced output tokens, template-default thinking), with the
# vllm-inkling2:bench client image (serving overlay + vllm[bench] deps). Also saves the server's spec-decode metrics.
#   bash /srv/qwen5090/inkling2/specbench.sh LABEL     -> /srv/qwen5090/results/<UTC>-inkling2-specbench-LABEL/
set -euo pipefail
LABEL=${1:?label}
D=/srv/qwen5090
DATA=$D/results/20261116T193116Z-1449174-headline-r1-chain/data
O=$D/results/$(date -u +%Y%m%dT%H%M%SZ)-inkling2-specbench-$LABEL
mkdir -p "$O"
cp -- "$0" "$O/"
sudo -n docker inspect -f '{{.Image}} {{.Args}}' inkling > "$O/server.txt"
for c in 1 4 8; do
    sudo -n docker run --rm --network host --entrypoint vllm -v "$DATA":/data:ro -v "$D/fast/models/inkling-2-mini":/model:ro \
        -v "$O":/out vllm-inkling2:bench bench serve --backend openai-chat --base-url http://127.0.0.1:8022 \
        --endpoint /v1/chat/completions --model inkling-2-mini --tokenizer /model --temperature 0 --request-rate inf \
        --max-concurrency "$c" --num-warmups 0 --num-prompts 120 --no-oversample --dataset-name spec_bench \
        --dataset-path "/data/specbench-subset-c$c.jsonl" --spec-bench-output-len 256 --seed "$((7310 + c))" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --save-detailed \
        --result-dir /out --result-filename "specbench-c$c.json" > "$O/specbench-c$c.log" 2>&1
    echo "c$c: $(grep -a -E 'Output token throughput|Mean TPOT|Median TTFT' "$O/specbench-c$c.log" | tr -s ' ' | tr '\n' ';')"
done
sudo -n docker logs inkling 2>&1 | grep -a "SpecDecoding metrics" > "$O/specdecode.log" || true
echo "$O"
