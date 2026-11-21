#!/usr/bin/env bash
# Spec-Bench headline subset against the running Inkling on :8022 (public protocol, flan/headline-r1: 120 stratified
# prompts per cell, c1/c4/c8 in one boot, temperature 0, 256 forced output tokens (--ignore-eos, checked), template-default
# thinking; per-cell drafter acceptance from /metrics counter deltas), with the
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
spec() { curl -s http://127.0.0.1:8022/metrics | { grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total' || true; } | awk '{s[$1]+=$2} END {for (k in s) print k, s[k]}' | sort || true; }   # no spec metrics without a drafter
for c in 1 4 8; do
    spec > "$O/spec-before-c$c.txt"
    sudo -n docker run --rm --network host --entrypoint vllm -v "$DATA":/data:ro -v "$D/fast/models/inkling-2-mini":/model:ro \
        -v "$O":/out vllm-inkling2:bench bench serve --backend openai-chat --base-url http://127.0.0.1:8022 \
        --endpoint /v1/chat/completions --model inkling-2-mini --tokenizer /model --temperature 0 --request-rate inf \
        --max-concurrency "$c" --num-warmups 0 --num-prompts 120 --no-oversample --dataset-name spec_bench \
        --dataset-path "/data/specbench-subset-c$c.jsonl" --spec-bench-output-len 256 --ignore-eos --seed "$((7310 + c))" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --save-detailed \
        --result-dir /out --result-filename "specbench-c$c.json" > "$O/specbench-c$c.log" 2>&1
    spec > "$O/spec-after-c$c.txt"
    python3 - "$O" "$c" <<'PY2'
import json, sys
from pathlib import Path
O, c = Path(sys.argv[1]), sys.argv[2]
r = json.loads((O / f'specbench-c{c}.json').read_text())
if r['completed'] != 120 or set(r['output_lens']) != {256}:
    sys.exit(f'c{c}: {r["completed"]} completed, output lengths {sorted(set(r["output_lens"]))[:5]}')
rd = lambda f: {l.split()[0].split('{')[0]: float(l.split()[1]) for l in (O / f).read_text().splitlines()}
b, a = rd(f'spec-before-c{c}.txt'), rd(f'spec-after-c{c}.txt')
d = {k: a[k] - b.get(k, 0) for k in a}
dr = d.get('vllm:spec_decode_num_drafts_total', 0)
acc = d.get('vllm:spec_decode_num_accepted_tokens_total', 0)
out = {'c': int(c), 'drafts': dr, 'accepted': acc, 'mean_acceptance_length': round(1 + acc / dr, 3) if dr else None}
(O / f'spec-c{c}.json').write_text(json.dumps(out) + '\n'); print(json.dumps(out))
PY2
    echo "c$c: $(grep -a -E 'Output token throughput|Mean TPOT|Median TTFT' "$O/specbench-c$c.log" | tr -s ' ' | tr '\n' ';')"
done
sudo -n docker logs inkling 2>&1 | grep -a "SpecDecoding metrics" > "$O/specdecode.log" || true
echo "$O"
