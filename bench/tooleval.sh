#!/usr/bin/env bash
# Install with: python3 -m pip install 'tool-eval-bench==2.1.0'
# Usage: [URL=http://127.0.0.1:8000] [MODEL=inkling-2-mini] bench/tooleval.sh OUTDIR
set -euo pipefail

out=${1:?Usage: bench/tooleval.sh OUTDIR}
url=${URL:-http://127.0.0.1:${PORT:-8000}}
mkdir -p -- "$out"
# The command the published results ran (tool-eval-bench 2.1.0). The headline is trial_statistics.final_score_mean.
exec tool-eval-bench --base-url "${url%/}/v1" --model "${MODEL:-inkling-2-mini}" \
  --temperature 1.0 --trials 4 --parallel 8 --json-file "$out/tooleval.json"
