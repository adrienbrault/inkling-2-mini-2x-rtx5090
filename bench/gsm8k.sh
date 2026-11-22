#!/usr/bin/env bash
# Install with: python3 -m pip install 'lm-eval[api]==0.4.12'
# Usage: URL=http://127.0.0.1:8000 bench/gsm8k.sh OUTDIR
set -euo pipefail

out=${1:?Usage: bench/gsm8k.sh OUTDIR}
url=${URL:-http://127.0.0.1:${PORT:-8000}}
mkdir -p -- "$out"
exec python3 - "${url%/}/v1/chat/completions" "$out/gsm8k-results.json" <<'PY'
import json
import sys

from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable

chat_url, output = sys.argv[1:]
results = simple_evaluate(
    model="local-chat-completions",
    model_args={"base_url": chat_url, "model": "inkling-2-mini", "num_concurrent": 4,
                "max_retries": 1, "tokenized_requests": False, "tokenizer_backend": None},
    tasks=["gsm8k"], apply_chat_template=True, fewshot_as_multiturn=True,
    num_fewshot=5, limit=200, batch_size=1, device="cuda:0", bootstrap_iters=100000, log_samples=False,
    random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
    gen_kwargs={"temperature": 0, "reasoning_effort": "none", "max_gen_toks": 4096,
                "until": ["<|inkling-eval-no-stop|>"]},
)
with open(output, "w") as report:
    json.dump(results, report, indent=2, default=handle_non_serializable)
print(output)
PY
