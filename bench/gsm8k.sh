#!/usr/bin/env bash
# Install with: python3 -m pip install 'lm-eval[api]==0.4.12'
# Usage: [URL=http://127.0.0.1:8000] [MODEL=inkling-2-mini] bench/gsm8k.sh OUTDIR
set -euo pipefail

out=${1:?Usage: bench/gsm8k.sh OUTDIR}
url=${URL:-http://127.0.0.1:${PORT:-8000}}
mkdir -p -- "$out"
exec python3 - "${url%/}/v1/chat/completions" "$out/gsm8k-results.json" "${MODEL:-inkling-2-mini}" <<'PY'
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

chat_url, output, model = sys.argv[1:]
task = Path(lm_eval.__file__).resolve().parent / "tasks/gsm8k/gsm8k.yaml"
with TemporaryDirectory() as directory:
    # Inherit the installed harness task and its answer extractors while pinning the dataset load.
    (Path(directory) / "gsm8k_pinned.yaml").write_text(
        f"include: {json.dumps(str(task))}\n"
        "task: gsm8k_pinned\n"
        "dataset_kwargs:\n"
        "  revision: 740312add88f781978c0658806c59bc2815b9866\n",
        encoding="utf-8",
    )
    results = simple_evaluate(
        model="local-chat-completions",
        model_args={"base_url": chat_url, "model": model, "num_concurrent": 4,
                    "max_retries": 1, "tokenized_requests": False, "tokenizer_backend": None},
        tasks=["gsm8k_pinned"], task_manager=TaskManager(include_path=directory),
        apply_chat_template=True, fewshot_as_multiturn=True,
        num_fewshot=5, limit=200, batch_size=1, device="cuda:0", bootstrap_iters=100000, log_samples=False,
        random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
        gen_kwargs={"temperature": 0, "reasoning_effort": "none", "max_gen_toks": 4096,
                    "until": ["<|inkling-eval-no-stop|>"]},
    )
with open(output, "w") as report:
    json.dump(results, report, indent=2, default=handle_non_serializable)
print(output)
PY
