# Reproduce the headline results

Run from the repository root after [Quick start](../README.md#quick-start). Export `MODEL_DIR` for the checkpoint and set `URL` to the server root if you change the host port. The commands below use `http://127.0.0.1:8000`. Use separate output directories and keep other traffic off the server during a run.

The headline boot used two RTX 5090s at stock power limits, TP2, `sm120-r10`, fp8 target KV, bf16 DSpark KV, fair prefill and the Rust frontend. The KV capacity and newcomer runs used the Python frontend as shown below. Save the launch settings, base digest from [the Dockerfile](../docker/Dockerfile), model commit, GPU topology, NVIDIA driver, power limits and client Python version with new results. The tool-eval report records Python 3.12.3. The archive does not establish the checkpoint commit or a host driver version for the headline boot. Use the references to compare the same workload and report repeat-run spread.

```bash
export URL=http://127.0.0.1:8000
```

## Performance on the headline boot

```bash
bash bench/specbench.sh runs/specbench
python3 bench/headline_inkling.py code runs/code --url "$URL"
python3 bench/headline_inkling.py prefill runs/prefill --url "$URL"
python3 bench/inkling_replay.py runs/pi-6 --url "$URL" --preset pi --slots 6 --duration 300 --warmup 150
python3 bench/inkling_replay.py runs/pi-10 --url "$URL" --preset pi --slots 10 --duration 300 --warmup 150
```

| Workload | Output | Validity conditions | Reference |
| --- | --- | --- | --- |
| Spec-Bench | `specbench-c{1,4,8,16}.json`, logs and counter snapshots, plus `specbench-summary.jsonl` | Use the four frozen subsets. Every cell needs 120 successful responses of exactly 256 output tokens. Use an idle server for the speculative counter deltas. | At 1 / 4 / 8 / 16 streams, mean 416 / 258 / 200 / 151 and median 379 / 248 / 185 / 142 tok/s per stream. Aggregate 416 / 1,034 / 1,601 / 2,419 tok/s. Wall-clock 367.37 / 904.38 / 1,379.87 / 2,020.57 tok/s. Median TTFT 32 / 54 / 69 / 106 ms. Acceptance length 2.97 / 3.01 / 2.99 / 2.97. [Raw Spec-Bench results](../bench/results/20261122T143353Z-inkling2-specbench-served-rustfe) |
| Code at depth | `code-summary.json` and per-request JSON files | Each measured stream must hit the warmed context cache and return 400 output tokens. Compare matching depth, append and concurrency. Use `aggregate_shared_window` when present. If streams lack a shared decode window, label `aggregate_sum_of_rates` as the fallback. | At 1K cached depth plus a 2K append, 532 tok/s per stream at concurrency 1 and 2,717 tok/s shared-window aggregate at concurrency 16. The plot averages across depths. [Raw code results](../bench/results/20261122T143647Z-inkling2-code-rustfe) |
| Cold prefill | `prefill-summary.json` and per-prompt JSON files | Each salted prompt must report zero cached tokens and return one output token. Compare actual prompt tokens divided by end-to-end TTFT, matching each row by target length. | 19,115 / 14,835 / 8,716 tok/s at 2K / 61K / 246K, respectively. 246K TTFT was 28.2 s. All seven cells stayed within 0.4% on the second pass of the same boot. [Raw prefill results](../bench/results/20261122T143852Z-inkling2-prefill-rustfe-a), [second pass](../bench/results/20261122T143959Z-inkling2-prefill-rustfe-b) |
| Agent replay `pi:6` | `runs/pi-6/summary.json` and `requests.jsonl` | The first 150 s of the 300 s run are warmup. Require `valid: true`, zero errors and zero forced-length misses. Record the scored request count and cached share. | 511 tok/s aggregate. TTFT p50 / p90 0.15 / 0.69 s. [Raw replay results](../bench/results/20261122T143310Z-inkling2-pi-rustfe) |
| Agent replay `pi:10` | `runs/pi-10/summary.json` and `requests.jsonl` | Use the same scored window and checks as `pi:6`. | 711 tok/s aggregate. TTFT p50 / p90 0.23 / 0.72 s. [Raw replay results](../bench/results/20261122T143310Z-inkling2-pi-rustfe) |

The Spec-Bench client builds from the serving image. The archive has no pandas or datasets versions to pin. The plotted drafter-off curve is from a historical r3/NCCL boot with other settings changed. [Drafter comparison](CONFIGURATION.md#drafter-on-or-off)

## KV capacity and newcomer

For KV capacity, boot with `VLLM_USE_RUST_FRONTEND=0 scripts/serve.sh` and keep fair prefill enabled.

```bash
python3 bench/kv_capacity.py bench/inputs/code-corpus.txt runs/kv --url "$URL" \
  --cohorts 14 --prompt-tokens 80000 --output-tokens 16384
```

Read `summary.json` and `cohort-14.json` under the run directory printed by the probe. Require no request issues, simultaneous decode by all 14 requests and zero preemptions. The 14 × 80K reference peaked at 14 running and 88% KV usage, with a median live-tokens / KV-usage estimate of 1,486,835 tokens. The logged 818,388 is the engine's full-window admission estimate. In the separate 16 × 96K run, the default full-prompt reserve admitted 12 at once with zero preemptions. With `--no-scheduler-reserve-full-isl`, all 16 were admitted and 326 preemptions followed. Without a measured all-decoding overlap, there is no capacity estimate. [Raw KV results](../bench/results/20261122T124729Z-inkling2-kv-capacity)

For newcomer, boot in off/on/off/on order. Start each server in one terminal and run its matching benchmark in another. Stop each server before the next boot.

```bash
# Control boot, repeat for off-1 and off-2
VLLM_USE_RUST_FRONTEND=0 FAIR_PREFILL=0 scripts/serve.sh

# Fair-prefill boot, repeat for on-1 and on-2
VLLM_USE_RUST_FRONTEND=0 FAIR_PREFILL=1 scripts/serve.sh
```

```bash
# Run each command on its matching boot.
python3 bench/newcomer.py runs/off-1 --url "$URL" --long 196608 --trials 3
python3 bench/newcomer.py runs/on-1  --url "$URL" --long 196608 --trials 3
python3 bench/newcomer.py runs/off-2 --url "$URL" --long 196608 --trials 3
python3 bench/newcomer.py runs/on-2  --url "$URL" --long 196608 --trials 3
```

Read `newcomer-summary.json` for each boot. Keep the salted cold prompts, the background decoding stream and the two-second offset. Compare median short and mid TTFT and the long prompt's TTFT relative to its solo run. With fair prefill the short / mid ranges were 0.51–0.52 s / 6.66–6.67 s and the long prompt took 1.16× as long as solo. Without fair prefill the short / mid ranges were 20.44–20.50 s / 21.53–21.54 s and the long prompt took 1.07× as long as solo. [Raw newcomer results](../bench/results/20261122T092118Z-inkling2-fair-prefill-768)

## Quality on the headline boot

### Tool-eval

```bash
python3 -m pip install 'tool-eval-bench==2.1.0'
mkdir -p runs/tooleval
tool-eval-bench --base-url "${URL%/}/v1" --model inkling-2-mini \
  --temperature 1.0 --trials 4 --parallel 8 --json-file runs/tooleval/tooleval.json
# Equivalent wrapper: bash bench/tooleval.sh runs/tooleval
```

This is the command the published results ran. Require the completed report, all 69 scenarios (TC-01 to TC-69), four trials and zero request errors. Read `trial_statistics.final_score_mean` in `tooleval.json` for the headline 83.5 (trial standard deviation 3.0). The top-level score is 86. The Python frontend comparison was 82.8 ± 2.5. [Raw quality results](../bench/results/20261122T143310Z-inkling2-quality-rustfe), [Python frontend results](../bench/results/20261122T122255Z-inkling2-quality-r10-fair)

### GSM8K

The archive records the settings but does not include the original CLI invocation. This command uses the lm-eval API to pass the recorded settings, including the stop string as a list. The wrapper runs the same command.

```bash
python3 -m pip install 'lm-eval[api]==0.4.12'
mkdir -p runs/gsm8k
python3 - "${URL%/}/v1/chat/completions" runs/gsm8k/gsm8k-results.json <<'PY'
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
# Equivalent wrapper: bash bench/gsm8k.sh runs/gsm8k
```

Read `runs/gsm8k/gsm8k-results.json`. Require the first 200 test examples, five shots and the same answer extractors. Keep the non-matching stop string and generation budget to avoid cutting off answers during reasoning. The runs used the `openai/gsm8k` main configuration at dataset revision `740312add88f781978c0658806c59bc2815b9866`. Compare 94.5% flexible extraction (standard error 1.6 percentage points) and 68.0% strict match. [Raw quality results](../bench/results/20261122T143310Z-inkling2-quality-rustfe)
