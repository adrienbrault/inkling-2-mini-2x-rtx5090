# Reproduce the benchmark protocol

Run from the repository root after [Quick start](../README.md#quick-start). Keep the server idle between workloads and use separate report directories. The harness passes the host-root URL, without `/v1`, to the clients. This preserves a URL you already exported:

```bash
export URL="${URL:-http://127.0.0.1:${PORT:-8000}}"
```

## Measurement boot

The headline measurements were taken on 2026-11-22 on image `inkling2-rtx5090:sm120-r13`. The recorded launch source is commit [`2aa8760`](https://github.com/adrienbrault/inkling-2-mini-2x-rtx5090/tree/2aa8760df59080609d60d2dc9880c7ad1ac4d858). Inspect the historical entry points with `git show 2aa8760:scripts/serve.sh`, `git show 2aa8760:scripts/build-image.sh`, and `git show 2aa8760:docker/Dockerfile`.

The headline configuration used two RTX 5090s at stock power limits, TP2, a 1,048,576-token window, both served names, fp8 target KV, bf16 DSpark KV, fair prefill, and the Rust frontend. The performance and quality captures share a boot. The KV capture records r13 at 1M with both names. The newcomer capture records r13, and its window and adapter state are **TO CONFIRM**. The r13 launcher defaults to Rust, and the separate KV and newcomer captures' frontends are **TO CONFIRM**. The checkpoint revision and host driver from the headline boot are **TO CONFIRM**. Tool-eval records client Python 3.12.3.

Quick start builds r14. Its difference from r13 is in-kernel DSpark RoPE. The [r14 A/B](../bench/results/20261122T203506Z-inkling2-r14-ab/summary.json) matched the r13 outputs on 11 lone-request prompts in each sampling mode and matched the one-stream draft counters. It measured 24,553 KV blocks at 1M against 23,734 on r13. The headline suite, live KV residency and a full two-name structured check were not repeated on r14. The commands here describe how to run the protocol against a running server. Record the image digest, checkpoint revision, window, adapter and frontend, along with the host setup and dependency versions, for any new run.

## Performance

The Spec-Bench wrapper requires curl, awk, Python 3 and a local Linux Docker Engine with host networking so the client container can reach the server's loopback URL. It builds a client from `IMAGE` by installing the wrapper's pins, `pandas==3.1.1` and `datasets==5.2.0`, from PyPI into the base's Python 3.14. The historical client environment and an install of these pins on that base are **TO CONFIRM**. A failing build is a client dependency failure. Use the top-level harness below when reading older archived job scripts.

```bash
bash bench/specbench.sh runs/specbench
python3 bench/headline_inkling.py code runs/code --url "$URL"
python3 bench/headline_inkling.py prefill runs/prefill --url "$URL"
python3 bench/inkling_replay.py runs/pi-6 --url "$URL" --preset pi --slots 6 --duration 300 --warmup 150
python3 bench/inkling_replay.py runs/pi-10 --url "$URL" --preset pi --slots 10 --duration 300 --warmup 150
```

| Workload | Outputs and checks | r13 reference |
| --- | --- | --- |
| Spec-Bench | Read `specbench-summary.jsonl` and the counter snapshots. Each of the four frozen subsets needs 120 successful responses of exactly 256 output tokens. | At 1 / 4 / 8 / 16 streams, mean 416 / 254 / 196 / 150 tok/s per stream and aggregate 416 / 1,017 / 1,569 / 2,399 tok/s. [Raw results](../bench/results/20261122T192436Z-inkling2-specbench-served-r13) |
| Code at depth | Read `code-summary.json`. Each stream needs a cache hit for the warmed context and 400 output tokens. Use the shared decode window when present, and label the sum of rates when used as a fallback. | At 1K cached depth and a 2K append, 519 tok/s on one stream and 2,897 tok/s shared-window aggregate at 16 streams. [Raw results](../bench/results/20261122T192732Z-inkling2-code-r13) |
| Cold prefill | Read `prefill-summary.json`. Salted prompts need zero cached tokens and one output token. Divide actual input tokens by end-to-end TTFT. | 19,012 / 14,830 / 8,724 tok/s at 2K / 61K / 246K. TTFT at 246K was 28.2 s. [First pass](../bench/results/20261122T192938Z-inkling2-prefill-r13-a), [second pass](../bench/results/20261122T193045Z-inkling2-prefill-r13-b) |
| Agent replay | Read `summary.json` and `requests.jsonl`. Score the last 150 s after 150 s of warmup. Require `valid: true`, zero request errors and no forced-length misses. | 6 / 10 sessions: 524 / 685 tok/s aggregate, median first token 0.16 / 0.22 s. [Raw results](../bench/results/20261122T192343Z-inkling2-pi-r13) |

The drafter-off line in the decode chart comes from a historical r3/NCCL boot. [Drafter comparison](CONFIGURATION.md#drafter-on-or-off)

## KV capacity and newcomer

The KV probe used r13, a 1M window and both names. Use the following parameters to match the workload. Read the timestamped `summary.json` and both cohort files. The 8-request cohort reached simultaneous decode, 66% peak KV usage and zero preemptions. Its capacity estimate was about 1.42M live tokens. Eleven of the twelve requests in the other cohort decoded together, so it yielded no full-cohort capacity estimate. [Raw KV results](../bench/results/20261122T200911Z-inkling2-kv-capacity-r13)

```bash
python3 bench/kv_capacity.py bench/inputs/code-corpus.txt runs/kv --url "$URL" \
  --cohorts 8,12 --prompt-tokens 96000 --output-tokens 24576
```

The r13 newcomer run used fair prefill and three trials. A background stream is already decoding when a salted 196,608-token prefill starts. The tool waits two seconds before sending the 1K and 16K newcomers. Read `newcomer-summary.json` for the median newcomer TTFT and the long request's slowdown relative to its solo run. [Raw newcomer results](../bench/results/20261122T200911Z-inkling2-newcomer-r13)

```bash
python3 bench/newcomer.py runs/newcomer --url "$URL" --long 196608 --trials 3
```

The historical r10 KV residency and admission runs used a 262,144-token window, base name only and the Python frontend. The separately archived fair-prefill off/on A/B also used r10 with a 262,144-token window, base name only and the Python frontend. Follow their archives for the different cohorts and controls. [r10 KV](../bench/results/20261122T124729Z-inkling2-kv-capacity), [r10 fair prefill](../bench/results/20261122T092118Z-inkling2-fair-prefill-768)

## Admission diagnostics

Launch with `VLLM_INKLING_CONVPAGE_PROBE=1` and capture the scheduler log from another terminal. Set `URL` to the port you chose for the launch.

```bash
mkdir -p runs/admission
docker logs -f inkling > runs/admission/scheduler.log 2>&1 &
log_pid=$!
python3 bench/kv_probe.py admission bench/inputs/code-corpus.txt runs/admission \
  --url "$URL" --scheduler-log runs/admission/scheduler.log
kill "$log_pid"
```

The log needs the `INKLING_KV_PROBE_CONFIG` entry and the queued and scheduled events for each probe request.

## Quality

Use a client venv with PyPI access. The r13 package versions below come from the reports. The complete venv lock is **TO CONFIRM**.

```bash
python3 -m venv .venv-eval
. .venv-eval/bin/activate
python -m pip install 'tool-eval-bench==2.1.0' 'lm-eval[api]==0.4.12'
```

### Tool-eval

```bash
bash bench/tooleval.sh runs/tooleval
MODEL=inkling-2-mini-ablit bash bench/tooleval.sh runs/tooleval-ablit
```

The wrapper sends 69 scenarios (TC-01 to TC-69) in four trials at temperature 1.0 and parallelism 8. Require a completed report with zero request errors. Read `trial_statistics.final_score_mean` in each `tooleval.json`. The r13 means were 83.2 (trial standard deviation 1.9) for the base name and 78.2 (2.2) for the adapter. The earlier Rust frontend comparison is historical r10. [r13 quality](../bench/results/20261122T192343Z-inkling2-quality-r13), [r10 quality](../bench/results/20261122T143310Z-inkling2-quality-rustfe)

### GSM8K

```bash
bash bench/gsm8k.sh runs/gsm8k
MODEL=inkling-2-mini-ablit bash bench/gsm8k.sh runs/gsm8k-ablit
```

The wrapper inherits the harness task and pins [`openai/gsm8k` at `740312add88f781978c0658806c59bc2815b9866`](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866). The archived task configuration names `openai/gsm8k` and `main`. Its resolved revision was not recorded by lm-eval. Compare the first 200 test examples with five shots, the same extractors and the 4,096-token budget with `reasoning_effort` set to `none`. Read flexible extraction and strict match from `gsm8k-results.json`. The r13 base name yielded 94.5% and 66.0%, and the adapter yielded 96.5% and 73.5%. [r13 quality](../bench/results/20261122T192343Z-inkling2-quality-r13)
