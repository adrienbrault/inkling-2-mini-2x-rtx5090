# Quality on the served configuration (image sm120-r13, 1M window, both names served), 2026-11-22

Same boot as `20261122T192436Z-inkling2-specbench-served-r13`. Commands: `bench/tooleval.sh` and `bench/gsm8k.sh` ([`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md)).

| | `inkling-2-mini` | `inkling-2-mini-ablit` |
| --- | ---: | ---: |
| tool-eval-bench 2.1.0, 69 scenarios × 4 trials, T 1.0: trial mean (sd), median | 83.2 (1.9), 82.5 | 78.2 (2.2), 78.0 |
| GSM8K, lm-eval 0.4.12, 5-shot, 200 problems: flexible extract / strict match | 94.5 % / 66.0 % | 96.5 % / 73.5 % |

`smoke.json`: tool call, image input and four sampled generations at T 0.8 on `inkling-2-mini`, all pass.
