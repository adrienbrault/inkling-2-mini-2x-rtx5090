# Quality rows, served configuration (image sm120-r10, fair prefill, Rust frontend), 2026-11-22

Same boot as the Spec-Bench, code-depth, prefill and agent-replay results of 2026-11-22 14:33 UTC.

- `tooleval.json`: tool-eval-bench 2.1.0, 69 scenarios, 4 trials, the checkpoint's default sampling (temperature 1.0), 8 in parallel. `trial_statistics`: final score mean 83.5 (standard deviation 3.0), median 84.0. The local report path was removed.
- `gsm8k-results.json`: lm-eval GSM8K, 5-shot, first 200 problems, chat template, temperature 0, reasoning effort "none", a 4,096-token generation budget. Flexible-extract exact match 0.945 (± 0.016); strict-match 0.680, because the model's answers rarely end in the strict "#### N" form. Local package paths are shown as `<site-packages>`.
- `smoke.json`: served-path checks: a tool call, an image input, and four sampled generations at temperature 0.8 with the drafter.
