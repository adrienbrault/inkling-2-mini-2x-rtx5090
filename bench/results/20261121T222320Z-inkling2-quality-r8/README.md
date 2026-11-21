# Quality rows, served configuration (overlay r8), 2026-11-21

From the measurement run that also produced the Spec-Bench, code-depth and prefill results of the same boot.

- `tooleval.json`: tool-eval-bench 2.1.0, 69 scenarios, the checkpoint's default sampling (temperature 1.0), 8 in
  parallel. The local report path was removed.
- `gsm8k-results.json`: lm-eval 0.4.12 `local-chat-completions`, GSM8K 5-shot, first 200 problems, temperature 0,
  `reasoning_effort` none, concurrency 4. The local package path in `config_source` is shortened to `<site-packages>`.
- `smoke.json`: served-path checks (tool call, image input, reasoning efforts).
