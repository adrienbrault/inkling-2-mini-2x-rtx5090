# Quality rows, served configuration (overlay r8), 2026-11-21

From the measurement run that also produced the Spec-Bench, code-depth and prefill results of the same boot.

- `tooleval.json`: tool-eval-bench 2.1.0, 69 scenarios, 4 trials, the checkpoint's default sampling (temperature
  1.0), 8 in parallel. The local report path was removed.
- `tooleval.json`'s `trial_statistics`: 4 trials, final score mean 84.2 (standard deviation 2.2), median 85.
- `gsm8k-results.json`: lm-eval 0.4.12 `local-chat-completions`, GSM8K 5-shot, first 200 problems, temperature 0,
  `reasoning_effort` none, concurrency 4. The local package path in `config_source` is shortened to `<site-packages>`.
  Score 53.5 % (flexible extract), 32.0 % (strict). This is not a usable accuracy figure. Even at the lowest effort the
  model reasons first, and 64 of the 200 answers are empty because reasoning used up lm-eval's generation budget. Of
  the 136 problems with an answer, 107 are correct (79 %). The README does not report it.
- `smoke.json`: served-path checks (tool call, image input, reasoning efforts).
