# Quality rows, served configuration (overlay r10), 2026-11-22

From the measurement run that also produced the Spec-Bench, code-depth and prefill results of the same boot.

- `tooleval.json`: tool-eval-bench 2.1.0, 69 scenarios, 4 trials, the checkpoint's default sampling (temperature
  1.0), 8 in parallel. The local report path was removed.
- `tooleval.json`'s `trial_statistics`: 4 trials, final score mean 82.5 (standard deviation 1.9), median 82. On r8
  (`20261121T222320Z-inkling2-quality-r8`) the same protocol gave mean 84.2 (standard deviation 2.2). r10's output is
  bitwise identical to r8's one request at a time (greedy and seeded); at temperature 1.0 without a seed every trial
  samples differently, so the two runs differ by sampling.
- `smoke.json`: served-path checks (tool call, image input, reasoning efforts).

GSM8K is not reported: the run used the same lm-eval settings as r8's, whose answers are cut off by reasoning (see the
r8 note).
