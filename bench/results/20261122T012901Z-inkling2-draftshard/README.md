# DSpark draft sharding alone (2026-11-22)

Four boots in the order A S A S. A = image r8 with the r8 served settings. S = image r10 with `VLLM_INKLING_DRAFT_SHARD=1`
only (conv pages as in r8).

- `screen.log`: per boot, the identity checks and Spec-Bench at 1, 4 and 8 streams (`bench/specbench_summary.py` lines).
- `<boot>.greedy.json` / `<boot>.seeded.json`: `bench/greedy.py --long` at temperature 0 and `--seeded` at temperature
  1.0; every boot matches A1 on all 11 prompts in both.
- `S1.gate.txt` / `S2.gate.txt`: the two-card GPU gate (`torchrun`, both ranks): draft token ids and the verifier's
  inputs bitwise equal with and without sharding, greedy and probabilistic drafting (fp32 and fp64 noise, mixed
  batches), eager and CUDA-graph replay, 20 trials per case.
- `<boot>.specbench.txt`: vLLM's own throughput and drafter counts per Spec-Bench cell.
