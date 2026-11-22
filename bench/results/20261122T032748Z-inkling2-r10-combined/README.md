# Overlay r10 against r8: conv pages and draft sharding together (2026-11-22)

Four boots in the order A C A C. A = image r8 with the r8 served settings. C = image r10 with
`VLLM_INKLING_CONV_PAGE=1`, `VLLM_INKLING_DRAFT_SHARD=1` and `--block-size 64` (the r10 served settings).

- `screen.log`: per boot, the identity checks, the cold-prefill ladder (prompt tokens, tok/s) and Spec-Bench at 1, 4 and
  8 streams (`bench/specbench_summary.py` lines).
- `<boot>.greedy.json` / `<boot>.seeded.json`: `bench/greedy.py --long` (8 short and 3 long prompts, one at a time) at
  temperature 0, and `--seeded` at temperature 1.0 with a fixed seed per prompt. Every boot matches A1 on all 11 prompts
  in both.
- `C1.gate.txt` / `C2.gate.txt`: the conv-page GPU gate inside the serving container (7 cases, 20 trials each, bitwise
  against the unpatched layout).
- `<boot>-prefill/`: the cold-prefill ladder's per-prompt results.
- `<boot>.specbench.txt`: vLLM's own throughput and drafter counts per Spec-Bench cell.

KV pool from the boot logs: 699,885 tokens on A, 818,388 on C.
