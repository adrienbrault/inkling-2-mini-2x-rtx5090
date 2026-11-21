# Batch-sharded sampling: Spec-Bench c8, paired boots (2026-11-21)

Four boots in the order A C A C on image r8, all with `VLLM_DECODE_STEP_BUDGET=1024`. C adds
`--enable-batch-sharded-sampling`. Spec-Bench c8 only (`CELLS=8 bench/specbench.sh`).

| Boot | Per stream (median) | Aggregate | TTFT (median) |
| --- | ---: | ---: | ---: |
| A1 | 166.6 tok/s | 1,465 tok/s | 72 ms |
| C1 | 182.3 tok/s | 1,551 tok/s | 68 ms |
| A2 | 168.9 tok/s | 1,454 tok/s | 72 ms |
| C2 | 179.6 tok/s | 1,558 tok/s | 71 ms |

Greedy text, one request at a time, was identical to A1 on all four boots (8 short and 3 long prompts).
