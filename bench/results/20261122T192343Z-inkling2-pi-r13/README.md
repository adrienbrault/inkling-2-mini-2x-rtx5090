# Agent replay on the served configuration (image sm120-r13, 1M window, both names served), 2026-11-22

`bench/inkling_replay.py --preset pi --slots N --warmup 150 --duration 300` on `inkling-2-mini`, same boot as `20261122T192436Z-inkling2-specbench-served-r13`. Coding-agent sessions whose prompts grow with every tool call (mostly prefix-cached), replies of 282 tokens median, temperature 1.0; requests submitted in a 150 s window after a 150 s warmup are scored.

| Sessions | aggregate decode | per stream | first token p50 / p90 | drafter acceptance length | errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 524 tok/s | 217 tok/s | 0.16 / 0.70 s | 3.86 | 0 |
| 10 | 685 tok/s | 144 tok/s | 0.22 / 0.82 s | 3.75 | 0 |

`pi-6.json` and `pi-10.json` hold the summaries; per-request records are not published.
