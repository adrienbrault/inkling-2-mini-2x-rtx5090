# Agent replay on the served configuration (image sm120-r10, fair prefill, Rust frontend), 2026-11-22

`bench/inkling_replay.py --preset pi` against the served configuration, the same boot as `20261122T143353Z-inkling2-specbench-served-rustfe`. The preset models coding-agent sessions: long-lived conversations that grow with every tool call (prompt p10 / p50 / p90 of about 39K / 115K / 202K tokens, almost all prefix-cached), replies of 282 tokens median, short gaps between steps. 6 and 10 sessions run at once; requests submitted in a 150 s window after a 150 s warmup are scored, temperature 1.0.

| Sessions | aggregate decode | per stream | first token p50 / p90 | drafter acceptance length |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 511 tok/s | 211 tok/s | 0.15 / 0.69 s | 3.88 |
| 10 | 711 tok/s | 147 tok/s | 0.23 / 0.72 s | 3.97 |

`pi-6.json` and `pi-10.json` hold the summaries; per-request records are not published.
