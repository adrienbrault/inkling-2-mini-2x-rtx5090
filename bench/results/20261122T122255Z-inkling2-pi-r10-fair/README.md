# Agent replay on the served configuration (image sm120-r10, fair prefill), 2026-11-22

`bench/inkling_replay.py --preset pi` against the served configuration, the same boot as `20261122T122338Z-inkling2-specbench-served-r10-fair`. The preset models coding-agent sessions: long-lived conversations that grow with every tool call (prompt p10 / p50 / p90 of about 39K / 115K / 202K tokens, almost all prefix-cached), replies of 282 tokens median, short gaps between steps. 6 and 10 sessions run at once; requests submitted in a 150 s window after a 60 s warmup are scored, temperature 1.0.

| Sessions | aggregate decode | per stream | first token p50 / p90 | drafter acceptance length |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 517 tok/s | 236 tok/s | 0.25 / 0.74 s | 4.06 |
| 10 | 697 tok/s | 144 tok/s | 0.29 / 0.72 s | 3.79 |

`pi-6.json` and `pi-10.json` hold the summaries; per-request records are not published.
