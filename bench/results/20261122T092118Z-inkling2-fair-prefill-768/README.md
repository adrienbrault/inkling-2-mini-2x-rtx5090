# Fair prefill: `--long-prefill-token-threshold 768 --long-prefill-token-threshold-adaptive`, image sm120-r10, 2026-11-22

Four boots in the order A G A G. A = the r10 configuration without the two flags; G = A plus the two flags (the served configuration since this run). With `-adaptive`, vLLM caps each prefilling request's chunk at max(768, 2,048 / number of prefilling requests) only while more than one request is prefilling, so a request served alone is scheduled exactly as without the flags.

| Boot | short newcomer TTFT | mid newcomer TTFT | long prompt TTFT vs alone | cold prefill 2K / 246K | agent replay, 6 / 10 sessions, aggregate decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 20.44 s | 21.53 s | 1.07x | 17,715 / 8,353 tok/s | 532.9 / 730.7 tok/s |
| A2 | 20.50 s | 21.54 s | 1.07x | 17,689 / 8,352 tok/s | 520.1 / 703.0 tok/s |
| G1 | 0.52 s | 6.67 s | 1.16x | 17,765 / 8,353 tok/s | 545.3 / 703.6 tok/s |
| G2 | 0.51 s | 6.66 s | 1.16x | 17,900 / 8,349 tok/s | 523.0 / 678.7 tok/s |

- `*-newcomer/` (`bench/newcomer.py`): one stream decoding in the background, a cold 196,608-token prompt at t=0, then a cold ~1K (short) and ~16K (mid) prompt at t=2 s; medians of three trials, plus each prompt alone.
- `*-prefill/`: the cold-prefill ladder, one prompt at a time.
- `*.greedy.json`: greedy output of 11 prompts (including three long ones) one at a time; identical on all four boots.
- `pi-{6,10}-*.json`: `bench/inkling_replay.py --preset pi` with 6 and 10 sessions, 150 s scored window; summaries only.

With the flags, a short request that arrives behind a long cold prefill gets its first token in 0.5 s instead of 20 s, and the long prompt takes 1.16x its time alone instead of 1.07x. At 10 agent sessions the aggregate decode is 2–4 % lower (both G boots below both A boots); at 6 sessions it is within the A-to-A spread.
