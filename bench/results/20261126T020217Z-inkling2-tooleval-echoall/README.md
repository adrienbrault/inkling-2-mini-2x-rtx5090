# Tool calling with reasoning sent back on every turn (image sm120-r14), 2026-11-26

One boot of image `inkling2-rtx5090:sm120-r14`, TP=2, speculative decoding with 7 draft tokens, 1,048,576-token window, nothing else using the server, 2026-11-26 02:02 to 02:08 UTC. 69 scenarios × 4 trials, temperature 1.0, 8 in parallel, model `inkling-2-mini`.

| | trial mean (sd), median |
| --- | ---: |
| A: tool-eval-bench 2.1.0 | 82.5 (1.3), 82.5 |
| D: 2.1.0 + reasoning echo on every assistant turn | 83.5 (1.3), 83.5 |

D sends each earlier assistant turn's reasoning back as `reasoning_content`, on all 497 tool-calling and all 328 other assistant turns, as pi and Hermes do; the chat template renders reasoning on every earlier turn. It uses [`bench/tooleval_echo.py`](../../tooleval_echo.py) with `ECHO_ALL=1`. The difference from A, +1.0, is within one trial standard deviation. The 2026-11-23 run echoed tool-calling turns only and also showed no difference ([`tooleval-think-r14`](../20261123T203803Z-inkling2-tooleval-think-r14)). Report files have their local `report_path` removed.
