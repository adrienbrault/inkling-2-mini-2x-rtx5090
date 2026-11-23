# Tool calling with earlier reasoning sent back (image sm120-r14, 1M window, both names served), 2026-11-23

One boot of `scripts/serve.sh` defaults. tool-eval-bench 2.1.0 dropped each assistant turn's reasoning from the conversation history; 3.0.0 sends it back as `reasoning_content` on tool-calling turns. `B` is 2.1.0 with only that change ([`bench/tooleval_echo.py`](../../tooleval_echo.py)). 69 scenarios × 4 trials, temperature 1.0, 8 in parallel.

| | `inkling-2-mini` | `inkling-2-mini-ablit` |
| --- | ---: | ---: |
| A: tool-eval-bench 2.1.0: trial mean (sd), median | 82.0 (1.4), 82.5 | 78.5 (1.3), 78.5 |
| B: 2.1.0 + reasoning echo | 81.5 (1.3), 81.5 | 77.2 (1.5), 77.0 |
| C: tool-eval-bench 3.0.0 | 87.5 (2.9), 87.5 | 82.2 (2.9), 81.5 |
| B: tool-calling turns with reasoning echoed | 503 of 505 | 556 of 557 |

`echo-probe.json` ([`bench/think_echo_probe.py`](../../think_echo_probe.py)): prompt tokens of one conversation without, and with, earlier reasoning under `reasoning_content`, `reasoning` or both, on a plain and a tool-calling turn. Both field names add the reasoning to the prompt on both turn kinds. `run.log` is the unit log. Report files have their local `report_path` removed.
