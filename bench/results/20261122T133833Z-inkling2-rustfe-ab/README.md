# Rust frontend A/B (image sm120-r10, fair prefill), 2026-11-22

Four boots in the order A1 F1 A2 F2. A: the served configuration with vLLM's Python API server. F: the same with `VLLM_USE_RUST_FRONTEND=1` (vLLM's Rust API server); the engine is unchanged. `*.container.txt` holds each boot's image digest, arguments and environment.

- `*.greedy.json`: `bench/greedy.py --long`, 11 prompts one at a time at temperature 0 (8 short, 3 long). Content, reasoning and completion token counts match A1 on every boot, except that on F1 and F2 one long prompt's reasoning ends in `"\n  "`: the Rust frontend returns trailing reasoning whitespace that the Python frontend drops. The F boots' `usage` omits the empty `prompt_tokens_details.multimodal_tokens` field.
- `*.smoke.json`: a tool call, an image input and four sampled generations at temperature 0.8 with the drafter; all passed.
- `pi-<sessions>-<boot>.json`: `bench/inkling_replay.py --preset pi` summaries at 6 and 10 sessions (150 s scored window, 150 s warmup); per-request records are not published.
- `specbench-summary.json`: Spec-Bench at 8 and 16 streams per boot (`bench/specbench_summary.py`; F1's 8-stream and A1's cells as printed by the driver).

| Boot | 6 sessions: aggregate, first token p50 / p90 | 10 sessions: aggregate, first token p50 / p90 | Spec-Bench 8 / 16 streams aggregate |
| --- | ---: | ---: | ---: |
| A1 | 517 tok/s, 0.26 / 0.71 s | 695 tok/s, 0.32 / 0.84 s | 1,586 / 2,420 tok/s |
| F1 | 533 tok/s, 0.16 / 0.58 s | 703 tok/s, 0.23 / 0.73 s | 1,610 / 2,410 tok/s |
| A2 | 514 tok/s, 0.23 / 0.70 s | 708 tok/s, 0.30 / 0.87 s | 1,611 / 2,456 tok/s |
| F2 | 517 tok/s, 0.15 / 0.55 s | 706 tok/s, 0.20 / 0.79 s | 1,604 / 2,417 tok/s |
