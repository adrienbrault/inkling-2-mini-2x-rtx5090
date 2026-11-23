# Overlay r14 (DSpark RoPE without a window-sized table), 2026-11-22

Two boots at the served settings (1,048,576-token window, both names): R1 image r13, T1 image r14 with `VLLM_INKLING_DRAFT_ROPE_ONTHEFLY=1`. `summary.json`:

- KV cache blocks after warmup: 23,734 (R1) and 24,553 (T1), +819 per card: the drafter's fp32 `[1,048,576, 128]` cos/sin table (512 MiB per card) is gone.
- `bench/greedy.py --long` (11 prompts, one at a time): T1 identical to R1 at temperature 0 and with a fixed seed (11/11 each).
- Spec-Bench drafter counters at 1 stream: identical (10,377 drafts, 20,448 accepted tokens on both), so the drafts are bitwise the same; aggregate decode at 1 / 8 / 16 streams 413 / 1,551 / 2,355 (R1) and 415 / 1,557 / 2,352 tok/s (T1).
- Smoke (tool call, image input, sampled output): pass on both.
