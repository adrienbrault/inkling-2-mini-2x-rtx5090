# Batch-sharded sampling: sampled output, one request at a time (2026-11-21)

Four boots in the order A C A C on image r8 with `VLLM_DECODE_STEP_BUDGET=1024`. A without, C with
`--enable-batch-sharded-sampling`. On each boot the 11 prompts of `bench/greedy.py` ran one at a time, twice:

- `<boot>.greedy.json`: temperature 0;
- `<boot>.seeded.json`: `bench/greedy.py --seeded`, temperature 1.0 with a fixed seed per prompt.

Every boot's text is identical to A1's on all 11 prompts, at temperature 0 and at temperature 1.0. The seeded texts
differ from the greedy ones on all 11 prompts, so the check covers the sampled path.
