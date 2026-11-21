# Decode-aware step budget: output identity and cold prefill (2026-11-21)

Four boots in the order A P A P. A = image r5 (no scheduler patch). P = image r8 with `VLLM_DECODE_STEP_BUDGET=1024`.
Neither arm had `--enable-batch-sharded-sampling`. Both used the fp8 target KV cache.

- `<boot>.greedy.json`: eight short prompts (512 tokens) and three code prompts of 3,629 / 12,681 / 38,825 tokens
  (256 tokens), one at a time, temperature 0. All four boots are text-identical to A1 on all 11 prompts. The long
  prompts' prefills span several scheduler chunks.
- `<boot>-prefill/`: the cold-prefill ladder (`bench/headline_inkling.py prefill`). A and P agree within 1 % on every
  cell, except P2's first 2K cell (16.9K vs 17.9K tok/s). A later boot with the budget measured 18.0–18.1K there.

The budget only applies while some request is decoding. A request served alone runs exactly as without it.
