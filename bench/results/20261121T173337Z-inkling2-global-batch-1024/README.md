# vLLM's global `--max-num-batched-tokens 1024` (2026-11-21): not used

Four boots in the order A D A D on image r5 with the fp8 target KV cache. D adds `--max-num-batched-tokens 1024`, a
1,024-token cap on every step (vLLM's default here is 2,048).

- Greedy, one request at a time (`<boot>.greedy.json`): D matches A1 on the eight short prompts. It differs on all three
  long prompts (3,629 / 12,681 / 38,825 tokens), whose prefill is now cut into different chunks.
- Cold prefill (`<boot>-prefill/`): D is 5–10 % slower (16.0–16.2K vs 17.8–18.2K tok/s at 2K, 13.4K vs 14.7K at 61K).

The served `VLLM_DECODE_STEP_BUDGET=1024` applies the same cap only while some request is decoding. See
`../20261121T192820Z-inkling2-stepbudget-identity`.
