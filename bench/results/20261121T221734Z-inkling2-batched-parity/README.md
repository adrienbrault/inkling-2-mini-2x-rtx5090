# Output under concurrency: batched parity (2026-11-21)

`bench/batched.py`: the 11 prompts of `bench/greedy.py` (8 short, 3 long), each sent twice at once, at temperature 0
and at temperature 1.0 with a fixed per-request seed. That is 22 concurrent requests against 16 sequences, so batches
mix decoding with long prefill chunks and some requests wait. Two passes per boot, four boots in the order A C A C on
image r8 with `VLLM_DECODE_STEP_BUDGET=1024`. C adds `--enable-batch-sharded-sampling`.

No pass reproduces another, not even two passes on one boot of one configuration (A1 pass 1 vs pass 2: 0 of 22
identical). Which requests share a step changes the kernels' arithmetic, so batched output varies from run to run.
Mean length of the common prefix (reasoning and answer text, in characters), over pairs of passes:

| Pair | Temperature 0 | Temperature 1.0, seeded |
| --- | ---: | ---: |
| A vs A | 136 | 100 |
| C vs C | 122 | 97 |
| A vs C | 144 | 103 |

Sharded sampling diverges from the other configuration no sooner than either configuration diverges from itself. One
request at a time, greedy output is identical on every boot.
