# fp8 versus bf16 target KV cache (2026-11-21)

Two boots of the r5 image, bf16 then fp8 target KV (`--kv-cache-dtype fp8`; the drafter's cache stays bf16), each
with Spec-Bench c1/c4/c8, a cold-prefill ladder, a served-path smoke test and a fidelity probe. `screen.log` has the
summary lines.

- KV pool: 481,529 tokens (bf16) and 699,259 tokens (fp8).
- Paired NLL over the same 8,191- and 32,767-token code texts (`fid-*/nll-*.json`): fp8 minus bf16 −0.00099 nats/token
  (se 0.00148) and +0.00109 (se 0.00073).
- Greedy text differs from bf16 after 0–204 characters (fp8 changes the arithmetic; not bitwise).
- The `needle` lines (0 of 27 for both caches) are not a retrieval result: at the minimum reasoning effort the model
  spent the whole answer budget reasoning and returned no answer text. They are kept as recorded.
