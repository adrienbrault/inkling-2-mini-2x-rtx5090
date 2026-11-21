# Code depth, paired boots: overlay r5 vs the served overlay r8 (2026-11-21)

Six boots in the order A N A N A N, `bench/headline_inkling.py code` on each (`<arm><boot>-code/`).

- A = image r5 without the r7/r8 settings: no `VLLM_DECODE_STEP_BUDGET`, no `--enable-batch-sharded-sampling`.
- N = the served configuration (`scripts/serve.sh`, image r8).

Mean of the six one-stream cells over three boots: A 474.6 tok/s (sd 101), N 523.9 tok/s (sd 61). Eight streams, per
stream: A 156.7, N 157.6. Single one-stream cells range 339–683 tok/s within one configuration: each is one request
whose prompt carries a fresh salt, and the drafter accepts more on some outputs than on others.
