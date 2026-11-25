# DFlash2 drafter versus the checkpoint's DSpark head, 2026-11-24

One session, four boots, A B A B: A = `sm120-r14` with the checkpoint's DSpark drafter and the r14 flags (compact conv
pages, draft sharding, in-kernel drafter RoPE), B = `sm120-r15` with the DFlash2 drafter
(`incoai/Inkling-2-Mini-DFlash2`), `DRAFTER=dflash2` and `--block-size 64`. Same target, 1,048,576-token window, both
names, fair prefill, Rust frontend, PCIe IPC all-reduce. Spec-Bench c1 and c4 at 60 prompts per cell, forced 256 tokens,
temperature 0; each arm run twice.

| | DSpark A1 / A2 | DFlash2 B1 / B2 | Change (B vs A) |
| --- | ---: | ---: | ---: |
| Decode median, 1 stream (tok/s) | 384.0 / 385.0 | 525.8 / 524.7 | **+36.6 %** |
| Decode median, 4 streams (tok/s) | 233.1 / 238.9 | 305.0 / 320.7 | **+32.6 %** |
| Aggregate, 1 stream (tok/s) | 426.8 / 428.2 | 549.0 / 547.6 | +28.6 % |
| Aggregate, 4 streams (tok/s) | 981.7 / 991.2 | 1,267.7 / 1,281.4 | +29.2 % |
| Acceptance length, c1 / c4 | 3.061 / 2.906 | 4.172 / 4.022–4.035 | +36 % / +38 % |
| Median TTFT, c1 / c4 (ms) | 32.4 / 59.9 | 33.3 / 60.5 | unchanged |
| KV pool (tokens) | 1,264,522 | 1,357,651 | +7.4 % |
| Greedy text versus A1 | 8/8 | 8/8 | identical |

Both repeats agree on draft counts and acceptance in each cell, so the A-A spread is effectively zero. DFlash2 drafts
fewer tokens for more accepted ones (3,699 drafts / 11,734 accepted at c1, against 5,034 / 10,376), and its smaller
per-step cost shows up in both decode and TTFT-adjacent overheads.

`*.specbench.txt` holds each boot's headline lines and drafter counters, `*.c1.json` the one-stream probe,
`*.greedy.json` the eight lone-request greedy captures, and `screen.log` the boot and restore log. The per-cell detailed
results (per-request inter-token timings) stayed in the measurement session's raw directories; `summary.json` carries
the derived rates.

DFlash2 needs overlay r15: Triton kernels for the drafter's context-KV precompute, the target's embedding norm on the
drafter's inputs, and the selector's `after_correction` softcap order. Its own layer norms and attention RoPE move off
the compiled core through `--ir-op-priority`, and its attention backend is Triton. Without the two binding fixes it
binds but accepts about 0.2 % of its draft tokens.
