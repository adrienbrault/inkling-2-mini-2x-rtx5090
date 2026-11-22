# Memory on the served configuration (image sm120-r10), 2026-11-22

From the boot log of the recipe measurement boot (the same boot as `20261122T053112Z-inkling2-specbench-served-r10`), `boot-memory.txt` (vLLM's own lines, unedited) and `nvidia-smi` after the cold-prefill passes.

Per card (31.4 GiB usable of 32 GB, `--gpu-memory-utilization 0.92` = 28.88 GiB budget):

| | GiB per card |
| --- | ---: |
| weights and non-torch memory | 10.27 (model loading 10.08) |
| KV cache (25,200 blocks) | 15.38 |
| transient activations kept by the KV allocator | 2.42 |
| CUDA graphs | 0.58 |
| total in the budget | 28.65 |

`nvidia-smi` read 28,650 MiB used of 32,607 MiB on each card after the cold-prefill passes. vLLM reports the pool as 818,388 tokens with 3.12 requests at the full 262,144-token window: an admission estimate that charges every sliding-window and short-conv layer for its window plus the tokens in flight.
