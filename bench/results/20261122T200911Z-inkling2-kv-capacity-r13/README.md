# Real KV capacity of the served configuration (image sm120-r13, 1M window, both names), 2026-11-22

`bench/kv_capacity.py` with cohorts of 8 and 12 distinct 96K-token prompts and 24,576 forced output tokens each; while every request decodes, tokens held divided by vLLM's `kv_cache_usage_perc`.

| Requests | ran at once | peak KV usage | preemptions | tokens held / usage |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 66 % | 0 | 1,420,624 |
| 12 | 11 (the 12th waited for room) | 89 % | 0 | n/a (never all decoding at once) |

`boot-memory.txt`: vLLM's memory lines from the same boot (23,734 KV blocks, 14.49 GiB per card).
