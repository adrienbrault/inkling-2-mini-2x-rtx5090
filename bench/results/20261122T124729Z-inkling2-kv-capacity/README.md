# Real KV capacity of the served configuration (image sm120-r10, fair prefill), 2026-11-22

vLLM logs `GPU KV cache size: 818,388 tokens` (`default/server-kv-lines.txt`). That figure is its admission estimate for requests of the full 262,144-token window: it charges every sliding-window and short-conv cache for its whole window plus the tokens in flight. At run time those caches hold only their window, and only the full-attention history grows with the context.

`bench/kv_capacity.py` sends N distinct prompts at once (salted, so nothing is shared), forces a long output from each, and while every request is decoding divides the tokens held (prompt + tokens generated so far) by vLLM's own `vllm:kv_cache_usage_perc`. `scheduler-timeline.txt` in each directory is vLLM's own 10-second scheduler log (running / waiting / KV usage).

| Directory | Requests | Prompt / output tokens | Ran at once | Peak KV usage | Preemptions | Tokens held / usage |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `default` | 8 | 96,000 / 24,576 | 8 | 61 % | 0 | 1,508,118 |
| `default` | 12 | 96,000 / 24,576 | 12 | 91 % | 0 | 1,507,624 |
| `c14-80k` | 14 | 80,000 / 16,384 | 14 | 88 % | 0 | 1,486,835 |
| `default` | 16 | 96,000 / 24,576 | 12, then 4 more as others finished | 93 % | 0 | n/a |
| `noisr` | 16 | 96,000 / 24,576 | 16 | 100 % | 326 | n/a |
| `c14-4k` | 14 | 96,000 / 4,096 | 11 (the first finished before the last started) | 72 % | 0 | n/a |

- The pool holds about 1.49–1.51M tokens of context at once: 25,200 blocks of 64 tokens of full-attention history (1,612,800 tokens), less roughly 10K tokens of window state per running request.
- With the default `--scheduler-reserve-full-isl`, vLLM admits a new request only when its whole prompt fits. With 16 × 96K prompts it ran 12 and queued the rest, which was where 13 prompts plus their outputs would have exceeded the pool; nothing was preempted.
- `noisr` adds `--no-scheduler-reserve-full-isl`: all 16 were admitted, the outputs then outgrew the pool, and vLLM preempted and recomputed requests 326 times. The default is kept.
