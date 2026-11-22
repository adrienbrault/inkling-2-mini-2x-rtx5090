# Abliterated name on the served configuration (image sm120-r13), 2026-11-22

`refusal-summary.json`: 100 harmful and 100 benign JailbreakBench behaviors (JBB-Behaviors revision `886acc35`) sent to both names, 8 at a time with the names interleaved, `max_tokens` 2,048, seed per prompt. Each final answer is scored by keywords as REFUSED or ANSWERED; NO_FINAL means the reasoning ran past 2,048 tokens. "Answered" means the answer does not refuse, not that it is complete or correct. Prompts and answers are not published.

`smoke-ablit.json`: tool call, image input and four sampled generations on `inkling-2-mini-ablit`, all pass.

`mixed.json`: prefix isolation (a ~19K-token prompt sent default, abliterated, abliterated, default: 0, 0, 18,944, 18,944 cached tokens, so the cache is not shared between names), lone repeats at T 0 (identical on warm repeats 4/4 for the default name, 2/4 for the abliterated name), and 256-token code continuations at 8 and 16 streams with all default, all abliterated, and half of each.
