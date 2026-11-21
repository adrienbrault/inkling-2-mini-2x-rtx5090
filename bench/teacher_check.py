#!/usr/bin/env python3
"""Is every token the drafter-on server emits under greedy the target model's own argmax?

Speculative decoding is lossless only up to numerics: verifying a block of drafts runs the target on several positions
at once (different GEMM shapes than one-token decode), so near-ties can flip. This replays each greedy generation as a
prompt (prefill never uses the drafter) with prompt_logprobs=2 and reports, for every generated position, whether the
emitted token is rank 1; where it is not, the logprob gap to rank 1 (a tiny gap = a numerical near-tie, a large gap =
a real verification bug).  usage: teacher_check.py [N_PROMPTS] [URL]   (default http://127.0.0.1:8000, model inkling-2-mini)"""
import json
import sys
import urllib.request

PROMPTS = ['Write a Python function that merges overlapping intervals, with three doctest examples.',
    'Explain the difference between a mutex and a semaphore in four sentences.',
    'Solve: a train leaves at 14:05 at 80 km/h; another leaves the same station at 14:35 at 110 km/h. When does the second catch up?', "Translate into French: 'The quick brown fox jumps over the lazy dog while the farmer sleeps.'", 'Write a SQL query that returns the three most recent orders per customer from orders(id, customer_id, created_at).',
    'List five failure modes of a distributed lock built on Redis SETNX and how to mitigate each.',
    'Write a bash one-liner that finds the ten largest files under the current directory, human-readable sizes.',
    'Summarize the plot of Romeo and Juliet in exactly three sentences.']

URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


n = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
tot = flips = 0
gaps = []
for p in PROMPTS[:n]:
    r = post("/v1/chat/completions", {"model": "inkling-2-mini", "temperature": 0, "max_tokens": 512,
                                       "return_token_ids": True, "messages": [{"role": "user", "content": p}]})
    pids, gids = r["prompt_token_ids"], r["choices"][0]["token_ids"]
    c = post("/v1/completions", {"model": "inkling-2-mini", "prompt": pids + gids, "max_tokens": 1, "temperature": 0,
                                  "prompt_logprobs": 2})
    plp = c["choices"][0]["prompt_logprobs"]
    for k, tok in enumerate(gids):
        lp = plp[len(pids) + k]
        ranked = sorted(lp.items(), key=lambda kv: -kv[1]["logprob"])
        top_id, top = int(ranked[0][0]), ranked[0][1]["logprob"]
        tot += 1
        if top_id != tok:
            flips += 1
            gaps.append(top - lp[str(tok)]["logprob"] if str(tok) in lp else float("inf"))
print(json.dumps({"positions": tot, "not_rank1": flips, "gaps": sorted(gaps)}, indent=None))
