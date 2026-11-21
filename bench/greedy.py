#!/usr/bin/env python3
"""Greedy capture for the Inkling drafter identity check: 8 fixed prompts, temperature 0, 512 tokens, default effort.
Speculative decoding is lossless under greedy sampling, so drafter-on and drafter-off runs must match token for token
(text compared; reasoning and content separately).  usage: greedy.py OUT.json [--long] [--seeded]
--seeded samples at temperature 1.0 with a fixed seed per prompt (the sampled-path identity check).
--long adds three code prompts of ~3K / ~10K / ~30K tokens (256 tokens out, distinct first line so none shares a cached
prefix with another) whose prefill spans several scheduler chunks: the identity check for chunk-size changes."""
import json
import os
import sys
import urllib.request
from pathlib import Path

URL = os.environ.get("URL", "http://127.0.0.1:8000") + "/v1/chat/completions"

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with three doctest examples.",
    "Explain the difference between a mutex and a semaphore in four sentences.",
    "Solve: a train leaves at 14:05 at 80 km/h; another leaves the same station at 14:35 at 110 km/h. When does the second catch up?",
    "Translate into French: 'The quick brown fox jumps over the lazy dog while the farmer sleeps.'",
    "Write a SQL query that returns the three most recent orders per customer from orders(id, customer_id, created_at).",
    "List five failure modes of a distributed lock built on Redis SETNX and how to mitigate each.",
    "Write a bash one-liner that finds the ten largest files under the current directory, human-readable sizes.",
    "Summarize the plot of Romeo and Juliet in exactly three sentences.",
]
jobs = [(p, 512) for p in PROMPTS]
if "--long" in sys.argv[2:]:
    corpus = (Path(__file__).resolve().parent / "inputs/code-corpus.txt").read_text()
    jobs += [(f"Long prompt {k}.\n" + corpus[:n] + "\n\nExplain what the code above does, function by function.", 256)
             for k, n in enumerate((12_000, 40_000, 120_000))]
out = []
for p, n in jobs:
    body = {"model": "inkling-2-mini", "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": p}]}
    if "--seeded" in sys.argv[2:]:   # sampled path: temperature 1.0, a fixed seed per prompt
        body.update(temperature=1.0, seed=2000 + len(out))
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    m = r["choices"][0]["message"]
    out.append({"prompt": p[:200], "reasoning": m.get("reasoning_content") or m.get("reasoning"), "content": m.get("content"),
                "finish": r["choices"][0]["finish_reason"], "usage": r["usage"]})
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("captured", len(out))
