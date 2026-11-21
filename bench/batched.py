#!/usr/bin/env python3
"""Batched parity capture: greedy.py's 8 short + 3 long prompts, each sent twice at once (temperature 0, and
temperature 1.0 with a fixed per-request seed), 22 concurrent requests against max_num_seqs 16, so batches mix
decodes with long prefill chunks and some requests wait and join mid-run. Two configs are equivalent under this test
when their captures match as closely as two captures of one config do (the control: batch timing differs between
runs, so batch-variant kernels can differ even within one config).  usage: batched.py OUT.json"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = os.environ.get("URL", "http://127.0.0.1:8000") + "/v1/chat/completions"

sys.argv[2:2] = ["--long"]
src = (Path(__file__).resolve().parent / "greedy.py").read_text()
ns = {"__file__": str(Path(__file__).resolve().parent / "greedy.py")}
exec(src.split("out = []")[0], ns)   # PROMPTS and jobs only, no requests
jobs = [(p, n, t, k) for k, (p, n) in enumerate(ns["jobs"]) for t in (0, 1.0)]


def run(job):
    p, n, t, k = job
    body = {"model": "inkling-2-mini", "temperature": t, "seed": 1000 + k, "max_tokens": n,
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    m = r["choices"][0]["message"]
    return {"prompt": p[:200], "temperature": t, "seed": 1000 + k, "reasoning": m.get("reasoning_content") or m.get("reasoning"),
            "content": m.get("content"), "finish": r["choices"][0]["finish_reason"], "usage": r["usage"]}


with ThreadPoolExecutor(len(jobs)) as ex:
    out = list(ex.map(run, jobs))
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("captured", len(out))
