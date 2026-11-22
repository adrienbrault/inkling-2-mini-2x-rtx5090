#!/usr/bin/env python3
"""Fair-prefill probe: how long a newcomer waits behind a long cold prefill, and what that costs the long request.

Per trial (TRIALS, default 3), with one background stream decoding throughout (an agent session mid-reply):
  t=0     a cold salted LONG prompt (default 196,608 tokens), 32 output tokens;
  t=2 s   a cold salted SHORT prompt (~1K tokens) and a MID prompt (~16K tokens), 32 output tokens each.
Solo references on the same boot: SHORT, MID and LONG alone (no background stream).
Reported: TTFT of each, the newcomers' TTFT minus their solo TTFT (the wait they inherit), the long request's TTFT
over its solo TTFT, and the background stream's decode rate during the trial.  usage: newcomer.py OUTDIR [--url URL]"""
import argparse
import json
import random
import statistics as st
import threading
import time
from pathlib import Path

from headline_inkling import HERE, fit, stream

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--long", type=int, default=196_608)
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()
out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
corpus = (HERE / "inputs/code-corpus.txt").read_text()
while len(corpus) < 2_000_000:
    corpus += corpus


def prompt(target, tag):
    salt = f"{tag} {random.SystemRandom().getrandbits(128):032x} {time.time_ns()}\n"

    def mk(n, salt=salt):
        return [{"role": "user", "content": salt + corpus[:n] + "\n\nSummarize the code above in one sentence."}]
    chars, _ = fit(a.url, mk, target, 0, min(len(corpus), target * 8))
    return mk(chars)


def run(msgs, key, n=32):
    r = stream(a.url, msgs, n, key, out)
    if r["cached_tokens"]:
        raise RuntimeError(f"{key}: {r['cached_tokens']} cached tokens on a salted prompt")
    return r


solo = {name: run(prompt(t, f"solo-{name}"), f"solo-{name}")["ttft_s"]
        for name, t in (("short", 1024), ("mid", 16384), ("long", a.long))}
rows = []
for k in range(a.trials):
    msgs = {name: prompt(t, f"t{k}-{name}") for name, t in (("short", 1024), ("mid", 16384), ("long", a.long))}
    bg_msgs = prompt(2048, f"t{k}-bg")
    res = {}

    def go(name, delay):
        time.sleep(delay)
        res[name] = run(msgs[name], f"t{k}-{name}")

    bg = threading.Thread(target=lambda: res.__setitem__("bg", run(bg_msgs, f"t{k}-bg", 6000)))
    bg.start()
    time.sleep(1.0)   # the background stream is decoding before the long prefill arrives
    ts = [threading.Thread(target=go, args=x) for x in (("long", 0.0), ("short", 2.0), ("mid", 2.0))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bg.join()
    row = {"trial": k, "long_ttft_s": round(res["long"]["ttft_s"], 3),
           "short_ttft_s": round(res["short"]["ttft_s"], 3), "mid_ttft_s": round(res["mid"]["ttft_s"], 3),
           "short_wait_s": round(res["short"]["ttft_s"] - solo["short"], 3),
           "mid_wait_s": round(res["mid"]["ttft_s"] - solo["mid"], 3),
           "long_vs_solo": round(res["long"]["ttft_s"] / solo["long"], 3),
           "bg_decode_tok_s": round(res["bg"]["decode_tok_s"] or 0, 1)}
    rows.append(row)
    print(json.dumps(row), flush=True)
summary = {"solo_ttft_s": {k: round(v, 3) for k, v in solo.items()},
           **{f"median_{key}": st.median(r[key] for r in rows)
              for key in ("short_ttft_s", "mid_ttft_s", "long_ttft_s", "short_wait_s", "mid_wait_s", "long_vs_solo",
                          "bg_decode_tok_s")}}
(out / "newcomer-summary.json").write_text(json.dumps({"summary": summary, "trials": rows}, indent=1) + "\n")
print(json.dumps(summary), flush=True)
