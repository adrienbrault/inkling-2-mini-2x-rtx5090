#!/usr/bin/env python3
"""Inkling-2-Mini headline speed cells against an OpenAI-compatible vLLM server (default 127.0.0.1:8000).

Prompt lengths are calibrated with the server's own /tokenize, so they are exact for this model and its chat template.

code  : a shared code context (inputs/code-corpus.txt, ExLlamaV3 sources) of 1,024 / 32,768 / 65,536 tokens,
        then per stream a fresh appended "tool output" (the next corpus slice, salted per cell and slot) of 2,048 / 8,192
        tokens plus a frozen HumanEval task (inputs/code-prompts.json, the same selection: c1 HumanEval/0,
        c4 /1-4, c8 /5-12), 400 forced output tokens, temperature 0, reasoning_effort "none". Per depth: one warm
        request (1 token), then per cell a simultaneous 1-token residency replay of the context by every stream, then
        the measured concurrent batch. Per stream: TTFT, decode rate = (tokens - 1) / (last - first token time);
        aggregate = sum of the concurrent streams' rates.
prefill: fresh prompts (a random salt line first, so no cached prefix) of 2K/8K/30K/60K/120K/200K/240K tokens, 1 output
        token, sequential; prompt tokens / TTFT.

usage: headline_inkling.py code|prefill OUTDIR [--url URL] [--corpus PATH] [--prompts PATH]
Raw per-request records (request body sizes, SSE receipt timestamps, usage) are written under OUTDIR.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "inkling-2-mini"
HERE = Path(__file__).resolve().parent


def post(url, path, body, timeout=900):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def count(url, messages):
    """Prompt tokens of a chat request, template included (vLLM /tokenize)."""
    r = post(url, "/tokenize", {"model": MODEL, "messages": messages, "add_generation_prompt": True})
    return r["count"]


def fit(url, make, target, lo, hi):
    """Largest prefix length n (chars) with count(make(n)) <= target; returns (n, tokens)."""
    best = (lo, count(url, make(lo)))
    while hi - lo > 16:
        mid = (lo + hi) // 2
        c = count(url, make(mid))
        if c <= target:
            lo, best = mid, (mid, c)
        else:
            hi = mid
    return best


def stream(url, messages, n, key, out, barrier=None):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": n, "min_tokens": n,
            "ignore_eos": True, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    times, chunks, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                chunks.append(time.perf_counter())   # vLLM streams choices only once a token exists
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    times.append(time.perf_counter())
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    assert usage and usage["completion_tokens"] == n, f"{key}: usage {usage}"
    # TTFT = first streamed choice (a structural token such as Inkling's content marker carries no text); decode is
    # timed on text-bearing chunks only.
    times = times or chunks
    rec = {"key": key, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "finish": finish,
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "ttft_s": chunks[0] - start, "first_s": times[0], "last_s": times[-1], "chunks": len(times),
           "decode_tok_s": (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else None}
    (out / f"{key}.json").write_text(json.dumps(rec) + "\n")
    return rec


def code(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus).read_text()
    prompts = json.loads(Path(args.prompts).read_text())
    tasks = [t for g in prompts["sets"].values() for b in ("warmup", "scored") for batch in g.get(b, []) for t in batch]
    select = {1: tasks[0:1], 4: tasks[1:5], 8: tasks[5:13]}
    rows = []
    for depth in (1024, 32768, 65536):
        def ctx_msgs(n):
            return [{"role": "user", "content": "Repository context follows.\n\n" + corpus[:n]}]
        ctx_chars, ctx_tokens = fit(args.url, ctx_msgs, depth, 64, min(len(corpus), depth * 8))
        ctx = ctx_msgs(ctx_chars)[0]["content"]
        stream(args.url, [{"role": "user", "content": ctx}], 1, f"warm-d{depth}", out)
        for append in (2048, 8192):
            for c in (1, 4, 8):
                cell = f"d{depth}-a{append}-c{c}"
                msgs = []
                for slot, task in enumerate(select[c]):
                    salt = f"[cell {cell} slot {slot} {time.time_ns()}]\n"
                    tail = "\n\nTask:\n" + task["source_prompt"] + "\nComplete the function."

                    def mk(n, salt=salt, tail=tail):
                        return [{"role": "user", "content": ctx + "\n\nTool output:\n" + salt +
                                 corpus[ctx_chars:ctx_chars + n] + tail}]
                    a_chars, total = fit(args.url, mk, ctx_tokens + append, 0, append * 8)
                    msgs.append((slot, mk(a_chars), total))
                gate = threading.Barrier(c)
                with ThreadPoolExecutor(c) as pool:   # residency replay of the shared context
                    list(pool.map(lambda s: stream(args.url, [{"role": "user", "content": ctx}], 1,
                                                   f"confirm-{cell}-s{s[0]}", out, gate), msgs))
                gate = threading.Barrier(c)
                with ThreadPoolExecutor(c) as pool:
                    recs = list(pool.map(lambda s: stream(args.url, s[1], 400, f"measure-{cell}-s{s[0]}", out, gate),
                                         msgs))
                rates = [r["decode_tok_s"] for r in recs]
                row = {"cell": cell, "depth": depth, "context_tokens": ctx_tokens, "append": append, "c": c,
                       "prompt_tokens": [r["prompt_tokens"] for r in recs], "cached": [r["cached_tokens"] for r in recs],
                       "ttft_s": [round(r["ttft_s"], 3) for r in recs], "decode_tok_s": [round(x, 1) for x in rates],
                       "per_stream_mean": round(sum(rates) / c, 1), "aggregate": round(sum(rates), 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    (out / "code-summary.json").write_text(json.dumps(rows, indent=1) + "\n")


def prefill(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus).read_text()
    while len(corpus) < 2_000_000:
        corpus += corpus
    rows = []
    for target in (2048, 8192, 30720, 61440, 122880, 204800, 245760):
        salt = f"{random.SystemRandom().getrandbits(128):032x} {time.time_ns()}\n"

        def mk(n, salt=salt):
            return [{"role": "user", "content": salt + corpus[:n] + "\n\nSummarize the code above in one sentence."}]
        chars, tokens = fit(args.url, mk, target, 0, min(len(corpus), target * 8))
        r = stream(args.url, mk(chars), 1, f"prefill-{target}", out)
        row = {"target": target, "prompt_tokens": r["prompt_tokens"], "cached": r["cached_tokens"],
               "ttft_s": round(r["ttft_s"], 3), "prefill_tok_s": round(r["prompt_tokens"] / r["ttft_s"], 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "prefill-summary.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["code", "prefill"])
    ap.add_argument("outdir")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--corpus", default=str(HERE / "inputs/code-corpus.txt"))
    ap.add_argument("--prompts", default=str(HERE / "inputs/code-prompts.json"))
    a = ap.parse_args()
    {"code": code, "prefill": prefill}[a.mode](a)
