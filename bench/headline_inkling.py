#!/usr/bin/env python3
"""Inkling-2-Mini headline speed cells against an OpenAI-compatible vLLM server (default 127.0.0.1:8000).

Prompt lengths are calibrated with the server's own /tokenize, so they are exact for this model and its chat template.

code  : a shared code context (inputs/code-corpus.txt, ExLlamaV3 sources) of 1,024 / 32,768 / 65,536 tokens,
        then per stream a fresh appended "tool output" (the next corpus slice, salted per cell and slot) of 2,048 / 8,192
        tokens plus a frozen HumanEval task (inputs/code-prompts.json, the same selection: c1 HumanEval/0,
        c4 /1-4, c8 /5-12; c16 = the 13 tasks then /0-2 again, the file holds 13), 400 forced output tokens, temperature 0, reasoning_effort "none". Per depth: one warm
        request (1 token), then per cell a simultaneous 1-token residency replay of the context by every stream, then
        the measured concurrent batch. Per stream: TTFT, decode rate = (tokens - 1) / (last - first token time);
        aggregate = tokens decoded by all streams during the window in which all are decoding / its length (the sum
        of the per-stream rates is also recorded). Checks: served prompt tokens match the calibration, and the shared
        context is a cache hit on every measured stream (vLLM --enable-prompt-tokens-details).
prefill: fresh prompts (a random salt line first, so no cached prefix; checked) of 2K/8K/30K/60K/120K/200K/240K
        tokens, 1 output token, sequential; end-to-end rate = prompt tokens / client TTFT.

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
    r = post(url, "/tokenize", {"model": MODEL, "messages": messages, "add_generation_prompt": True,
                                "chat_template_kwargs": {"reasoning_effort": "none"}})
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
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    marks, usage, finish = [], None, None   # (receipt time, completion tokens so far, chunk carries text)
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            now = time.perf_counter()
            if d.get("usage"):
                usage = d["usage"]
            text = False
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                text = text or bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if d.get("choices") and usage:
                marks.append((now, usage["completion_tokens"], text))
    if not (usage and usage["completion_tokens"] == n and marks):
        raise RuntimeError(f"{key}: usage {usage}, {len(marks)} chunks")
    # TTFT = the first chunk with a generated token (Inkling's first token can be a structural marker with no text).
    # Decode = tokens after the first text-bearing chunk / time from that chunk to the last one, using the server's
    # per-chunk token counts (a speculative step can emit several tokens in one chunk).
    tx = [m for m in marks if m[2]] or marks
    first, last = tx[0], tx[-1]
    rec = {"key": key, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "finish": finish,
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "ttft_s": marks[0][0] - start, "first_s": first[0], "last_s": last[0], "chunks": len(marks),
           "marks": [[round(m[0], 6), m[1]] for m in marks],
           "decode_tok_s": (last[1] - first[1]) / (last[0] - first[0]) if last[0] > first[0] else None}
    (out / f"{key}.json").write_text(json.dumps(rec) + "\n")
    return rec


def window_aggregate(recs):
    """Tokens decoded by all streams together during the window in which every stream is decoding (from the latest
    first-text chunk to the earliest last chunk), divided by its length. None when the streams do not overlap."""
    t0 = max(r["first_s"] for r in recs)
    t1 = min(r["last_s"] for r in recs)
    if t1 - t0 < 0.05:
        return None

    def at(r, ts):
        return max((m[1] for m in r["marks"] if m[0] <= ts), default=0)
    return round(sum(at(r, t1) - at(r, t0) for r in recs) / (t1 - t0), 1)


def code(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus).read_text()
    prompts = json.loads(Path(args.prompts).read_text())
    tasks = [t for g in prompts["sets"].values() for b in ("warmup", "scored") for batch in g.get(b, []) for t in batch]
    select = {1: tasks[0:1], 4: tasks[1:5], 8: tasks[5:13], 16: tasks[0:13] + tasks[0:3]}
    rows = []
    for depth in (1024, 32768, 65536):
        def ctx_msgs(n):
            return [{"role": "user", "content": "Repository context follows.\n\n" + corpus[:n]}]
        ctx_chars, ctx_tokens = fit(args.url, ctx_msgs, depth, 64, min(len(corpus), depth * 8))
        ctx = ctx_msgs(ctx_chars)[0]["content"]
        stream(args.url, [{"role": "user", "content": ctx}], 1, f"warm-d{depth}", out)
        for append in (2048, 8192):
            for c in (1, 4, 8, 16):
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
                for (slot, _, total), r in zip(msgs, recs):
                    if abs(r["prompt_tokens"] - total) > 8 or abs(total - (ctx_tokens + append)) > 0.02 * append:
                        raise RuntimeError(f"{cell} slot {slot}: prompt {r['prompt_tokens']} vs calibrated {total}")
                    if r["cached_tokens"] is None or r["cached_tokens"] < ctx_tokens - 256:
                        raise RuntimeError(f"{cell} slot {slot}: shared context cache hit failed "
                                           f"({r['cached_tokens']} cached tokens for {ctx_tokens} context tokens)")
                rates = [r["decode_tok_s"] for r in recs]
                row = {"cell": cell, "depth": depth, "context_tokens": ctx_tokens, "append": append, "c": c,
                       "prompt_tokens": [r["prompt_tokens"] for r in recs], "cached": [r["cached_tokens"] for r in recs],
                       "context_cached": True, "completion_tokens": [r["completion_tokens"] for r in recs],
                       "ttft_s": [round(r["ttft_s"], 3) for r in recs], "decode_tok_s": [round(x, 1) for x in rates],
                       "per_stream_mean": round(sum(rates) / c, 1), "aggregate_sum_of_rates": round(sum(rates), 1),
                       "aggregate_shared_window": window_aggregate(recs)}
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
        if r["cached_tokens"] != 0:
            raise RuntimeError(f"prefill {target}: expected zero cached tokens on a salted prompt, got {r['cached_tokens']}")
        if abs(r["prompt_tokens"] - target) > 0.02 * target:
            raise RuntimeError(f"prefill {target}: got {r['prompt_tokens']} prompt tokens")
        # End to end: request upload, queueing, prefill and the first token, as a client sees it.
        row = {"target": target, "prompt_tokens": r["prompt_tokens"], "cached": r["cached_tokens"],
               "ttft_s": round(r["ttft_s"], 3), "e2e_prefill_tok_s": round(r["prompt_tokens"] / r["ttft_s"], 1)}
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
