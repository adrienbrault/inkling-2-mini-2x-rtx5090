#!/usr/bin/env python3
"""Agent-shaped traffic for Inkling-2-Mini on vLLM: prefill interleaved with in-flight decode, prompts mostly cached.

Shape: growing conversations, mostly prefix-cached, tool output and model reply appended every step, think time
between steps, staggered and respawned sessions. Specific to this engine:
  * chat endpoint (Inkling's template), template-default reasoning effort, temperature 1.0 (the checkpoint default);
  * real code (inputs/code-corpus.txt) instead of word salad: DSpark acceptance sets the decode rate here;
  * scored client-side from the server's per-chunk usage counts (continuous_usage_stats), plus /metrics deltas for
    drafter acceptance (no TabbyAPI log exists to score from).

Session s (slot i of the ladder): base = a salted, slot-unique corpus slice of LADDER base tokens; step k sends
[user: base + task, assistant: reply0, user: tool0, ..., assistant: reply(k-1), user: tool(k-1)] and forces G tokens
(lognormal-ish around --gen); then sleeps --think seconds (gaussian, 40 % sd). A finished slot restarts (--respawn
semantics always on) with a new salt, so the ladder's prompt distribution holds for the whole run.

Window: requests SUBMITTED in [warmup, duration) are scored; nothing is submitted after `duration`; in-flight requests
finish (bounded by --grace). Reported (summary.json):
  per_stream_decode   token-weighted: sum(tokens after the first text chunk) / sum(first-to-last text seconds)
  request_decode_p50  median of per-request decode rates
  aggregate_decode    tokens the scored requests decoded inside the window / window length
  mean_decoding       mean number of scored requests in their decode phase over the window (concurrency actually hit)
  ttft p50 / p90, prompt tokens p50 / p90, cached share (sum cached / sum prompt), stalls (chunk gaps >= 0.25 s),
  forced-length violations, errors, drafter acceptance length (1 + accepted / drafts over the window, /metrics; Spec-Bench's definition).
Validity gates (summary "valid"): no errors, every scored request returned exactly its forced length, cached share
>= 0.85, and at least 30 scored requests.

usage: inkling_replay.py OUTDIR [--url URL] [--sessions-ladder L] [--think S] [--duration S] [--warmup S] ...
"""
import argparse
import json
import math
import random
import statistics as st
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "inkling-2-mini"
# SWE-agent shape: (base tokens, steps) per slot, ~2,700 prompt tokens added per step (gen 700 + tool 2,020).
LADDER_PROD = "3500:9,4000:9,5000:28,6000:23,7000:16,9000:28,9500:7,10500:23,12500:30,20500:19"
TASK = ("\n\nTask: find why the loader refuses the shard and propose a patch. Work step by step; read the tool output "
        "after each step.")


def post_json(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def spec_counters(url):
    try:
        text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in text.splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                  "vllm:spec_decode_num_draft_tokens_total"):
            if line.startswith(k):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def chat(url, messages, n, timeout):
    body = {"model": MODEL, "messages": messages, "max_tokens": n, "min_tokens": n, "ignore_eos": True,
            "temperature": 1.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    marks, usage, text, first_any = [], None, [], None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            now = time.time()
            if d.get("usage"):
                usage = d["usage"]
            piece = ""
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece += (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
            if d.get("choices") and usage:
                if first_any is None:
                    first_any = now
                marks.append((now, usage["completion_tokens"], bool(piece)))
            text.append(piece)
    return t0, first_any, marks, usage, "".join(text)


def draw(rng, p50, p90):
    """Lognormal with the given median and 90th percentile."""
    sigma = math.log(p90 / p50) / 1.2815515655446004
    return p50 * math.exp(rng.gauss(0.0, sigma))


def quantile_draw(rng, table):
    """Inverse-CDF draw, linear between the given (probability, value) points."""
    pts = [tuple(float(x) for x in e.split(":")) for e in table.split(",")]
    u = rng.random()
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if u <= p1:
            return v0 + (v1 - v0) * (u - p0) / (p1 - p0) if p1 > p0 else v1
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--corpus", default=str(HERE / "inputs/code-corpus.txt"))
    ap.add_argument("--ladder", default=LADDER_PROD, help="base:steps per slot (one session slot each)")
    ap.add_argument("--slots", type=int, default=0, help="use this many slots, cycling the ladder (0 = ladder length)")
    ap.add_argument("--gen", type=int, default=700)
    ap.add_argument("--tool", type=int, default=2020)
    ap.add_argument("--think", type=float, default=11.0)
    ap.add_argument("--stagger", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--warmup", type=float, default=90.0)
    ap.add_argument("--grace", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--dist", choices=("gauss", "lognormal"), default="gauss",
                    help="lognormal: --gen / --tool / --think are medians and the --*-p90 options their p90")
    ap.add_argument("--gen-p90", type=float, default=0.0)
    ap.add_argument("--tool-p90", type=float, default=0.0)
    ap.add_argument("--think-p90", type=float, default=0.0)
    ap.add_argument("--think-quantiles", default="", help="p:seconds,... empirical think-time quantile table (overrides --think)")
    ap.add_argument("--preset", choices=("swe", "pi"), default="swe",
                    help="pi: a coding-agent traffic shape (think = autonomous step gaps, see main): prompt p10/p50/p90"
                         " 39K/115K/202K tokens, ~99 %% cached, reply p50/p90 282/1,430 tokens, tool output p50/p90"
                         " 680/6,643 chars, 8 deep long-lived slots (6-10 sessions at once)")
    a = ap.parse_args()
    if a.preset == "pi":
        a.ladder = "15000:40,35000:40,55000:40,75000:40,95000:40,115000:40,135000:40,160000:40"   # + ~1.4K/step
        a.slots = a.slots or 8
        a.dist, a.gen, a.gen_p90, a.tool, a.tool_p90 = "lognormal", 282, 1430, 218, 2129
        # gap from a reply's end to the session's next request (tool run + client), autonomous steps only:
        # p10 0.01, p50 0.05, p75 0.48, p90 5.17, p99 229 s (cap 60)
        a.think_quantiles = "0:0,0.1:0.01,0.5:0.05,0.75:0.48,0.9:5.17,0.99:60,1:60"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(a.corpus).read_text()
    while len(corpus) < 6_000_000:
        corpus += corpus
    probe = corpus[:200_000]
    cpt = len(probe) / post_json(a.url, "/tokenize", {"model": MODEL, "prompt": probe, "add_special_tokens": False})["count"]
    ladder = [tuple(int(x) for x in e.split(":")) for e in a.ladder.split(",") if e.strip()]
    slots = [ladder[i % len(ladder)] for i in range(a.slots or len(ladder))]
    rng0 = random.Random(a.seed)
    t_start = time.time()
    deadline = t_start + a.duration
    lock = threading.Lock()
    sink = open(out / "requests.jsonl", "w")
    errors = []
    m0 = {}

    def slot_worker(i, base_tokens, steps):
        rng = random.Random(rng0.randrange(1 << 30) + i)
        time.sleep(i * a.stagger)
        inst = 0
        while time.time() < deadline:
            salt = f"[session {i}.{inst} {rng.getrandbits(64):016x}]\n"
            off = rng.randrange(0, len(corpus) - 4_000_000)
            base_chars = int(base_tokens * cpt)
            msgs = [{"role": "user", "content": salt + "Repository context:\n" + corpus[off:off + base_chars] + TASK}]
            cur = off + base_chars
            for step in range(steps):
                if time.time() >= deadline:
                    return
                if a.dist == "lognormal":
                    g = int(min(4096, max(16, draw(rng, a.gen, a.gen_p90))))
                else:
                    g = max(64, int(rng.gauss(a.gen, a.gen * 0.5)))
                try:
                    t0, first_any, marks, usage, text = chat(a.url, msgs, g, a.duration + a.grace)
                except Exception as e:  # noqa: BLE001
                    with lock:
                        errors.append({"slot": i, "inst": inst, "step": step, "error": repr(e)[:300], "t": time.time()})
                    return
                tx = [m for m in marks if m[2]] or marks
                rec = {"slot": i, "inst": inst, "step": step, "t_submit": t0, "ttft_s": (first_any - t0) if first_any else None,
                       "first_s": tx[0][0] if tx else None, "last_s": tx[-1][0] if tx else None,
                       "first_tok": tx[0][1] if tx else None, "last_tok": tx[-1][1] if tx else None,
                       "asked": g, "completion_tokens": (usage or {}).get("completion_tokens"),
                       "prompt_tokens": (usage or {}).get("prompt_tokens"),
                       "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                       "marks": [[round(m[0], 4), m[1]] for m in marks]}
                with lock:
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
                tool_tokens = min(8000.0, max(8.0, draw(rng, a.tool, a.tool_p90))) if a.dist == "lognormal" else a.tool
                tool = corpus[cur:cur + int(tool_tokens * cpt)]
                cur += int(tool_tokens * cpt)
                msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "[tool output]\n" + tool}]
                if time.time() + 0.0 >= deadline:
                    return
                if a.think_quantiles:
                    think = quantile_draw(rng, a.think_quantiles)
                elif a.dist == "lognormal":
                    think = min(120.0, draw(rng, a.think, a.think_p90))
                else:
                    think = max(0.0, rng.gauss(a.think, a.think * 0.4))
                time.sleep(think)
            inst += 1

    def metrics_at(t):
        time.sleep(max(0.0, t - time.time()))
        m0[t] = spec_counters(a.url)

    threads = [threading.Thread(target=slot_worker, args=(i, b, s), daemon=True) for i, (b, s) in enumerate(slots)]
    mt = [threading.Thread(target=metrics_at, args=(t_start + a.warmup,), daemon=True),
          threading.Thread(target=metrics_at, args=(deadline,), daemon=True)]
    for t in threads + mt:
        t.start()
    for t in mt:
        t.join()
    for t in threads:
        t.join(timeout=max(0.0, deadline + a.grace - time.time()))
    sink.close()
    w0, w1 = t_start + a.warmup, deadline
    recs = [json.loads(l) for l in open(out / "requests.jsonl")]
    scored = [r for r in recs if w0 <= r["t_submit"] < w1]
    ok = [r for r in scored if r["first_s"] and r["last_s"] and r["last_s"] > r["first_s"]]
    dec_tok = sum(r["last_tok"] - r["first_tok"] for r in ok)
    dec_s = sum(r["last_s"] - r["first_s"] for r in ok)

    def tokens_in(r, t0, t1):
        def at(ts):
            return max((m[1] for m in r["marks"] if m[0] <= ts), default=0)
        return at(t1) - at(t0)
    agg = sum(tokens_in(r, w0, w1) for r in ok) / (w1 - w0)
    busy = sum(max(0.0, min(r["last_s"], w1) - max(r["first_s"], w0)) for r in ok)
    stalls = []
    for r in ok:
        ms = [m for m in r["marks"] if m[0] >= r["first_s"]]
        stalls += [b[0] - a_[0] for a_, b in zip(ms, ms[1:]) if b[0] - a_[0] >= 0.25]
    prompt = sum(r["prompt_tokens"] or 0 for r in scored)
    cached = sum(r["cached_tokens"] or 0 for r in scored)
    forced_bad = [r for r in scored if r["completion_tokens"] != r["asked"]]
    c0, c1 = m0.get(w0, {}), m0.get(w1, {})
    drafts = c1.get("vllm:spec_decode_num_drafts_total", 0) - c0.get("vllm:spec_decode_num_drafts_total", 0)
    acc = c1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - c0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    ttfts = sorted(r["ttft_s"] for r in scored if r["ttft_s"] is not None)
    pts = sorted(r["prompt_tokens"] or 0 for r in scored)
    rates = [(r["last_tok"] - r["first_tok"]) / (r["last_s"] - r["first_s"]) for r in ok]

    def q(xs, p):
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None
    summary = {"window_s": round(w1 - w0, 1), "scored_requests": len(scored), "errors": len(errors),
               "chars_per_token": round(cpt, 3), "slots": len(slots), "preset": a.preset, "dist": a.dist,
               "think_s": a.think, "gen": a.gen, "tool": a.tool,
               "per_stream_decode": round(dec_tok / dec_s, 1) if dec_s else None,
               "request_decode_p50": round(st.median(rates), 1) if rates else None,
               "aggregate_decode": round(agg, 1), "mean_decoding": round(busy / (w1 - w0), 2),
               "ttft_p50_s": q(ttfts, 0.5), "ttft_p90_s": q(ttfts, 0.9),
               "prompt_p50": q(pts, 0.5), "prompt_p90": q(pts, 0.9),
               "cached_share": round(cached / prompt, 3) if prompt else None,
               "stalls_ge_0_25s": len(stalls), "stall_s_total": round(sum(stalls), 2),
               "forced_length_bad": len(forced_bad),
               "acceptance_length": round(1 + acc / drafts, 3) if drafts else None}   # = Spec-Bench mean_acceptance_length
    gens = sorted(r["asked"] for r in scored)
    summary["reply_p50"], summary["reply_p90"] = q(gens, 0.5), q(gens, 0.9)
    # pi preset: deep sessions can outgrow the KV pool; a low cached share is then a finding, not a harness fault
    cache_ok = a.preset == "pi" or (summary["cached_share"] or 0) >= 0.85
    summary["valid"] = bool(not errors and not forced_bad and len(scored) >= 30 and cache_ok)
    (out / "errors.json").write_text(json.dumps(errors, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
