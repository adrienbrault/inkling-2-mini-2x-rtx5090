#!/usr/bin/env python3
"""Measure the real KV capacity of a running Inkling-2-Mini server (default 127.0.0.1:8000).

vLLM logs 818,388 tokens: an admission estimate for 262K requests that charges every windowed / short-conv layer for
its window plus the tokens in flight. This probe holds many long distinct contexts at once and reads vLLM's own
kv_cache_usage: for N cohorts of simultaneous requests (default 8, 12, 16 x 96K prompt tokens, 24,576 forced output
tokens each, long enough that every request is still decoding after the last prefill, salted so nothing is shared), during the window in which every request is decoding, live tokens =
sum(prompt + tokens generated so far) and capacity = live tokens / usage. Also records running / waiting / preemptions
(admission limits). Reuses kv_probe.py (same directory) for HTTP, metrics and salting.
usage: kv_capacity.py TEXT_FILE OUTPUT_DIR [--cohorts 8,12,16] [--prompt-tokens 96000] [--output-tokens 24576]"""
import argparse
import json
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kv_probe as kp  # noqa: E402


def stream(job, timeout, barrier):
    barrier.wait(timeout)
    job["submitted"], job["marks"], n = time.monotonic(), [], 0
    try:
        with kp.http("/v1/completions", timeout, job["request_body"], job["request_id"]) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]":
                    job["done"] = True
                    break
                item = json.loads(chunk)
                if "error" in item:
                    raise RuntimeError(str(item["error"]))
                for choice in item.get("choices", []):
                    k = len(choice.get("token_ids") or [])
                    if k:
                        n += k
                        job["marks"].append((time.monotonic(), n))
                if item.get("usage"):
                    job["usage"] = item["usage"]
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished"] = time.monotonic()


def generated_at(job, t):
    return max((n for ts, n in job["marks"] if ts <= t), default=0)


def run(args, corpus, n):
    jobs = []
    for _ in range(n):
        rid = kp.request_id()
        jobs.append(kp.make_job(args, kp.salted(args, corpus, rid, args.prompt_tokens), args.output_tokens, rid))
    kp.idle(args.timeout, args.interval)
    samples, stop = [], threading.Event()
    mon = threading.Thread(target=kp.collect_samples, args=(args, stop, samples, True))
    mon.start()
    barrier = threading.Barrier(n + 1)
    with ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(stream, j, args.timeout, barrier) for j in jobs]
        barrier.wait(args.timeout)
        for f in futs:
            f.result()
    time.sleep(1)
    stop.set()
    mon.join()
    issues = [j.get("error") or ("incomplete" if not j.get("done") else None) for j in jobs]
    issues += ["prompt tokens" for j in jobs if (j.get("usage") or {}).get("prompt_tokens") != j["expected_prompt_tokens"]]
    firsts = [j["marks"][0][0] for j in jobs if j["marks"]]
    lasts = [j["marks"][-1][0] for j in jobs if j["marks"]]
    t0, t1 = (max(firsts), min(lasts)) if len(firsts) == n else (None, None)
    est = []
    good = [s for s in samples if "kv_usage" in s]
    for s in good:
        if t0 is not None and t0 <= s["t"] <= t1 and s["kv_usage"] > 0.05:
            live = sum(j["expected_prompt_tokens"] + generated_at(j, s["t"]) for j in jobs)
            est.append({"t": s["t"] - t0, "live_tokens": live, "kv_usage": s["kv_usage"],
                        "capacity_tokens": round(live / s["kv_usage"]), "running": s["running"], "waiting": s["waiting"]})
    pre = [s["preemptions_total"] for s in good]
    out = {
        "cohort": n, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
        "issues": [i for i in issues if i], "all_decoding_window_s": None if t0 is None else round(t1 - t0, 2),
        "peak_running": max(s["running"] for s in good), "peak_waiting": max(s["waiting"] for s in good),
        "peak_kv_usage": max(s["kv_usage"] for s in good), "preemption_delta": pre[-1] - pre[0] if pre else None,
        "ttft_s": sorted(round(j["marks"][0][0] - j["submitted"], 2) for j in jobs if j["marks"]),
        "capacity_estimates": est,
        "capacity_tokens_median": round(statistics.median(e["capacity_tokens"] for e in est)) if est else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text_file", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--cohorts", default="8,12,16")
    ap.add_argument("--prompt-tokens", type=int, default=96_000)
    ap.add_argument("--output-tokens", type=int, default=24576)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()
    out = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-kvcap-" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    corpus = kp.tokenize(args, args.text_file.read_text()[:1_000_000])
    summary = []
    for n in (int(x) for x in args.cohorts.split(",")):
        r = run(args, corpus, n)
        (out / f"cohort-{n}.json").write_text(json.dumps(r, indent=1) + "\n")
        row = {k: r[k] for k in ("cohort", "issues", "all_decoding_window_s", "peak_running", "peak_waiting",
                                 "peak_kv_usage", "preemption_delta", "capacity_tokens_median")}
        row["live_tokens_at_start"] = n * args.prompt_tokens
        summary.append(row)
        print(json.dumps(row), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
