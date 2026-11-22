#!/usr/bin/env python3
"""Validate and summarize detailed Spec-Bench results.

Usage: specbench_summary.py RESULT_DIR [CONCURRENCY ...]

Decode rate uses tokens after the first token divided by the sum of inter-token latencies.
Aggregate multiplies the mean request rate by concurrency. Wall output rate uses vLLM's
benchmark duration, which includes TTFT and request turnover.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def counters(path):
    totals = defaultdict(float)
    for line in path.read_text().splitlines():
        metric, value = line.split()[:2]
        totals[metric.split("{", 1)[0]] += float(value)
    return totals


def acceptance(directory, c):
    before, after = (directory / f"spec-{stage}-c{c}.txt" for stage in ("before", "after"))
    if not before.exists() and not after.exists():
        return {}
    prefix = "vllm:spec_decode_num_"
    start, end = counters(before), counters(after)
    delta = {key: end.get(key, 0) - start.get(key, 0) for key in start.keys() | end.keys()}
    if any(value < 0 for value in delta.values()):
        raise SystemExit(f"c{c}: a speculative decoding counter decreased during the cell")
    drafts = delta.get(prefix + "drafts_total", 0)
    accepted = delta.get(prefix + "accepted_tokens_total", 0)
    return {"drafts": drafts, "draft_tokens": delta.get(prefix + "draft_tokens_total", 0), "accepted": accepted,
            "mean_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None}


def summarize(directory, c):
    result = json.loads((directory / f"specbench-c{c}.json").read_text())
    lengths = result["output_lens"]
    if result["completed"] != 120 or result.get("failed", 0) or lengths != [256] * 120:
        raise SystemExit(f"c{c}: expected 120 successful 256-token responses, got "
                         f"{result['completed']} completions with lengths {sorted(set(lengths))}")
    intervals = [sum(itl) for itl in result["itls"]]
    if len(intervals) != len(lengths) or any(duration <= 0 for duration in intervals):
        raise SystemExit(f"c{c}: missing or invalid inter-token timing")
    rates = [(n - 1) / duration for n, duration in zip(lengths, intervals)]
    return {"c": c, "requests": len(rates), "per_stream_decode_mean": round(statistics.mean(rates), 1),
            "per_stream_decode_median": round(statistics.median(rates), 1),
            "aggregate": round(c * statistics.mean(rates), 1), "ttft_median_ms": round(result["median_ttft_ms"], 1),
            "wall_output_tok_s": round(result["output_throughput"], 2), **acceptance(directory, c)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("cells", nargs="*", type=int)
    args = parser.parse_args()
    for c in args.cells or (1, 4, 8, 16):
        print(json.dumps(summarize(args.directory, c)))
