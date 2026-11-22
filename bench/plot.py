#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the README's figures from the committed raw results.

    uv run bench/plot.py            # writes docs/img/decode.svg and docs/img/prefill.svg

Every figure reads bench/results/<run>/ and fails if a file is missing, so no figure carries a number that is not in
this repository. Each prints the values it drew.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"
SPEC = RES / "20261122T122338Z-inkling2-specbench-served-r10-fair"
SPEC_OFF = RES / "20261121T012150Z-inkling2-specbench-nospec-r3"
CODE = RES / "20261122T122632Z-inkling2-code-r10-fair"
PREFILL = RES / "20261122T122925Z-inkling2-prefill-r10-fair-a"
CS = (1, 4, 8, 16)
CS_OFF = (1, 4, 8)   # the drafter-off run predates the c16 cell


def specbench(d, cs=CS):
    """Per-stream mean decode rate and aggregate (c x mean request rate) per concurrency."""
    per, agg = [], []
    for c in cs:
        r = json.loads((d / f"specbench-c{c}.json").read_text())
        rates = [(n - 1) / sum(i) for n, i in zip(r["output_lens"], r["itls"]) if n > 1 and sum(i) > 0]
        per.append(st.mean(rates))
        agg.append(c * st.mean(rates))
    return per, agg


def decode():
    on_p, on_a = specbench(SPEC)
    off_p, off_a = specbench(SPEC_OFF, CS_OFF)
    rows = json.loads((CODE / "code-summary.json").read_text())
    code = {r["c"]: [] for r in rows}
    for r in rows:
        if r["append"] == 2048:
            code[r["c"]].append((r["per_stream_mean"], r["aggregate_shared_window"] or r["aggregate_sum_of_rates"]))
    code_p = [st.mean(x[0] for x in code[c]) for c in CS]
    code_a = [st.mean(x[1] for x in code[c]) for c in CS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in ((ax1, (code_a, on_a, off_a), "Aggregate decode (tok/s)"),
                            (ax2, (code_p, on_p, off_p), "Decode rate per stream, mean (tok/s)")):
        ax.plot(CS, data[0], "o-", color="#2563eb", label="code, 1K-64K cached + 2K append (mean of 3 depths)")
        ax.plot(CS, data[1], "o-", color="#b45309", label="Spec-Bench")
        ax.plot(CS_OFF, data[2], "o--", color="#9ca3af", label="Spec-Bench, drafter off (NCCL all-reduce, image r3, 2026-11-21)")
        ax.set_xticks(CS)
        ax.set_xlabel("concurrent streams")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc="upper left")
    fig.suptitle("Inkling-2-Mini, 2x RTX 5090, TP2 + DSpark, 2026-11-22 (r10, fair prefill)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "decode.svg")
    print("decode code agg", [round(x) for x in code_a], "per", [round(x) for x in code_p])
    print("decode spec agg", [round(x) for x in on_a], "per", [round(x) for x in on_p])
    print("decode off agg", [round(x) for x in off_a], "per", [round(x) for x in off_p])


def prefill():
    rows = json.loads((PREFILL / "prefill-summary.json").read_text())
    x = [r["prompt_tokens"] / 1000 for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, [r["e2e_prefill_tok_s"] for r in rows], "o-", color="#2563eb")
    ax1.set_xlabel("prompt tokens (thousands), cold")
    ax1.set_ylabel("prefill rate, end to end (tok/s)", color="#2563eb")
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["ttft_s"] for r in rows], "s--", color="#b45309")
    ax2.set_ylabel("time to first token (s)", color="#b45309")
    ax2.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.set_title("Inkling-2-Mini cold prefill, 2x RTX 5090, 2026-11-22 (r10, fair prefill)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "prefill.svg")
    print("prefill", [(r["prompt_tokens"], r["e2e_prefill_tok_s"], r["ttft_s"]) for r in rows])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "inkling"   # deterministic SVG ids
    plt.rcParams["svg.fonttype"] = "none"
    decode()
    prefill()
