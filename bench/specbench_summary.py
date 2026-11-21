#!/usr/bin/env python3
"""Per-stream decode rate, aggregate and TTFT from `vllm bench serve --save-detailed` results.

Per request: decode rate = (output tokens - 1) / sum(inter-token latencies), i.e. from the first to the last token.
Per-stream decode = median over requests; aggregate = c x mean request decode rate (the sum of the rates of the c
streams running together, as they run in steady state). Wall output tok/s is vLLM's own figure (total output tokens /
benchmark duration), which also counts TTFT and the ramp.   usage: specbench_summary.py RESULT_DIR"""
import json
import statistics
import sys
from pathlib import Path

for c in (1, 4, 8):
    p = Path(sys.argv[1]) / f"specbench-c{c}.json"
    r = json.loads(p.read_text())
    rates = [(n - 1) / sum(itl) for n, itl in zip(r["output_lens"], r["itls"]) if n > 1 and sum(itl) > 0]
    print(json.dumps({"c": c, "requests": len(rates), "per_stream_decode_median": round(statistics.median(rates), 1),
                      "aggregate": round(c * statistics.mean(rates), 1), "ttft_median_ms": round(r["median_ttft_ms"], 1),
                      "wall_output_tok_s": round(r["output_throughput"], 1)}))
