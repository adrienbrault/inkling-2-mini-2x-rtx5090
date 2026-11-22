#!/usr/bin/env python3
"""Measure Inkling KV residency or warm-prefix admission on localhost:8000.

Use an idle test server. Admission needs VLLM_INKLING_CONVPAGE_PROBE=1 and a
readable, continuously captured server log passed as --scheduler-log.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


BASE = "http://127.0.0.1:8000"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
METRIC = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)')
TRACE = re.compile(
    r'INKLING_KV_PROBE (queued|scheduled) id=\S*?(kvprobe-[0-9a-f]+)\S*'
    r' ts=([0-9.]+)(?: cached=(\d+) chunk=(\d+))?'
)
PROBE_CONFIG = "INKLING_KV_PROBE_CONFIG "
PREFIX = 86_016  # 128 aligned in both pool layouts
TOTAL = 90_000


def http(path, timeout, body=None, request_id=None):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(BASE + path, data=data, headers=headers)
    return OPENER.open(request, timeout=timeout)


def tokenize(args, content):
    with http("/tokenize", args.timeout, {
        "model": args.model, "prompt": content, "add_special_tokens": False,
    }) as response:
        tokens = json.load(response)["tokens"]
    if not tokens:
        raise ValueError("the server tokenizer returned zero tokens")
    return tokens


def fit(tokens, n):
    return (tokens * ((n + len(tokens) - 1) // len(tokens)))[:n]


def request_id():
    return "kvprobe-" + uuid.uuid4().hex


def salted(args, corpus, name, length):
    salt = tokenize(args, f"Unique KV probe request {name}\n")
    return salt + fit(corpus, length - len(salt))


def make_job(args, tokens, output_tokens, identifier):
    job = {
        "request_id": identifier,
        "request_body": {
            "model": args.model,
            "request_id": identifier,
            "prompt": tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        "expected_prompt_tokens": len(tokens),
        "expected_output_tokens": output_tokens,
        "first": threading.Event(),
    }
    return job


def stream(job, timeout, barrier=None):
    if barrier is not None:
        barrier.wait(timeout)
    job["submitted"] = time.monotonic()
    try:
        with http("/v1/completions", timeout, job["request_body"], job["request_id"]) as response:
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
                    if choice.get("token_ids") or choice.get("text"):
                        now = time.monotonic()
                        if "first_token" not in job:
                            job["first_token"] = now
                            job["first"].set()
                        job["last_token"] = now
                    if choice.get("finish_reason"):
                        job["finish_reason"] = choice["finish_reason"]
                if item.get("usage"):
                    job["usage"] = item["usage"]
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished"] = time.monotonic()
        job["first"].set()


def metrics(timeout):
    with http("/metrics", timeout) as response:
        exposition = response.read().decode()
    series = []
    for line in exposition.splitlines():
        match = METRIC.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if (name.startswith("vllm:kv_cache_usage") or name in (
            "vllm:num_requests_running", "vllm:num_requests_waiting",
            "vllm:num_preemptions", "vllm:num_preemptions_total",
        )):
            series.append({"metric": name, "labels": labels, "value": float(value)})
    by_name = {}
    for item in series:
        by_name.setdefault(item["metric"], []).append(item["value"])
    usages = next((by_name[key] for key in (
        "vllm:kv_cache_usage_perc", "vllm:kv_cache_usage",
    ) if key in by_name), [])
    if not usages:
        usages = [item["value"] for item in series
                  if item["metric"].startswith("vllm:kv_cache_usage")]
    preemptions = by_name.get("vllm:num_preemptions_total") or by_name.get("vllm:num_preemptions")
    running = by_name.get("vllm:num_requests_running")
    waiting = by_name.get("vllm:num_requests_waiting")
    if not usages or preemptions is None or running is None or waiting is None:
        raise RuntimeError("/metrics lacks KV usage, running, waiting or preemption series")
    if any(not 0 <= value <= 1 for value in usages):
        raise RuntimeError("expected fractional KV usage in [0, 1]")
    return {
        "t": time.monotonic(),
        "running": sum(running),
        "waiting": sum(waiting),
        "preemptions_total": sum(preemptions),
        "kv_usage": max(usages),
        "series": series,
    }


def free_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            timeout=3, check=True, text=True, capture_output=True,
        )
        return dict((index, int(free)) for index, free in
                    (map(str.strip, line.split(",")) for line in result.stdout.splitlines()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def idle(timeout, interval):
    deadline = time.monotonic() + 30
    while True:
        sample = metrics(min(timeout, 10))
        if not (sample["running"] or sample["waiting"]):
            return sample
        if time.monotonic() >= deadline:
            raise RuntimeError("server did not become idle within 30 seconds")
        time.sleep(interval)


def collect_samples(args, stop, samples, gpu_available):
    next_gpu_sample = 0.0
    while not stop.is_set():
        try:
            sample = metrics(min(args.timeout, 5))
            if gpu_available and sample["t"] >= next_gpu_sample:
                sample["free_vram_mib"] = free_vram()
                next_gpu_sample = sample["t"] + 2.0
            samples.append(sample)
        except Exception as exc:
            samples.append({"t": time.monotonic(), "error": str(exc)})
        stop.wait(args.interval)


def read_trace(path, job_ids):
    config, by_id = None, {job: {} for job in job_ids}
    if path is None:
        return config, by_id
    # Docker and systemd prefixes can precede each record.
    with path.open(errors="replace") as lines:
        for line in lines:
            if PROBE_CONFIG in line:
                payload = line.split(PROBE_CONFIG, 1)[1]
                try:
                    config, _ = json.JSONDecoder().raw_decode(payload)
                except ValueError:
                    continue
            match = TRACE.search(line)
            if not match:
                continue
            event, identifier, timestamp, cached, chunk = match.groups()
            if identifier in by_id:
                by_id[identifier].setdefault(event, {
                    "t": float(timestamp),
                    "cached": int(cached) if cached else None,
                    "chunk": int(chunk) if chunk else None,
                })
    return config, by_id


def summarize_request(job, trace, origin, minimum_hit):
    first = job.get("first_token")
    submitted = job["submitted"]
    usage = job.get("usage", {})
    scheduled, queued = trace.get("scheduled"), trace.get("queued")
    cached = scheduled["cached"] if scheduled else (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    issues = []
    if job.get("error"):
        issues.append(job["error"])
    if not job.get("done") or first is None:
        issues.append("incomplete SSE stream")
    if usage.get("prompt_tokens") != job["expected_prompt_tokens"]:
        issues.append("unexpected prompt token count")
    if usage.get("completion_tokens") != job["expected_output_tokens"]:
        issues.append("unexpected completion token count")
    if minimum_hit is not None and (cached is None or cached < minimum_hit):
        issues.append("shared prefix did not hit")
    queue_wait = None
    if scheduled and queued and scheduled["t"] >= queued["t"]:
        queue_wait = scheduled["t"] - queued["t"]
    return {
        "request_id": job["request_id"],
        "submitted_s": submitted - origin,
        "first_token_s": None if first is None else first - origin,
        "last_token_s": job["last_token"] - origin if "last_token" in job else None,
        "finished_s": job["finished"] - origin,
        "ttft_s": None if first is None else first - submitted,
        "scheduler_queue_wait_s": queue_wait,
        "first_scheduled_chunk_tokens": scheduled["chunk"] if scheduled else None,
        "scheduler_timestamps": trace,
        "cached_tokens": cached,
        "usage": usage,
        "finish_reason": job.get("finish_reason"),
        "issues": issues,
    }


def run_cohort(args, n, jobs, *, simultaneous, minimum_hit=None):
    origin = time.monotonic()
    initial = idle(args.timeout, args.interval)
    gpu_available = shutil.which("nvidia-smi") is not None
    if gpu_available:
        initial["free_vram_mib"] = free_vram()
    samples = [initial]
    stop = threading.Event()
    monitor = threading.Thread(target=collect_samples, args=(args, stop, samples, gpu_available))
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=n) as executor:
            if simultaneous:
                barrier = threading.Barrier(n + 1)
                pending = [executor.submit(stream, job, args.timeout, barrier) for job in jobs]
                barrier.wait(args.timeout)
            else:
                pending = []
                for i, job in enumerate(jobs):
                    pending.append(executor.submit(stream, job, args.timeout))
                    if i + 1 < n:
                        job["first"].wait(args.timeout)
            for future in pending:
                future.result()
        # Give the last engine-to-frontend stats update a chance to land.
        time.sleep(args.interval)
        final = idle(args.timeout, args.interval)
    finally:
        stop.set()
        monitor.join()
    samples.append(final)
    samples.sort(key=lambda row: row["t"])
    ids = [job["request_id"] for job in jobs]
    trace_deadline = time.monotonic() + 3
    while True:
        config, trace = read_trace(args.scheduler_log, ids)
        if (args.scheduler_log is None or all("scheduled" in item for item in trace.values())
                or time.monotonic() >= trace_deadline):
            break
        time.sleep(0.1)
    requests = [summarize_request(job, trace[job["request_id"]], origin, minimum_hit) for job in jobs]
    count = args.num_blocks if args.num_blocks is not None else (config or {}).get("num_blocks")
    good_samples = []
    for sample in samples:
        sample["elapsed_s"] = sample.pop("t") - origin
        if "kv_usage" in sample:
            if count is not None:
                sample["held_blocks"] = round(sample["kv_usage"] * (count - 1))
            good_samples.append(sample)
    preemptions = [sample["preemptions_total"] for sample in good_samples]
    delta = preemptions[-1] - preemptions[0]
    monotonic_counter = all(x <= y for x, y in zip(preemptions, preemptions[1:]))
    begin = [row["first_token_s"] for row in requests]
    end = [row["last_token_s"] for row in requests]
    overlap = max(0.0, min(end) - max(begin)) if None not in begin + end else 0.0
    complete = not any(row["issues"] for row in requests)
    exact_waits = all(row["scheduler_queue_wait_s"] is not None for row in requests)
    return {
        "cohort": n,
        "simultaneous_submission": simultaneous,
        "requests": requests,
        "samples": samples,
        "preemption_delta": delta if monotonic_counter else None,
        "peak_running": max(sample["running"] for sample in good_samples),
        "peak_waiting": max(sample["waiting"] for sample in good_samples),
        "all_streams_decode_overlap_s": overlap,
        "completed_without_preemption": complete and monotonic_counter and delta == 0,
        "valid_admission_run": complete and exact_waits,
        "num_gpu_blocks": count,
        "server_config": config,
    }


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def boot_excerpt(path):
    if path is None:
        return None
    keywords = ("kv cache layout", "kv cache group sizes", "GPU KV cache size", "num_gpu_blocks",
                "non-default args", "max_num_batched_tokens", "admission")
    with path.open(errors="replace") as lines:
        return [line.strip() for line in lines if any(key in line for key in keywords)][-100:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capacity", "admission"))
    parser.add_argument("text_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="inkling-2-mini")
    parser.add_argument("--scheduler-log", type=Path, help="log containing probe events and pool configuration")
    parser.add_argument("--boot-log", type=Path, help="save boot configuration lines in the run summary")
    parser.add_argument("--num-blocks", type=int, help="final physical num_gpu_blocks if telemetry is unavailable")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--capacity-launch", choices=("staggered", "simultaneous"), default="staggered")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0 or (args.num_blocks is not None and args.num_blocks <= 1):
        parser.error("interval and timeout must be positive, and num-blocks must exceed one")
    if args.mode == "admission" and args.scheduler_log is None:
        parser.error("admission requires --scheduler-log for per-request queue waits")
    for path in (args.scheduler_log, args.boot_log):
        if path is not None and not path.is_file():
            parser.error(f"log file does not exist: {path}")

    started_utc = datetime.now(timezone.utc)
    output = args.output_dir / (started_utc.strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode
                                + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    with args.text_file.open(encoding="utf-8") as data:
        corpus = tokenize(args, data.read(1_000_000))
    cohorts = (1, 3, 5, 6) if args.mode == "capacity" else (6, 8, 10, 12)
    summary = {
        "mode": args.mode, "started_utc": started_utc.isoformat(), "endpoint": BASE,
        "model": args.model, "boot_log_excerpt": boot_excerpt(args.boot_log), "cohorts": [],
    }
    for n in cohorts:
        jobs = []
        if args.mode == "admission":
            warm_id = request_id()
            prefix = salted(args, corpus, warm_id, PREFIX)
            warm = make_job(args, prefix, 1, warm_id)
            idle(args.timeout, args.interval)
            stream(warm, args.timeout)
            warm_result = summarize_request(warm, {}, warm["submitted"], None)
            save(output / f"warm-{n}.json", warm_result)
            if warm_result["issues"]:
                raise RuntimeError(f"warm pass for {n} failed; see {output / f'warm-{n}.json'}")
        for _ in range(n):
            identifier = request_id()
            tokens = (prefix + salted(args, corpus, identifier, TOTAL - PREFIX)
                      if args.mode == "admission" else salted(args, corpus, identifier, 260_000))
            jobs.append(make_job(args, tokens, 2_000, identifier))
        result = run_cohort(args, n, jobs,
                            simultaneous=args.mode == "admission" or args.capacity_launch == "simultaneous",
                            minimum_hit=PREFIX - 128 if args.mode == "admission" else None)
        filename = f"cohort-{n}.json"
        save(output / filename, result)
        summary["cohorts"].append({
            "cohort": n, "file": filename, "peak_running": result["peak_running"],
            "peak_waiting": result["peak_waiting"], "preemption_delta": result["preemption_delta"],
            "decode_overlap_s": result["all_streams_decode_overlap_s"],
            "verified_overlap_without_preemption": (
                result["completed_without_preemption"] and result["all_streams_decode_overlap_s"] > 0
            ),
            "valid_admission_run": result["valid_admission_run"],
        })
        summary["largest_decoding_cohort_without_preemption"] = max(
            (row["cohort"] for row in summary["cohorts"] if row["verified_overlap_without_preemption"]),
            default=None,
        )
        summary["server_config"] = result["server_config"]
        save(output / "summary.json", summary)
        print(f"{filename}: peak running {result['peak_running']:.0f}, "
              f"waiting {result['peak_waiting']:.0f}, preemptions {result['preemption_delta']}",
              flush=True)
    print(output / "summary.json")
    if args.mode == "admission":
        return 0 if all(row["valid_admission_run"] for row in summary["cohorts"]) else 1
    return 0 if summary["largest_decoding_cohort_without_preemption"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
