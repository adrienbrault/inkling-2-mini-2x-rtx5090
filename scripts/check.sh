#!/usr/bin/env bash
# Wait for a running server and smoke-test every name it serves.
# Usage: URL=http://127.0.0.1:8000 scripts/check.sh [REPORT_DIR]
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
export URL=${URL:-http://127.0.0.1:${PORT:-8000}}
python3 - "$root/bench" "${1:-}" "${READY_TIMEOUT:-300}" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

bench, reports, timeout_arg = sys.argv[1:]
url = os.environ["URL"].rstrip("/")

def request(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)

def check(report_dir):
    try:
        timeout = int(timeout_arg)
        if timeout <= 0:
            raise ValueError
    except ValueError:
        sys.exit("READY_TIMEOUT must be a positive integer in seconds")

    print(f"Waiting for {url}/v1/models", flush=True)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            sys.exit(f"Timed out waiting for {url}/v1/models")
        try:
            models = request("/v1/models", timeout=min(5, remaining))
        except HTTPError as exc:
            sys.exit(f"{url}/v1/models returned HTTP {exc.code}")
        except (URLError, TimeoutError):
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        else:
            break

    names = list(dict.fromkeys(model["id"] for model in models["data"]))
    if not names or any(not name for name in names):
        sys.exit("/v1/models returned no usable model names")

    failures = []
    for index, name in enumerate(names, start=1):
        try:
            chat = request("/v1/chat/completions", {
                "model": name,
                "messages": [{"role": "user", "content": "Reply with one word: hello."}],
                "temperature": 0,
                "max_tokens": 512,
                "chat_template_kwargs": {"reasoning_effort": "none"},
            })
            if not (chat["choices"][0]["message"].get("content") or "").strip():
                raise ValueError("empty assistant content")
        except (URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            failures.append(f"{name}: chat failed ({exc})")
            continue

        print(f"{name}: chat passed", flush=True)
        for script, report in (("smoke_features.py", "features"), ("structured_smoke.py", "structured")):
            cmd = [sys.executable, str(Path(bench) / script), str(report_dir / f"{index}-{report}.json")]
            if report == "structured":
                cmd += ["--names", name]
            result = subprocess.run(cmd, env={**os.environ, "SMOKE_MODEL": name})
            if result.returncode:
                failures.append(f"{name}: {script} failed")

    if failures:
        sys.exit("\n".join(failures))
    print(f"Smoke checks passed for {', '.join(names)}")

if reports:
    directory = Path(reports)
    directory.mkdir(parents=True, exist_ok=True)
    check(directory)
else:
    with TemporaryDirectory() as directory:
        check(Path(directory))
PY
