#!/usr/bin/env bash
# Fetch the pinned rank-2 abliteration LoRA and verify its weights.
# Usage: scripts/fetch-adapter.sh DIR
set -euo pipefail

dir=${1:?Usage: scripts/fetch-adapter.sh DIR}
revision=b328f8112173024d0f097aba1ea09e5f475ca7f9
sha256=c67b9d8d18c4a916440eafaf57a42f1510bc00562f52c8a0a29f8f417ed41443
command -v hf >/dev/null || { echo 'Activate the Hugging Face CLI venv from the README quick start.' >&2; exit 1; }
if [[ $dir != /* ]]; then dir=./$dir; fi
while [[ $dir == */ && $dir != / ]]; do dir=${dir%/}; done
parent=$(dirname -- "$dir")
mkdir -p -- "$parent"
stage=$(mktemp -d "$parent/.inkling-adapter.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
hf download OpenUncensored/Inkling-2-Mini-Uncensored-LoRA adapter_config.json adapter_model.safetensors README.md \
  --revision "$revision" --local-dir "$stage"
python3 - "$stage" "$dir" "$sha256" <<'PY'
import errno
import hashlib
import json
import os
import sys
from pathlib import Path

stage, destination = map(Path, sys.argv[1:3])
expected = sys.argv[3]
weights = stage / "adapter_model.safetensors"
data = weights.read_bytes()
if len(data) != 865296 or hashlib.sha256(data).hexdigest() != expected:
    sys.exit("adapter_model.safetensors failed verification (expected 865,296 bytes and the pinned SHA-256)")
try:
    config = json.loads((stage / "adapter_config.json").read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    sys.exit(f"adapter_config.json failed verification: {exc}")
if not isinstance(config, dict) or config.get("peft_type") != "LORA" or config.get("r") != 2:
    sys.exit("adapter_config.json failed verification (expected a rank-2 LoRA)")
if not (stage / "README.md").is_file():
    sys.exit("The adapter download did not include README.md")

try:
    os.rename(stage, destination)
except OSError as exc:
    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR) or not destination.is_dir():
        raise
    # Refresh a previous install after validation, preserving unrelated files.
    for name in ("README.md", "adapter_model.safetensors", "adapter_config.json"):
        os.replace(stage / name, destination / name)
print(f"Verified {weights.name}: 865,296 bytes, SHA-256 {expected}")
PY
