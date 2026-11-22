#!/usr/bin/env bash
# Fetch the pinned rank-2 abliteration LoRA and verify its weights.
# Usage: scripts/fetch-adapter.sh DIR
set -euo pipefail

dir=${1:?Usage: scripts/fetch-adapter.sh DIR}
revision=b328f8112173024d0f097aba1ea09e5f475ca7f9
sha256=c67b9d8d18c4a916440eafaf57a42f1510bc00562f52c8a0a29f8f417ed41443
command -v hf >/dev/null || { echo 'Install the Hugging Face CLI with pip install -U huggingface_hub.' >&2; exit 1; }
if [[ $dir != /* ]]; then dir=./$dir; fi
mkdir -p -- "$dir"
dir=$(cd -- "$dir" && pwd)
hf download OpenUncensored/Inkling-2-Mini-Uncensored-LoRA adapter_config.json adapter_model.safetensors README.md \
  --revision "$revision" --local-dir "$dir"
python3 - "$dir/adapter_model.safetensors" "$sha256" <<'PY'
import hashlib
import sys
from pathlib import Path

weights, expected = Path(sys.argv[1]), sys.argv[2]
data = weights.read_bytes()
if len(data) != 865296 or hashlib.sha256(data).hexdigest() != expected:
    sys.exit("adapter_model.safetensors failed verification (expected 865,296 bytes and the pinned SHA-256)")
print(f"Verified {weights.name}: 865,296 bytes, SHA-256 {expected}")
PY
