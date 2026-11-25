#!/usr/bin/env bash
# Fetch the pinned DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, BF16, ~945 MB) and verify its weights.
# Usage: scripts/fetch-drafter.sh DIR
set -euo pipefail

dir=${1:?Usage: scripts/fetch-drafter.sh DIR}
revision=4f9088ea66c21f2b99ed657d1941c306b03e84d2
sha256=217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee
command -v hf >/dev/null || { echo 'Activate the Hugging Face CLI venv from the README quick start.' >&2; exit 1; }
if [[ $dir != /* ]]; then dir=./$dir; fi
while [[ $dir == */ && $dir != / ]]; do dir=${dir%/}; done
parent=$(dirname -- "$dir")
mkdir -p -- "$parent"
stage=$(mktemp -d "$parent/.inkling-drafter.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
hf download incoai/Inkling-2-Mini-DFlash2 config.json model.safetensors README.md sglang_inkling_dflash2.py \
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
weights = stage / "model.safetensors"
data = weights.read_bytes()
if len(data) != 989975616 or hashlib.sha256(data).hexdigest() != expected:
    sys.exit("model.safetensors failed verification (expected 989,975,616 bytes and the pinned SHA-256)")
config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
dflash = config.get("dflash_config")
if not isinstance(dflash, dict):
    sys.exit("config.json failed verification (expected a dflash_config object)")
target_ids = dflash.get("target_layer_ids")
if config.get("architectures") != ["DFlash2DraftModel"] or dflash.get("block_size") != 8 or \
        target_ids != [5, 23, 41, 48] or dflash.get("selector_softcap_mode") != "after_correction":
    sys.exit("config.json failed verification (expected DFlash2DraftModel, block_size 8, target_layer_ids "
             "[5, 23, 41, 48] and selector_softcap_mode after_correction)")
# The draft's aux capture ids are engine-side: vLLM's generic DFlash path adds one to dflash_config.target_layer_ids,
# which selects layer visits (6, 24, 42, 49) on this target and measures near-zero acceptance. Inkling captures at
# 0-based visits, and the target's own DSpark drafter resolves to the same four ids, so the key is written verbatim
# and read as-is (see README, "drafter").
if config.get("eagle_aux_hidden_state_layer_ids") != target_ids:
    config["eagle_aux_hidden_state_layer_ids"] = target_ids
    (stage / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

try:
    os.rename(stage, destination)
except OSError as exc:
    if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR) or not destination.is_dir():
        raise
    # Refresh a previous install after validation, preserving unrelated files.
    for name in ("README.md", "config.json", "model.safetensors", "sglang_inkling_dflash2.py"):
        os.replace(stage / name, destination / name)
print(f"Verified {weights.name}: 989,975,616 bytes, SHA-256 {expected}")
PY
