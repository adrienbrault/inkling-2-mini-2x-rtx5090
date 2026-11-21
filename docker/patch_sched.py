"""Decode-aware step budget for vLLM's V1 scheduler (overlay r8; off unless VLLM_DECODE_STEP_BUDGET is set).

While any running request is decoding, cap the step's scheduled tokens at VLLM_DECODE_STEP_BUDGET (0 or unset = off),
so a chunked prefill interleaves with the decoders in smaller slices. A lone prefill (nobody decoding) keeps the full
max_num_batched_tokens budget: its chunk boundaries, and so its output and cold-prefill rate, stay as without the
patch. Motivation (agent-traffic replays, 2026-11-21): a global --max-num-batched-tokens 1024 gave +12-15 % aggregate
decode at 10 concurrent agent sessions but changed long lone-prompt outputs (0/3 identical) and cost 5-10 % cold
prefill; per-request caps (--long-prefill-token-threshold) kept both but gained only 2-3 %.
usage: patch_sched.py SCHEDULER_PY"""
import hashlib
import sys
from pathlib import Path

P = Path(sys.argv[1])
EXPECT = "3d22eaf8c429a2cc4e481c8249228180659df6ce53b5916da9c45b7339e9f443"
src = P.read_text()
got = hashlib.sha256(src.encode()).hexdigest()
if got != EXPECT:
    sys.exit(f"patch_sched: {P} sha256 {got} != expected {EXPECT}")
EDITS = [
    ("import itertools\nimport time\n",
     "import itertools\nimport os\nimport time\n\n"
     "# Overlay r8: decode-aware step budget (0 = off).\n"
     "_DECODE_STEP_BUDGET = int(os.environ.get(\"VLLM_DECODE_STEP_BUDGET\", \"0\"))\n"),
    ("        token_budget = self.max_num_scheduled_tokens\n"
     "        spec = self.vllm_config.speculative_config\n",
     "        token_budget = self.max_num_scheduled_tokens\n"
     "        # Overlay r8: while any running request decodes, cap this step's tokens so prefills interleave in\n"
     "        # smaller slices; a lone prefill keeps the full budget.\n"
     "        if 0 < _DECODE_STEP_BUDGET < token_budget and any(\n"
     "            not r.is_prefill_chunk for r in self.running\n"
     "        ):\n"
     "            token_budget = _DECODE_STEP_BUDGET\n"
     "        spec = self.vllm_config.speculative_config\n"),
]
for old, new in EDITS:
    if src.count(old) != 1:
        sys.exit(f"patch_sched: anchor found {src.count(old)} times: {old[:60]!r}")
    src = src.replace(old, new)
P.write_text(src)
print("patch_sched: applied")
