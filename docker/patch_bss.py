# SPDX-License-Identifier: Apache-2.0
# Contains vLLM Llama/Qwen3 logits and Inkling fork code, adapted and modified for batch sharding.
"""Let vLLM's batch-sharded sampling engage for Inkling-2 (overlay r7).

vLLM samples each request on one tensor-parallel rank from local vocab shards exchanged all-to-all, instead of
all-gathering the full logits on every rank, but only for models with compute_logits_local (the engine logs
"Disabling batch-sharded sampling: Inkling2LoopForConditionalGeneration does not implement compute_logits_local").
model.py: compute_logits_local = the same head and soft-cap without the TP gather (as llama.py / qwen3.py do; the
soft-cap is element-wise, so each shard's values are the gathered ones). speculator.py: Inkling's DSpark
_sample_sequential writes the full draft-logits cache by batch row, so the cache stays replicated
(supports_batch_sharded_draft_logits False). Engaged by --enable-batch-sharded-sampling.  usage: patch_bss.py V"""
import hashlib
import sys
from pathlib import Path

V = Path(sys.argv[1])
EXPECT = {"nvidia/model.py": "d5a4b105fd4dcf15619b8fd4ff783e6b9b056ae0076d72ba8aafc752c76b797d", "speculator.py": "c2a536adfc6e349bcdbbe5622f26a8b88b7440fc243c2db5d3ea0f21b3ae86b5"}
EDITS = {
    "nvidia/model.py": [(
        "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
        "        return self.logits_processor(self.lm_head, hidden_states)\n",
        "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
        "        return self.logits_processor(self.lm_head, hidden_states)\n\n"
        "    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
        "        # Overlay r7: this rank's vocab shard, for vLLM's batch-sharded sampling.\n"
        "        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)\n")],
    "speculator.py": [(
        "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n",
        "    def supports_batch_sharded_draft_logits(self) -> bool:\n"
        "        # Overlay r7: _sample_sequential writes the full-size cache by batch row.\n"
        "        return False\n\n"
        "    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:\n")],
}
for name, edits in EDITS.items():
    f = V / name
    raw = f.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECT[name]:
        sys.exit(f"{f}: unexpected sha256 (vLLM changed); refusing to patch")
    text = raw.decode()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{f}: anchor found {text.count(old)} times: {old[:60]!r}")
        text = text.replace(old, new)
    f.write_text(text)
    print(f"patched {f}")
