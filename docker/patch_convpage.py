"""Compact Inkling2 SM120 conv pages after the r8 overlay.

Usage: python3 patch_convpage.py /path/to/site-packages/vllm
VLLM_INKLING_CONV_PAGE=1 selects BLNHC and exact-width, eight-token pages.
VLLM_INKLING_CONVPAGE_PROBE=1 records scheduler events for kv_probe.py.
"""
import hashlib
import sys
from pathlib import Path


OWNER = "models/inkling/nvidia/sconv_swa_attn.py"
SCHED = "v1/core/sched/scheduler.py"
EXPECT = {
    OWNER: "6b3c610e043ec00ce2a8ea65607eca9f15b1194d0951bda75c0494eac38db48a",
    SCHED: "e72fed9cbeb90523debc4ff3fb759f92968b87035805e1277241543d346424d3",
}

EDITS = {
    OWNER: [
        (
            "from dataclasses import dataclass\n",
            "import os\nfrom dataclasses import dataclass\n",
        ),
        (
            "from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig\n",
            "from vllm.platforms import current_platform\n"
            "from vllm.v1.kv_cache_interface import (\n"
            "    AttentionSpec, KVCacheLayerConfig, KVCacheLayout,\n"
            ")\n",
        ),
        (
            "class InklingConvState(nn.Module, AttentionLayerBase):\n",
            '''class InklingCompactSconvBackend(InklingSconvBackend):
    """Eight-token conv pages in the shared block-outer pool."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # BLNHC preserves the NHC order of the default layer-outer layout.
        return (KVCacheLayout.BLNHC,)

    @classmethod
    def get_supported_kernel_block_sizes(cls, kv_cache_spec=None) -> list[int]:
        return [8]


class InklingConvState(nn.Module, AttentionLayerBase):
''',
        ),
        (
            "        self.block_size = kernel_size\n",
            """        self.block_size = kernel_size
        self._compact_conv_page = (
            replicate_hidden and os.environ.get("VLLM_INKLING_CONV_PAGE") == "1"
        )
        if self._compact_conv_page:
            if (
                not current_platform.is_device_capability_family(120)
                or (tp_size, self.num_kv_heads, raw_head_size) != (2, 4, 1408)
                or kernel_size != 4
            ):
                raise ValueError(
                    "VLLM_INKLING_CONV_PAGE requires SM120 TP2 and "
                    "Inkling2-Mini conv geometry (4 heads, 1408 channels/head, 4 taps)"
                )
            self.head_size = raw_head_size
            self.block_size = 8
""",
        ),
        (
            """    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = AttentionSpec(
""",
            """    def get_attn_backend(self) -> type[AttentionBackend]:
        if self._compact_conv_page:
            return InklingCompactSconvBackend
        return InklingSconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        if self._compact_conv_page:
            cache = vllm_config.cache_config
            parallel = vllm_config.parallel_config
            draft = vllm_config.speculative_config
            if (
                cache.block_size != 64
                or cache.cache_dtype not in ("fp8", "fp8_e4m3")
                or self._dtype != torch.bfloat16
                or draft is None
                or draft.method != "dspark"
                or draft.kv_cache_dtype not in ("auto", "bfloat16")
                or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                or vllm_config.kv_transfer_config is not None
                or parallel.pipeline_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or cache.kv_cache_layout not in (None, "BLNHC")
            ):
                raise ValueError(
                    "VLLM_INKLING_CONV_PAGE needs --block-size 64, FP8 target/BF16 "
                    "DSpark KV, hybrid KV management, BLNHC, and no KV connector, PP or CP"
                )
        spec = AttentionSpec(
""",
        ),
    ],
    SCHED: [
        (
            "import itertools\nimport os\n",
            "import itertools\nimport json\nimport os\n",
        ),
        (
            '_DECODE_STEP_BUDGET = int(os.environ.get("VLLM_DECODE_STEP_BUDGET", "0"))\n',
            '_DECODE_STEP_BUDGET = int(os.environ.get("VLLM_DECODE_STEP_BUDGET", "0"))\n'
            '_INKLING_KV_PROBE = os.environ.get("VLLM_INKLING_CONVPAGE_PROBE") == "1"\n',
        ),
        (
            "        self.kv_cache_config = kv_cache_config\n",
            "        self.kv_cache_config = kv_cache_config\n"
            "        if _INKLING_KV_PROBE:\n"
            "            self._inkling_probe_logged_config = False\n",
        ),
        (
            """                self.running.append(request)
                if num_external_computed_tokens > 0:
""",
            """                self.running.append(request)
                if _INKLING_KV_PROBE and "kvprobe-" in request.request_id:
                    logger.info(
                        "INKLING_KV_PROBE scheduled id=%s ts=%.9f cached=%d chunk=%d",
                        request.request_id, time.monotonic(), num_computed_tokens, num_new_tokens,
                    )
                if num_external_computed_tokens > 0:
""",
        ),
        (
            """            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
""",
            """            self._enqueue_waiting_request(request)
            if _INKLING_KV_PROBE and "kvprobe-" in request.request_id:
                logger.info(
                    "INKLING_KV_PROBE queued id=%s ts=%.9f",
                    request.request_id, time.monotonic(),
                )
                if not self._inkling_probe_logged_config:
                    self._inkling_probe_logged_config = True
                    cfg = self.kv_cache_config
                    logger.info("INKLING_KV_PROBE_CONFIG %s", json.dumps({
                        "num_blocks": self.kv_cache_manager.block_pool.num_gpu_blocks,
                        "layout": self.cache_config.kv_cache_layout,
                        "batch_tokens": self.scheduler_config.max_num_batched_tokens,
                        "in_flight_tokens": self.vllm_config.max_in_flight_tokens,
                        "decode_step_budget": _DECODE_STEP_BUDGET,
                        "scheduler_reserve_full_isl": self.scheduler_reserve_full_isl,
                        "max_active_requests": self.max_num_active_reqs,
                        "watermark": self.scheduler_config.watermark,
                        "groups": [{
                            "manager": g.manager_cls.__name__,
                            "block_tokens": g.block_size,
                            "page_bytes": g.page_size_bytes,
                            "layer_pages": [s.page_size_bytes for s in g.kv_cache_specs.values()],
                            "peak_blocks": g.max_allocated_blocks_per_request(self.vllm_config),
                        } for g in cfg.kv_cache_groups],
                        "tensor_block_strides": sorted({t.block_stride for t in cfg.kv_cache_tensors}),
                    }))
            self.requests[request.request_id] = request
""",
        ),
    ],
}


def apply(root: Path) -> None:
    replacements = {}
    for name, edits in EDITS.items():
        path = root / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != EXPECT[name]:
            raise SystemExit(f"patch_convpage: {name}: sha256 {digest} != {EXPECT[name]}")
        source = raw.decode()
        for old, new in edits:
            count = source.count(old)
            if count != 1:
                raise SystemExit(f"patch_convpage: {name}: anchor found {count} times: {old[:80]!r}")
            source = source.replace(old, new, 1)
        compile(source, str(path), "exec")
        replacements[path] = source.encode()
    for path, replacement in replacements.items():
        path.write_bytes(replacement)
        print(f"patch_convpage: patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    apply(Path(sys.argv[1]))
