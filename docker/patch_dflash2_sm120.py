"""SM120 RoPE for the DFlash/DFlash2 drafter on consumer Blackwell.

Usage: python3 patch_dflash2_sm120.py /usr/local/lib/python3.14/dist-packages/vllm

WHY. The day-0 vllm/vllm-openai:inkling-2-mini image compiles vLLM's CUDA core for sm_80/89/90/100 (no PTX), so the
drafter's context-KV precompute raises cudaErrorNoKernelImageForDevice on an RTX 5090. Every `_custom_ops` call in
`qwen3_dflash.py` is affected: `rotary_embedding` (the RoPE), `rms_norm` (the hidden norm and the grouped K norm), and
`reshape_and_cache_flash` behind `do_kv_cache_update`. Serving never touches any of them - its layers go through
vllm.ir, whose kernel_config picks Triton/native implementations - so only a drafter that calls `_C` directly hits the
gap. Upstream main carries the same calls; no merged PR touches them (#56552 is open and only dequantizes NVFP4 draft
weights), so this is ours to shim. r15 shims the norms and the RoPE in qwen3_dflash.py, and the KV cache write in
_custom_ops.py (reusing the r8 overlay's kv_scatter; the draft needs kv_cache_dtype "auto", i.e. bf16, because the
scatter is unquantized - the target keeps fp8). The draft's attention backend also has to leave FlashAttention (its
fa3 kernels have no SM12x cubins either): boot with speculative attention_backend TRITON_ATTN.

Same contract as the CUDA op for the case the drafter uses: in-place, key=None, `is_neox_style=True`, full rotary
(rot_dim == head_size), any f32/f16/bf16 cache dtype, positions int64. Bitwise-identical arithmetic (fp32 accumulate,
same pair rotation) with the cache read as-is, so the `.to(dtype)` cast on the hot path disappears too.

Gated at runtime: VLLM_INKLING_DFLASH2_SM120=1 turns it on; default off keeps the patched image equal to the day-0 one.
"""
import hashlib
import sys
from pathlib import Path

TARGET = "model_executor/models/qwen3_dflash.py"
EXPECT = "5f50b3c7eeeb6b840783669362e53e56051759462ac0cf17a02de76d2397543b"
TARGET_OPS = "_custom_ops.py"
EXPECT_OPS = "47ee55673c17b170d7c6f1a5686a5d03d687609b79a043a1fe35173c6e9fcd71"

KERNEL = '''# Overlay r15: SM120 DFlash2 drafter RoPE (Triton). See flan/inkling2/dflash2/overlay/patch_dflash2_sm120.py.
@triton.jit
def _dflash2_rope_neox_kernel(
    K, POS, CACHE,
    embed_dim,
    rot_dim,
    s_tok,
    s_head,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + t).to(tl.int64)
    i = tl.arange(0, BLOCK)
    m = i < embed_dim
    base = K + t * s_tok + h * s_head
    x = tl.load(base + i, mask=m, other=0.0)
    y = tl.load(base + embed_dim + i, mask=m, other=0.0)
    cptr = CACHE + pos * rot_dim
    c = tl.load(cptr + i, mask=m, other=1.0).to(tl.float32)
    s = tl.load(cptr + embed_dim + i, mask=m, other=0.0).to(tl.float32)
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    od = K.dtype.element_ty
    tl.store(base + i, (xf * c - yf * s).to(od), mask=m)
    tl.store(base + embed_dim + i, (yf * c + xf * s).to(od), mask=m)


@triton.jit
def _dflash2_rms_norm_kernel(X, W, Y, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    m = i < n
    base = X + row.to(tl.int64) * n
    x = tl.load(base + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + i, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n
    y = x * tl.rsqrt(var + eps) * w
    tl.store(Y + row.to(tl.int64) * n + i, y.to(Y.dtype.element_ty), mask=m)


def _dflash2_rms_norm_out(out, x, weight, eps) -> None:
    """rms_norm(out, x, weight, eps) for [rows, n] with a 1-D weight."""
    rows, n = x.shape
    _dflash2_rms_norm_kernel[(rows,)](x, weight, out, n, eps, BLOCK=triton.next_power_of_2(n))


@triton.jit
def _dflash2_rms_norm_grouped_kernel(X, W, Y, rows_per_layer, hd, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    layer = pid // rows_per_layer
    i = tl.arange(0, BLOCK)
    m = i < hd
    base = X + pid.to(tl.int64) * hd
    x = tl.load(base + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + layer * hd + i, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hd
    y = x * tl.rsqrt(var + eps) * w
    tl.store(Y + pid.to(tl.int64) * hd + i, y.to(Y.dtype.element_ty), mask=m)


def _dflash2_rms_norm_grouped_out(out, x, weight, eps, rows_per_layer, hd) -> None:
    """rms_norm(out, x, weight[L, hd], eps) over [L, rows, hd]: layer picks the weight row."""
    total = x.numel() // hd
    _dflash2_rms_norm_grouped_kernel[(total,)](
        x, weight, out, rows_per_layer, hd, eps, BLOCK=triton.next_power_of_2(hd)
    )


def _dflash2_rope_neox_inplace(k_flat, positions, cos_sin_cache, head_size) -> None:
    """rotary_embedding(positions, k_flat, None, head_size, cos_sin_cache, is_neox=True), in place."""
    num_tokens, kv = k_flat.shape
    rot_dim = cos_sin_cache.shape[1]
    if rot_dim != head_size:
        raise NotImplementedError(f"dflash2 rope: partial rotary {rot_dim}/{head_size}")
    if kv % head_size:
        raise ValueError("dflash2 rope: kv width is not a multiple of head_size")
    num_heads = kv // head_size
    embed_dim = rot_dim // 2
    if embed_dim > 1024:
        raise NotImplementedError(f"dflash2 rope: embed_dim {embed_dim} > 1024")
    _dflash2_rope_neox_kernel[(num_tokens, num_heads)](
        k_flat, positions, cos_sin_cache,
        embed_dim, rot_dim,
        k_flat.stride(0), head_size,
        BLOCK=triton.next_power_of_2(embed_dim),
    )


'''

OLD_NORM_HIDDEN = '''        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        return normed_context_states'''

NEW_NORM_HIDDEN = '''        if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1":
            _dflash2_rms_norm_out(
                normed_context_states, context_states, self._hidden_norm_weight, self._rms_norm_eps
            )
            return normed_context_states
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        return normed_context_states'''

OLD_NORM_K = '''        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed'''

NEW_NORM_K = '''        if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1":
            num_layers, rows, nkv, hd = all_k.shape
            _dflash2_rms_norm_grouped_out(
                all_k_normed, all_k, self._k_norm_weights, self._rms_norm_eps, rows * nkv, hd
            )
            return all_k_normed
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed'''

OLD_ROPE = '''        all_k_flat = all_k.view(num_layers * num_ctx, -1)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            context_positions.repeat(num_layers),
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )'''

NEW_ROPE = '''        all_k_flat = all_k.view(num_layers * num_ctx, -1)
        positions = context_positions.repeat(num_layers)
        if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1":
            if not self._rope_is_neox:
                raise NotImplementedError("dflash2 SM120 rope: only is_neox_style=True is wired")
            _dflash2_rope_neox_inplace(
                all_k_flat, positions, self._rope_cos_sin_cache, self._rope_head_size
            )
            return
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )'''


CACHE_SHIM = '''# Overlay r15: SM120 KV cache scatter (the _C_cache_ops kernels have no SM12x cubins in this image).
def _dflash2_sm120_kv_scatter(key, value, key_cache, value_cache, slot_mapping) -> bool:
    """True when the write was handled here; False leaves the original op in charge."""
    import os

    if os.environ.get("VLLM_INKLING_DFLASH2_SM120") != "1":
        return False
    if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
        return False  # quantized cache: an fp8 write needs a scale, not wired
    if slot_mapping.numel() == 0:
        return True
    from vllm.models.inkling2.nvidia.ops.kv_write_sm120 import kv_scatter

    kv_scatter(key, value, key_cache, value_cache, slot_mapping)
    return True


'''

CACHE_CALL = "    if _dflash2_sm120_kv_scatter(key, value, key_cache, value_cache, slot_mapping):\n        return\n"


def patch_ops(vllm: Path) -> None:
    path = vllm / TARGET_OPS
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EXPECT_OPS:
        raise SystemExit(f"ABORT: {path} sha256 {digest} != {EXPECT_OPS}")
    if text.count("import torch\n") != 1:
        raise SystemExit("ABORT: _custom_ops import anchor")
    text = text.replace("import torch\n", "import torch\n", 1)
    anchor = "logger = init_logger(__name__)\n"
    if text.count(anchor) != 1:
        raise SystemExit("ABORT: _custom_ops: logger anchor")
    text = text.replace(anchor, anchor + "\n" + CACHE_SHIM, 1)
    for call in (
        "    torch.ops._C_cache_ops.reshape_and_cache(\n",
        "    torch.ops._C_cache_ops.reshape_and_cache_flash(\n",
    ):
        if text.count(call) != 1:
            raise SystemExit(f"ABORT: _custom_ops: {call.strip()} found {text.count(call)} times")
        text = text.replace(call, CACHE_CALL + call, 1)
    path.write_text(text)
    print("SUMMARY patch_dflash2_sm120 cache shim DONE:", path)



# --- over the same env gate: the draft's embedding normalization and the selector's softcap order ---------------
# qwen3_dflash.py and qwen3_dflash2.py hash pins are above; these two files carry the rest of the Inkling binding.
TARGET_DF2 = "model_executor/models/qwen3_dflash2.py"
EXPECT_DF2 = "1dcab5ff1416174af2380eea16d3b74fdad66bb75bcb008a7f9806bbac7bf318"
TARGET_UTILS = "v1/worker/gpu/spec_decode/dflash/utils.py"
EXPECT_UTILS = "5ed8574db58d9c4799ec75b097bcf54693875f8896546cdad4819ca375bb3285"

OLD_EMBED = "        embeds = self.embed_tokens(input_ids)\n"
NEW_EMBED = """        embeds = self.embed_tokens(input_ids)
        # Overlay r15: Inkling drafts were trained on embed_norm(embed_tokens(ids)); the engine only shares the raw
        # table (load_dflash_model attaches the target's norm). None for drafts that do not use one.
        _embed_norm = getattr(self, "embed_norm", None)
        if _embed_norm is not None and os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1":
            embeds = _embed_norm(embeds)
"""

OLD_ATTACH = "    maybe_share_target_embed(dflash_model, draft_inner, target_inner)\n"
NEW_ATTACH = """    maybe_share_target_embed(dflash_model, draft_inner, target_inner)
    # Overlay r15: Inkling drafts read normalized embeddings (models/inkling2/nvidia/model.py:303), and the
    # DSpark head attaches the same norm (dspark.py attach_target). qwen3_dflash.py applies it when set.
    import os

    if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1":
        draft_inner.embed_norm = getattr(target_inner, "embed_norm", None)
"""

OLD_SCORES = """        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )
"""
NEW_SCORES = """        hidden = self.hidden_projection(hidden_states)
        scores = _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )
        # Overlay r15: this checkpoint trains selector_softcap_mode "after_correction": the cap belongs on the
        # corrected scores, not on the unary logits (which the engine caps in LogitsProcessor).
        _cap = getattr(self, "post_softcap", None)
        if _cap:
            scores = _cap * torch.tanh(scores / _cap)
        return scores
"""

OLD_EMBED2 = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale
"""
NEW_EMBED2 = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    def _apply_selector_softcap_mode(self) -> None:
        # Overlay r15: "after_correction" caps unary + correction; the engine predates the field.
        mode = str(self.config.dflash_config.get("selector_softcap_mode", "before_correction"))
        if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1" and mode == "after_correction":
            cap = float(self.config.dflash_config.get("final_logit_softcapping") or 0.0)
            self.candidate_selector.post_softcap = cap if cap > 0 else None
"""

OLD_CREATE = """                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"""
NEW_CREATE = """                prefix=maybe_prefix(prefix, "candidate_selector"),
            )
        self._apply_selector_softcap_mode()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"""

OLD_UNARY = "            soft_cap=softcap if softcap > 0 else None,\n"
NEW_UNARY = """            soft_cap=(
                None
                if os.environ.get("VLLM_INKLING_DFLASH2_SM120") == "1"
                and str(draft_config.get("selector_softcap_mode", "before_correction")) == "after_correction"
                else (softcap if softcap > 0 else None)
            ),
"""


def patch_dflash_utils(vllm: Path) -> None:
    path = vllm / TARGET_UTILS
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EXPECT_UTILS:
        raise SystemExit(f"ABORT: {path} sha256 {digest} != {EXPECT_UTILS}")
    if text.count(OLD_ATTACH) != 1:
        raise SystemExit(f"ABORT: embed-norm attach anchor found {text.count(OLD_ATTACH)} times")
    path.write_text(text.replace(OLD_ATTACH, NEW_ATTACH, 1))
    print("SUMMARY patch_dflash2_sm120 embed-norm attach DONE:", path)


def patch_qwen3_dflash2(vllm: Path) -> None:
    path = vllm / TARGET_DF2
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EXPECT_DF2:
        raise SystemExit(f"ABORT: {path} sha256 {digest} != {EXPECT_DF2}")
    if text.count("import torch\n") != 1:
        raise SystemExit("ABORT: qwen3_dflash2 import anchor")
    text = text.replace("import torch\n", "import os\n\nimport torch\n", 1)
    for old, new, tag in (
        (OLD_SCORES, NEW_SCORES, "selector score softcap"),
        (OLD_CREATE, NEW_CREATE, "selector softcap call"),
        (OLD_EMBED2, NEW_EMBED2, "selector softcap mode"),
        (OLD_UNARY, NEW_UNARY, "unary softcap off"),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"ABORT: {tag} anchor found {text.count(old)} times")
        text = text.replace(old, new, 1)
    path.write_text(text)
    print("SUMMARY patch_dflash2_sm120 selector softcap DONE:", path)


def main() -> None:
    vllm = Path(sys.argv[1])
    patch_ops(vllm)
    patch_dflash_utils(vllm)
    patch_qwen3_dflash2(vllm)
    path = vllm / TARGET
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EXPECT:
        raise SystemExit(f"ABORT: {path} sha256 {digest} != {EXPECT}")
    for old, new, tag in (
        ("import io\n", "import io\nimport os\n", "os import"),
        ("from vllm import _custom_ops as ops\n",
         "from vllm import _custom_ops as ops\nfrom vllm.triton_utils import tl, triton\n", "triton import"),
        ("class DFlashQwen3Attention(nn.Module):", KERNEL + "class DFlashQwen3Attention(nn.Module):", "kernel"),
        (OLD_NORM_HIDDEN, NEW_NORM_HIDDEN, "hidden norm"),
        (OLD_NORM_K, NEW_NORM_K, "grouped K norm"),
        (OLD_ROPE, NEW_ROPE, "rope call"),
        (OLD_EMBED, NEW_EMBED, "embed norm"),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"ABORT: {tag} anchor found {text.count(old)} times")
        text = text.replace(old, new, 1)
    path.write_text(text)
    out = path.read_text()
    print("SUMMARY patch_dflash2_sm120 DONE:", path)
    print("  sha256", hashlib.sha256(out.encode()).hexdigest())


if __name__ == "__main__":
    main()
