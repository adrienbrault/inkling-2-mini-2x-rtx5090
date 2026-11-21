# SPDX-License-Identifier: Apache-2.0
"""Triton KV-cache scatter for the Inkling 2 DSpark drafter on consumer Blackwell (SM12x).

The vllm/vllm-openai:inkling-2-mini day-0 image builds vLLM's compiled core (_C_stable_libtorch) for sm_80/89/90/100
only, with no PTX, so `reshape_and_cache_flash` raises cudaErrorNoKernelImageForDevice on an RTX 5090. The drafter's
`_write_kv` is the only caller on the serving path (the target model uses its own Triton/Gluon kernels). This kernel
performs the same scatter for the unquantized ("auto") cache: token t goes to slot slot_mapping[t] (negative = padding,
skipped); key/value are [T, H, D]; key_cache/value_cache are the (B, N, H, D) strided views that
FlashAttentionImpl.do_kv_cache_update derives from the (B, H, N, 2*D) cache. Bitwise the same values as the CUDA op:
a plain copy, no arithmetic. Triton compiles for the running GPU, so no arch list is involved.
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _kv_scatter_kernel(
    K, V, KC, VC, SLOT,
    s_kt, s_kh, s_vt, s_vh,
    s_cb, s_cn, s_ch, s_vcb, s_vcn, s_vch,
    BLOCK_N: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(SLOT + t).to(tl.int64)
    if slot < 0:
        return
    b = slot // BLOCK_N
    n = slot % BLOCK_N
    d = tl.arange(0, D)
    for h in range(HEADS):
        k = tl.load(K + t * s_kt + h * s_kh + d)
        v = tl.load(V + t * s_vt + h * s_vh + d)
        tl.store(KC + b * s_cb + n * s_cn + h * s_ch + d, k)
        tl.store(VC + b * s_vcb + n * s_vcn + h * s_vch + d, v)


def kv_scatter(key, value, key_cache, value_cache, slot_mapping):
    """reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", ...) for SM12x."""
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    _, block_n, heads, d = key_cache.shape
    if not (key.shape[-2:] == (heads, d) and value.shape[-2:] == (heads, d)):
        raise ValueError("kv_scatter: key/value shape does not match the cache")
    if not (key.stride(-1) == 1 and value.stride(-1) == 1 and key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1):
        raise ValueError("kv_scatter: head dimension must be contiguous")
    if not (key.dtype == key_cache.dtype and value.dtype == value_cache.dtype):
        raise ValueError("kv_scatter: key/value dtype differs from the cache dtype")
    if d & (d - 1):
        raise ValueError("kv_scatter: head_dim must be a power of two (Inkling's is 128)")
    _kv_scatter_kernel[(num_tokens,)](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        BLOCK_N=block_n, HEADS=heads, D=d,
    )


def needs_sm12x_kv_scatter(kv_cache_dtype: str) -> bool:
    """True on SM12x, where the compiled writer has no code. Only an unquantized cache is supported: an fp8 cache
    would need scaled writes and an fp8-aware draft reader, so it fails here with a clear message instead of reaching
    the missing kernel."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(120):
        return False
    if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise RuntimeError(f"Inkling DSpark on SM12x: kv_cache_dtype {kv_cache_dtype!r} is not supported by this overlay; "
                           "use the default (auto)")
    return True
