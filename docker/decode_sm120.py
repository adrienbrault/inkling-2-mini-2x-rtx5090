# SPDX-License-Identifier: Apache-2.0
"""Decode-path kernels for Inkling-2-Mini on SM12x (RTX 5090), added by the inkling-2-mini-2x-rtx5090 overlay.

stream_gemv: the loop controllers' small batched projection, `torch.bmm(x, w.transpose(1, 2))` for bf16 x (S, T, K)
    and w (S, N, K) with T <= 16, in one launch. Each (stream, column block) program runs exactly the arithmetic of
    Inkling's `_gemv_kernel` (ops/gemv.py: BN 32, BK 256, MP 16, 4 warps), which matches cuBLAS bitwise on these
    shapes; cuBLAS takes ~17 us for two to four streams at T = 8, one launch of this ~7 us.
    Bitwise equal to cuBLAS (20 of 20 random trials per shape, kernels/decode_bench.py, 2026-11-21) for 2 to 4
    streams at 2 to 16 tokens; cuBLAS takes other kernels for one token or one stream at 8+ tokens, so
    `use_stream_gemv` admits only the measured envelope.
"""
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BN, _BK, _MP = 32, 256, 16


@triton.jit
def _stream_gemv_kernel(X, W, Y, M, N, K, SXS, SXM, SWS, SWN, SYS, SYM,
                        BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.program_id(1).to(tl.int64)
    X += s * SXS
    W += s * SWS
    Y += s * SYS
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SWN + ks[:, None], (cols[None, :] < N) & (ks[:, None] < K), other=0)
        x = tl.load(X + rows[:, None] * SXM + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
    tl.store(Y + rows[:, None] * SYM + cols[None, :], acc.to(Y.dtype.element_ty), (rows[:, None] < M) & (cols[None, :] < N))


def use_stream_gemv(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (current_platform.is_device_capability_family(120) and x.is_cuda and x.dim() == 3 and w.dim() == 3
            and x.dtype == w.dtype == torch.bfloat16 and x.shape[0] == w.shape[0] and x.shape[2] == w.shape[2]
            and 2 <= x.shape[0] <= 4 and 2 <= x.shape[1] <= _MP and x.stride(2) == 1 and w.stride(2) == 1)


def stream_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    s, m, k = x.shape
    n = w.shape[1]
    y = torch.empty((s, m, n), device=x.device, dtype=x.dtype)
    _stream_gemv_kernel[(triton.cdiv(n, _BN), s)](
        x, w, y, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        BN=_BN, BK=_BK, MP=_MP, num_warps=4)
    return y
