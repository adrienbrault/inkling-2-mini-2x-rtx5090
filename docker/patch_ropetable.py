# SPDX-License-Identifier: Apache-2.0
# Contains DSpark code from the vLLM Inkling fork, modified to calculate RoPE in-kernel.
"""Replace the optional Inkling2 DSpark RoPE table with on-the-fly trig (r14).

Usage: python3 patch_ropetable.py /path/to/site-packages/vllm
"""

import ast
import hashlib
import sys
from pathlib import Path
from textwrap import indent

MODEL = "models/inkling2/nvidia/dspark.py"
OPS = "models/inkling2/nvidia/ops/dspark.py"
EXPECT = {
    MODEL: "3d4fc0e65837a5ecc6a14614e84fd9994ff3ceea678bec97eaf68b861ee0638f",
    OPS: "11eab54c16b7e67ca2a35d75c311e207a2f4a9fe9e6e2233206fa912dd915c0a",
}

TABLE = '''        # fp32 [cos | sin] table, read directly by the draft's RoPE kernels.
        self.cos_sin_cache = get_rope(
            config.head_dim,
            max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            dtype=torch.float32,
        ).cos_sin_cache
'''

INIT = '''        if os.environ.get("VLLM_INKLING_DRAFT_ROPE_ONTHEFLY") == "1":
            # Reuse get_rope's default inverse-frequency routine on the loader's
            # device. No full table is constructed or kept in the global cache.
            rope = RotaryEmbedding(
                head_size=config.head_dim,
                rotary_dim=config.head_dim,
                max_position_embeddings=0,
                base=config.rope_theta,
                is_neox_style=True,
                dtype=torch.float32,
                init_cache=False,
            )
            self.register_buffer(
                "cos_sin_cache", rope._compute_inv_freq(config.rope_theta), persistent=False
            )
        else:
'''

LOOKUP = '''    cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
    sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)
'''

TRIG = '''    if ON_THE_FLY:
        inv_freq = tl.load(COS_SIN + half_ar).to(tl.float32)
        phase = pos.to(tl.float32) * inv_freq
        # Use libdevice's float trig, including its large-angle range reduction.
        cos = tldevice.cos(phase)
        sin = tldevice.sin(phase)
    else:
'''

EDITS = {
    MODEL: (
        ("from collections.abc import Iterable\n", "import os\nfrom collections.abc import Iterable\n"),
        (
            "from vllm.model_executor.layers.rotary_embedding import get_rope\n",
            "from vllm.model_executor.layers.rotary_embedding import (\n"
            "    RotaryEmbedding,\n"
            "    get_rope,\n"
            ")\n",
        ),
        (TABLE, INIT + indent(TABLE, "    ")),
    ),
    OPS: (
        (
            "    NORM: tl.constexpr,\n):\n",
            "    NORM: tl.constexpr,\n    ON_THE_FLY: tl.constexpr,\n):\n",
        ),
        (LOOKUP, TRIG + indent(LOOKUP, "    ")),
        (
            "    head_dim = cos_sin_cache.shape[-1]\n",
            "    # A vector holds inverse frequencies, while a matrix holds table rows.\n"
            "    on_the_fly = cos_sin_cache.ndim == 1\n"
            "    head_dim = q_weight.numel() if on_the_fly else cos_sin_cache.shape[-1]\n",
        ),
        ("            NORM=True,\n", "            NORM=True,\n            ON_THE_FLY=on_the_fly,\n"),
        (
            "            NORM=False,\n",
            "            NORM=False,\n            ON_THE_FLY=cos_sin_cache.ndim == 1,\n",
        ),
    ),
}


def apply(root: Path) -> None:
    changes = {}
    for name, expected in EXPECT.items():
        path = root / name
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} != r13 input {expected}")
        source = raw.decode()
        for old, new in EDITS[name]:
            count = source.count(old)
            if count != 1:
                raise ValueError(f"{path}: anchor matched {count} times: {old[:90]!r}")
            source = source.replace(old, new, 1)
        ast.parse(source, filename=str(path))
        changes[path] = source

    for path, source in changes.items():
        path.write_bytes(source.encode())
        print(f"ropetable: patched {path}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        apply(Path(sys.argv[1]))
    except (ValueError, OSError, SyntaxError) as exc:
        sys.exit(f"ropetable: {exc}")


if __name__ == "__main__":
    main()
