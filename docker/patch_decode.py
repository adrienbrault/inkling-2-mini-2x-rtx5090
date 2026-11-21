"""Route two bf16 decode products off cuBLAS onto bitwise-identical Triton kernels (overlay r5).
loop.py: the loop controllers' down projection uses decode_sm120.stream_gemv inside its measured envelope.
sconv.py: the dynamic short-conv tap projection uses Inkling's own sm12x_bf16_gemv for 1 to 16 tokens (bitwise equal
to cuBLAS on its (T, 2304) x (32, 2304) shape, 20 of 20 trials per T, kernels/decode_bench.py). usage: patch_decode.py V"""
import hashlib
import sys
from pathlib import Path

V = Path(sys.argv[1])
EXPECT = {"loop.py": "93983ff9e2ef2e2cdd7569c18408214c8091dc0bc476b1d541636d84cf1efb9d",
          "sconv.py": "2ed04fc8945781fa40b4f35dcb422d8e71eadbb5dbd39c2aec5106b1bda9f058"}
EDITS = {
    "loop.py": [
        ("from .ops.loop import blend_rms_norm, pair_rms_norm\n",
         "from .ops.loop import blend_rms_norm, pair_rms_norm\nfrom .ops.decode_sm120 import stream_gemv, use_stream_gemv\n"),
        ("        hidden = torch.bmm(pair, self.controller_down[streams].transpose(1, 2))\n",
         "        down = self.controller_down[streams]\n"
         "        if use_stream_gemv(pair, down):\n"
         "            hidden = stream_gemv(pair, down)  # bitwise equal to the bmm below in this envelope\n"
         "        else:\n"
         "            hidden = torch.bmm(pair, down.transpose(1, 2))\n"),
    ],
    "sconv.py": [
        ("from .ops.linear import linear_k_groups\n",
         "from .ops.gemv import sm12x_bf16_gemv, use_sm12x_bf16_gemv\nfrom .ops.linear import linear_k_groups\n"),
        ("            return linear_k_groups(x.contiguous(), self.weight_proj, x.dtype)\n        return nn.functional.linear(x, self.weight_proj)\n",
         "            return linear_k_groups(x.contiguous(), self.weight_proj, x.dtype)\n"
         "        if x.dim() == 2 and x.shape[0] <= 16 and use_sm12x_bf16_gemv(x, self.weight_proj):\n"
         "            return sm12x_bf16_gemv(x, self.weight_proj)  # bitwise equal to cuBLAS here\n"
         "        return nn.functional.linear(x, self.weight_proj)\n"),
    ],
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
