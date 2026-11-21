"""Route the DSpark drafter's KV write through the SM12x Triton scatter (see kv_write_sm120.py)."""
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text()
assert hashlib.sha256(src.encode()).hexdigest() == "4f015dc1efe9d3df9a9d4a1a3143e264d6304c788beb444a563910064e26f7f9", "dspark.py changed"
old = """        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
new = """        impl = cast(FlashAttentionImpl, self.attn.impl)
        if needs_sm12x_kv_scatter(impl.kv_cache_dtype):
            # Consumer Blackwell: the image's compiled reshape_and_cache_flash has no sm_120 code.
            key_cache, value_cache = self.attn.kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
            kv_scatter(k, v, key_cache, value_cache, slot_mapping)
            return
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
imp_old = "from .ops.linear import linear\n"
imp_new = "from .ops.kv_write_sm120 import kv_scatter, needs_sm12x_kv_scatter\nfrom .ops.linear import linear\n"
assert src.count(old) == 1 and src.count(imp_old) == 1
p.write_text(src.replace(old, new).replace(imp_old, imp_new))
