"""SM12x: use vLLM's unfused vocab-parallel embedding (masked gather + all-reduce, the same values) instead of the
fused ops.vocab_parallel_embedding, which lives in the image's sm_120-less compiled core and fails at TP>1."""
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
src = raw.decode()
if hashlib.sha256(raw).hexdigest() != "0d781e812fbaa4d37373d48dffdc1afaa5a4fe2585791f51dda83017ba4edc53":
    sys.exit("vocab_parallel_embedding.py changed: refusing to patch")
old = """            and isinstance(quant_method, UnquantizedEmbeddingMethod)
        )
"""
new = """            and isinstance(quant_method, UnquantizedEmbeddingMethod)
            # Consumer Blackwell: the compiled fused kernel has no sm_120 code in this image.
            and not current_platform.is_device_capability_family(120)
        )
"""
if src.count(old) != 1 or "is_device_capability_family(120)" in src:
    sys.exit("vocab_parallel_embedding.py: expected exactly one unpatched use_fused_embedding condition")
p.write_text(src.replace(old, new))
