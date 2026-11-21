# Inkling-2-Mini on RTX 5090 (SM120): the vLLM day-0 image plus SM12x fallbacks for two kernels its compiled core
# lacks (built for sm_80/89/90/100): a Triton KV scatter for the DSpark drafter, and the unfused TP embedding.
# Base = vllm/vllm-openai:inkling-2-mini pulled 2026-11-21 (image id below; vLLM 0.1.dev25855+g7cb5dae22).
FROM vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
ARG V=/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia
COPY kv_write_sm120.py $V/ops/kv_write_sm120.py
COPY patch_dspark.py patch_embedding.py /tmp/
RUN python3 /tmp/patch_dspark.py $V/dspark.py && \
    python3 /tmp/patch_embedding.py /usr/local/lib/python3.14/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py && \
    rm /tmp/patch_dspark.py /tmp/patch_embedding.py && \
    python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" $V/dspark.py $V/ops/kv_write_sm120.py && \
    grep -q "kv_scatter(k, v, key_cache, value_cache, slot_mapping)" $V/dspark.py
