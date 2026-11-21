# Inkling-2-Mini on RTX 5090 (SM120): the vLLM day-0 image plus SM12x fallbacks for two kernels its compiled core
# lacks (built for sm_80/89/90/100): a Triton KV scatter for the DSpark drafter, and the unfused TP embedding.
# r5 (2026-11-21): two bf16 decode products moved off cuBLAS onto bitwise-identical Triton kernels (patch_decode.py).
# r6 (draft length < 7) measured and not adopted: dropped from the build.
# r7: compute_logits_local so vLLM's batch-sharded sampling engages (patch_bss.py; serve flag
# --enable-batch-sharded-sampling).
# r8: decode-aware step budget in vLLM's scheduler (patch_sched.py; env VLLM_DECODE_STEP_BUDGET, 0 = off).
# Base = vllm/vllm-openai:inkling-2-mini pulled 2026-11-21 (image id below; vLLM 0.1.dev25855+g7cb5dae22).
FROM vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
ARG V=/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia
COPY kv_write_sm120.py decode_sm120.py $V/ops/
COPY patch_dspark.py patch_embedding.py patch_decode.py patch_bss.py patch_sched.py /tmp/
RUN python3 /tmp/patch_dspark.py $V/dspark.py && \
    python3 /tmp/patch_bss.py $V/.. && \
    python3 /tmp/patch_decode.py $V && \
    python3 /tmp/patch_sched.py /usr/local/lib/python3.14/dist-packages/vllm/v1/core/sched/scheduler.py && \
    python3 /tmp/patch_embedding.py /usr/local/lib/python3.14/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py && \
    rm /tmp/patch_dspark.py /tmp/patch_embedding.py /tmp/patch_decode.py /tmp/patch_bss.py /tmp/patch_sched.py && \
    python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" $V/dspark.py $V/ops/kv_write_sm120.py $V/ops/decode_sm120.py $V/loop.py $V/sconv.py && \
    grep -q "kv_scatter(k, v, key_cache, value_cache, slot_mapping)" $V/dspark.py && \
    grep -q "stream_gemv(pair, down)" $V/loop.py && grep -q "sm12x_bf16_gemv(x, self.weight_proj)" $V/sconv.py && \
    grep -q "def compute_logits_local" $V/model.py && grep -q "Overlay r7: _sample_sequential" $V/../speculator.py && \
    grep -q "Overlay r8: decode-aware" /usr/local/lib/python3.14/dist-packages/vllm/v1/core/sched/scheduler.py
