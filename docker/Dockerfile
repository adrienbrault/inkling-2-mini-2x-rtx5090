# Inkling-2-Mini on RTX 5090 (SM120): the vLLM day-0 image plus SM12x fallbacks for two kernels its compiled core
# lacks (built for sm_80/89/90/100): a Triton KV scatter for the DSpark drafter, and the unfused TP embedding.
# r5 (2026-11-21): two bf16 decode products moved off cuBLAS onto Triton kernels (patch_decode.py).
# r7: compute_logits_local so vLLM's batch-sharded sampling engages (patch_bss.py; serve flag
# --enable-batch-sharded-sampling).
# r8: decode-aware step budget in vLLM's scheduler (patch_sched.py; env VLLM_DECODE_STEP_BUDGET, 0 = off).
# r9: exact-size short-conv cache pages (patch_convpage.py; env VLLM_INKLING_CONV_PAGE=1 + --block-size 64; off = r8).
# r10: DSpark draft sharding (patch_draftshard.py; env VLLM_INKLING_DRAFT_SHARD=1; off = r9): each card samples its
# vocabulary shard and the cards exchange winners instead of gathering the full-vocabulary draft logits.
# Base = vllm/vllm-openai:inkling-2-mini pulled 2026-11-21 (image id below; vLLM 0.1.dev25855+g7cb5dae22).
FROM vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
ARG V=/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia
ARG VLLM_DIR=/usr/local/lib/python3.14/dist-packages/vllm
COPY docker/kv_write_sm120.py docker/decode_sm120.py $V/ops/
COPY docker/patch_dspark.py docker/patch_embedding.py docker/patch_decode.py docker/patch_bss.py docker/patch_sched.py docker/patch_convpage.py /tmp/
RUN python3 /tmp/patch_dspark.py $V/dspark.py && \
    python3 /tmp/patch_bss.py $V/.. && \
    python3 /tmp/patch_decode.py $V && \
    python3 /tmp/patch_sched.py /usr/local/lib/python3.14/dist-packages/vllm/v1/core/sched/scheduler.py && \
    python3 /tmp/patch_convpage.py /usr/local/lib/python3.14/dist-packages/vllm && \
    python3 /tmp/patch_embedding.py /usr/local/lib/python3.14/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py && \
    rm /tmp/patch_dspark.py /tmp/patch_embedding.py /tmp/patch_decode.py /tmp/patch_bss.py /tmp/patch_sched.py /tmp/patch_convpage.py && \
    python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" $V/dspark.py $V/ops/kv_write_sm120.py $V/ops/decode_sm120.py $V/loop.py $V/sconv.py && \
    grep -q "kv_scatter(k, v, key_cache, value_cache, slot_mapping)" $V/dspark.py && \
    grep -q "stream_gemv(pair, down)" $V/loop.py && grep -q "sm12x_bf16_gemv(x, self.weight_proj)" $V/sconv.py && \
    grep -q "def compute_logits_local" $V/model.py && grep -q "Overlay r7: _sample_sequential" $V/../speculator.py && \
    grep -q "Overlay r8: decode-aware" /usr/local/lib/python3.14/dist-packages/vllm/v1/core/sched/scheduler.py && \
    grep -q "VLLM_INKLING_CONV_PAGE" /usr/local/lib/python3.14/dist-packages/vllm/models/inkling/nvidia/sconv_swa_attn.py && \
    grep -q "return (KVCacheLayout.BLNHC,)" /usr/local/lib/python3.14/dist-packages/vllm/models/inkling/nvidia/sconv_swa_attn.py && \
    grep -q "INKLING_KV_PROBE_CONFIG" /usr/local/lib/python3.14/dist-packages/vllm/v1/core/sched/scheduler.py
# r10: DSpark Markov draft sampling from vocab shards (patch_draftshard.py; env VLLM_INKLING_DRAFT_SHARD=1).
COPY docker/patch_draftshard.py /tmp/patch_draftshard.py
RUN python3 /tmp/patch_draftshard.py "$VLLM_DIR" && \
    grep -Fq 'def markov_logits_local(' "$VLLM_DIR/models/inkling2/nvidia/dspark.py" && \
    grep -Fq 'VLLM_INKLING_DRAFT_SHARD' "$VLLM_DIR/models/inkling2/speculator.py" && \
    grep -Fq 'tensor_model_parallel_all_gather(packet, dim=0)' "$VLLM_DIR/v1/worker/gpu/sample/inkling_draftshard.py" && \
    grep -Fq 'draft_shard_cache.materialize(' "$VLLM_DIR/v1/worker/gpu/model_runner.py" && \
    rm /tmp/patch_draftshard.py

# r13: skip empty attention LoRA deltas and restore the Inkling head independently.
COPY docker/patch_lora.py /tmp/patch_lora.py
RUN grep -Fq 'self.token_mapping_meta.prepare_tensors(self.token_lora_indices)' "$VLLM_DIR/lora/punica_wrapper/punica_gpu.py" && \
    grep -Fq 'no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool, device="cpu")' "$VLLM_DIR/lora/ops/triton_ops/lora_kernel_metadata.py" && \
    grep -Fq 'no_lora = torch.all(token_lora_mapping == -1)' "$VLLM_DIR/lora/ops/triton_ops/lora_kernel_metadata.py" && \
    grep -Fq 'self.no_lora_flag_cpu[0] = no_lora' "$VLLM_DIR/lora/ops/triton_ops/lora_kernel_metadata.py" && \
    python3 /tmp/patch_lora.py "$VLLM_DIR" && \
    grep -Fq 'VLLM_INKLING_LORA_SKIP' "$V/lora.py" && \
    grep -Fq 'VLLM_INKLING_LORA_HEAD' "$V/model.py" && \
    rm /tmp/patch_lora.py

# r14: compact DSpark RoPE, enabled at model load by VLLM_INKLING_DRAFT_ROPE_ONTHEFLY=1.
COPY docker/patch_ropetable.py /tmp/patch_ropetable.py
RUN python3 /tmp/patch_ropetable.py "$VLLM_DIR" && \
    grep -Fq 'os.environ.get("VLLM_INKLING_DRAFT_ROPE_ONTHEFLY") == "1"' "$V/dspark.py" && \
    grep -Fq 'init_cache=False,' "$V/dspark.py" && \
    grep -Fq 'ON_THE_FLY=on_the_fly,' "$V/ops/dspark.py" && \
    grep -Fq 'ON_THE_FLY=cos_sin_cache.ndim == 1,' "$V/ops/dspark.py" && \
    grep -Fq 'tldevice.cos(phase)' "$V/ops/dspark.py" && \
    grep -Fq 'tldevice.sin(phase)' "$V/ops/dspark.py" && \
    rm /tmp/patch_ropetable.py

COPY LICENSE LICENSE-APACHE THIRD_PARTY.md /usr/share/doc/inkling-overlay/
