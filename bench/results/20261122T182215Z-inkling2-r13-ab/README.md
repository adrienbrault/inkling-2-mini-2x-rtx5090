# Overlay r13 (LoRA fast path), 2026-11-22

Three boots at the 262K window: A1 image r10 without the adapter, L1 image r10 with the adapter, R1 image r13 with the adapter and `VLLM_INKLING_LORA_SKIP=1 VLLM_INKLING_LORA_HEAD=1`. All on `inkling-2-mini`: greedy and fixed-seed output identical to A1 on L1 and R1 (11/11 each), Spec-Bench aggregate at 1 / 8 / 16 streams, the 10-session agent replay, and vLLM's logged KV estimate (`summary.json`).
