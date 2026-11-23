# SPDX-License-Identifier: Apache-2.0
# Contains vLLM sampler and Inkling fork code, modified for sharded DSpark sampling.
"""Apply the opt-in Inkling2 DSpark vocab sharding overlay after r8.

Usage: python3 patch_draftshard.py /path/to/site-packages/vllm
"""
import ast
import hashlib
import sys
from itertools import takewhile
from pathlib import Path
from textwrap import dedent

MODEL = "models/inkling2/nvidia/dspark.py"
SPEC = "models/inkling2/speculator.py"
GUMBEL = "v1/worker/gpu/sample/gumbel.py"
RUNNER = "v1/worker/gpu/model_runner.py"
KERNELS = "v1/worker/gpu/sample/inkling_draftshard.py"

EXPECT = {
    MODEL: "820b0fcb95a20fd138345329007b4e59a3b2db8c528e4947c2d5cc1231b2158f",
    SPEC: "219ccd017728ad873a8e26ce9790616ccaa0a50fa623fedaedfa371912095185",
    GUMBEL: "820184641a0e4ea48bd837d66724a85f1f6fc2edc3bc1354577e2d635c0a68f1",
    RUNNER: "8be0e35c282b19853143334f484178144d30efc18eb0b434a74a3ec34279690d",
}


def class_block(src):
    return "".join(f"    {line}\n" if line else "\n"
                   for line in dedent(src).strip("\n").splitlines())


LOAD_MODEL = class_block('''
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        # Overlay: VLLM_INKLING_DRAFT_SHARD is selected once, before graph capture.
        if os.environ.get("VLLM_INKLING_DRAFT_SHARD") != "1":
            return

        from vllm.distributed.parallel_state import get_tp_group
        from vllm.logger import init_logger
        from vllm.v1.worker.gpu.sample.batch_shard import (
            batch_sharded_sampling_enabled,
            get_tp_pynccl_comm,
        )

        parallel = self.vllm_config.parallel_config
        tp = get_tp_group()
        head = self.model.lm_head
        if (
            tp.world_size != 2
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not batch_sharded_sampling_enabled(self.vllm_config, target_model)
            or get_tp_pynccl_comm() is None
            or self.draft_watermarker is not None
            or self.use_acceptance_estimator
            or self._draft_topk is not None
            or self._d2t_scatter_index is not None
            or self.dtype != torch.bfloat16
            or head is None
            or getattr(head, "tp_size", None) != 2
            or getattr(head, "num_added_embeddings", None) != 0
            or self.vocab_size != self.model.config.vocab_size
        ):
            return

        from vllm.v1.worker.gpu.sample.inkling_draftshard import (
            DraftShardCache,
            DraftShardLayout,
            sample_shard,
        )

        width = head.weight.shape[0]
        layout = DraftShardLayout(width, self.vocab_size, tp.rank_in_group)
        if not layout.supported:
            return
        if self.draft_logits is not None:
            self._draft_shard_cache = DraftShardCache(self.draft_logits, layout)
        self._sample_draft_shard = sample_shard
        self._draft_shard_layout = layout
        init_logger(__name__).info(
            "Inkling2 DSpark vocab-sharded draft sampling enabled (TP2)."
        )

''') + "\n"

EDITS = {
    MODEL: [
        (
            "    def markov_logits(\n",
            class_block('''
                def markov_logits_local(
                    self, base_logits: torch.Tensor, bias: torch.Tensor
                ) -> torch.Tensor:
                    """The pre-gather BF16 logits, in physical TP shard order."""
                    return add_softcap(
                        base_logits, bias, self.config.final_logit_softcapping
                    )

            ''') + "\n    def markov_logits(\n",
        ),
    ],
    SPEC: [
        ("import torch\n", "import os\n\nimport torch\n"),
        (
            "    def supports_batch_sharded_draft_logits(self) -> bool:\n",
            LOAD_MODEL + "    def supports_batch_sharded_draft_logits(self) -> bool:\n",
        ),
        (
            '    _speculator_name = "Inkling2 DSpark"\n',
            '    _speculator_name = "Inkling2 DSpark"\n'
            '    _draft_shard_layout = None\n'
            '    _draft_shard_cache = None\n',
        ),
        (
            "            if self.draft_logits is None:\n"
            "                prev = self.model.markov_argmax(base_logits[:, i], bias)\n",
            "            if self._draft_shard_layout is not None:\n"
            "                local = self.model.markov_logits_local(base_logits[:, i], bias)\n"
            "                prev = self._sample_draft_shard(\n"
            "                    self._draft_shard_layout, local, idx_map[:, i], self.temperature,\n"
            "                    self.seeds, sample_pos[:, i] - 1, i, self.use_fp64_gumbel,\n"
            "                    self._draft_shard_cache,\n"
            "                )\n"
            "            elif self.draft_logits is None:\n"
            "                prev = self.model.markov_argmax(base_logits[:, i], bias)\n",
        ),
    ],
    GUMBEL: [
        (
            "    PER_TOKEN_COL: tl.constexpr = False,\n):\n"
            "    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)\n",
            "    PER_TOKEN_COL: tl.constexpr = False,\n"
            "    VOCAB_OFFSET: tl.constexpr = 0,\n"
            "    DRAFT_STEP: tl.constexpr = None,\n"
            "):\n"
            "    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)\n",
        ),
        (
            "    if logits_cache_ptr is not None:\n"
            "        seed = spec_step_seed(seed, col)\n"
            "    return gumbel_noised_argmax(\n"
            "        logits,\n"
            "        block,\n",
            "    if logits_cache_ptr is not None:\n"
            "        seed = spec_step_seed(seed, col)\n"
            "    elif DRAFT_STEP is not None:\n"
            "        seed = spec_step_seed(seed, DRAFT_STEP)\n"
            "    return gumbel_noised_argmax(\n"
            "        logits,\n"
            "        block + VOCAB_OFFSET,\n",
        ),
        (
            "    HAS_BIAS: tl.constexpr,\n):\n"
            "    token_idx = tl.program_id(0).to(tl.int64)\n",
            "    HAS_BIAS: tl.constexpr,\n"
            "    VOCAB_OFFSET: tl.constexpr = 0,\n"
            "    DRAFT_STEP: tl.constexpr = None,\n"
            "):\n"
            "    token_idx = tl.program_id(0).to(tl.int64)\n",
        ),
        (
            "        PER_TOKEN_COL=PER_TOKEN_COL,\n"
            "    )\n"
            "    token_id = block_idx * BLOCK_SIZE + idx\n",
            "        PER_TOKEN_COL=PER_TOKEN_COL,\n"
            "        VOCAB_OFFSET=VOCAB_OFFSET,\n"
            "        DRAFT_STEP=DRAFT_STEP,\n"
            "    )\n"
            "    token_id = block_idx * BLOCK_SIZE + idx + VOCAB_OFFSET\n",
        ),
    ],
    RUNNER: [
        (
            "        global_input_batch = input_batch\n"
            "        if self.batch_sharder is not None:\n",
            "        global_input_batch = input_batch\n"
            "        if self.batch_sharder is not None:\n"
            "            draft_shard_cache = getattr(self.speculator, \"_draft_shard_cache\", None)\n"
            "            if draft_shard_cache is not None and input_batch.num_draft_tokens:\n"
            "                draft_shard_cache.materialize(\n"
            "                    input_batch.idx_mapping_np, input_batch.cu_num_logits_np,\n"
            "                    self.sampler.sampling_states.temperature.np,\n"
            "                )\n",
        ),
    ],
}

KERNEL_SOURCE = r'''"""Inkling2 TP2 Markov sampling and verifier handoff."""
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.batch_shard import (
    all_to_all_vocab_shards,
    shard_owners,
)
from vllm.v1.worker.gpu.sample.greedy_argmax import _argmax_partial
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

GUMBEL_BLOCK = 1024
GREEDY_BLOCK = 4096


@dataclass(frozen=True)
class DraftShardLayout:
    width: int
    vocab: int
    rank: int

    @property
    def supported(self):
        # Dense stock concatenates physical shards then trims the right tail.
        # Require at least one unshared Gumbel block on each rank.
        return (
            self.width >= GUMBEL_BLOCK
            and self.width + (-self.width % GUMBEL_BLOCK) < self.vocab <= 2 * self.width
            and self.rank in (0, 1)
        )

    @property
    def offset(self):
        return self.width * self.rank

    @property
    def valid(self):
        return min(self.width, self.vocab - self.offset)

    @property
    def cut(self):
        return self.width % GUMBEL_BLOCK

    @property
    def boundary(self):
        return self.width - self.cut

    def gumbel_slice(self):
        if self.cut == 0:
            return slice(0, self.valid)
        if self.rank == 0:
            return slice(0, self.boundary)
        return slice(GUMBEL_BLOCK - self.cut, self.valid)

    def fragment_slice(self):
        if self.cut == 0:
            return slice(0, 0)
        if self.rank == 0:
            return slice(self.boundary, self.width)
        return slice(0, GUMBEL_BLOCK - self.cut)


def verifier_plan(slots, cu_num_logits, temperature, rank):
    """Retain r7's owner and target-logit ordering, including under slot churn."""
    selected = (np.diff(cu_num_logits) > 1) & (temperature[slots] != 0.0)
    owners = shard_owners(len(slots), 2)
    by_owner = [slots[selected & (owners == peer)] for peer in range(2)]
    return np.concatenate(by_owner), [len(rows) for rows in by_owner], by_owner[rank]


class DraftShardCache:
    def __init__(self, dense_cache, layout):
        self.dense = dense_cache
        self.layout = layout
        self.steps = dense_cache.values.shape[1]
        self.local = dense_cache.values.new_zeros(
            (dense_cache.values.shape[0], self.steps, layout.width)
        )

    def stage(self, local_logits, slots, temperature, step):
        rows, width = local_logits.shape
        # Bitwise copy, including BF16 NaN payloads. The launch is capture-safe.
        _stage_logits[(rows, triton.cdiv(width, GUMBEL_BLOCK))](
            local_logits.view(torch.int16), slots, temperature,
            self.local.view(torch.int16), width, self.steps, step,
            GUMBEL_BLOCK,
        )

    def materialize(self, slots, cu_num_logits, temperature):
        packed_slots, counts, owned_slots = verifier_plan(
            slots, cu_num_logits, temperature, self.layout.rank
        )
        if not len(packed_slots):
            return
        device = self.local.device
        packed_slots = torch.as_tensor(packed_slots, dtype=torch.long, device=device)
        send = self.local.index_select(0, packed_slots).view(-1, self.layout.width)
        dense = all_to_all_vocab_shards(
            send, [count * self.steps for count in counts], len(owned_slots) * self.steps
        )
        if len(owned_slots):
            owned_slots = torch.as_tensor(owned_slots, dtype=torch.long, device=device)
            dense = dense.view(len(owned_slots), self.steps, 2 * self.layout.width)
            self.dense.values.index_copy_(0, owned_slots, dense[..., :self.layout.vocab])


@triton.jit
def _stage_logits(
    LOGITS, SLOTS, TEMP, OUT, WIDTH: tl.constexpr, STEPS: tl.constexpr,
    STEP: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(SLOTS + row).to(tl.int64)
    temp = tl.load(TEMP + slot, slot >= 0, 0.0)
    if temp != 0.0:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        data = tl.load(LOGITS + row * WIDTH + col, col < WIDTH, 0)
        tl.store(OUT + (slot * STEPS + STEP) * WIDTH + col, data, col < WIDTH)


@triton.jit
def _better(value, token, best, best_token):
    nan = value != value
    best_nan = best != best
    # Mirrors stock Gumbel's first-NaN block policy; greedy has no NaN scores.
    return (nan & ~best_nan) | (
        (nan == best_nan)
        & ((value > best) | (((value == best) | nan) & (token < best_token)))
    )


@triton.jit
def _pack_winner(
    SCORE, IDS, LOGITS_BITS, PACKET, PACKET_BITS,
    PARTS: tl.constexpr, PART_BLOCK: tl.constexpr, ID_OFFSET: tl.constexpr,
    TILE_OFFSET: tl.constexpr, WIDTH: tl.constexpr, WORDS: tl.constexpr,
    FRAGMENT_START: tl.constexpr, FRAGMENT_COUNT: tl.constexpr,
    FRAGMENT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.arange(0, PART_BLOCK)
    scores = tl.load(SCORE + row * PARTS + part, part < PARTS, float("-inf"))
    is_nan = scores != scores
    first_nan = tl.min(tl.where(is_nan, part, PART_BLOCK), 0)
    if first_nan < PART_BLOCK:
        winner = first_nan
    else:
        highest = tl.max(scores, 0)
        winner = tl.min(tl.where((part < PARTS) & (scores == highest), part, PART_BLOCK), 0)
    score = tl.load(SCORE + row * PARTS + winner).to(tl.float64)
    token = tl.load(IDS + row * PARTS + winner).to(tl.int64) + ID_OFFSET
    if TILE_OFFSET is not None:
        # The dense Gumbel combine chooses a tile before reading its token ID.
        token = ((winner.to(tl.int64) + TILE_OFFSET) << 32) | (token & 0xFFFFFFFF)
    tl.store(PACKET + row * WORDS, score.to(tl.int64, bitcast=True))
    tl.store(PACKET + row * WORDS + 1, token)

    if FRAGMENT_BLOCK > 0:
        i = tl.arange(0, FRAGMENT_BLOCK)
        bits = tl.load(
            LOGITS_BITS + row * WIDTH + FRAGMENT_START + i, i < FRAGMENT_COUNT, 0
        )
        tl.store(PACKET_BITS + (row * WORDS + 2) * 4 + i, bits, i < (WORDS - 2) * 4)


@triton.jit
def _stitch_boundary(
    PACKET_BITS, OUT, ROWS: tl.constexpr, WORDS: tl.constexpr,
    CUT: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    from_left = lane < CUT
    left = (row * WORDS + 2) * 4
    right = ((ROWS + row) * WORDS + 2) * 4
    a = tl.load(PACKET_BITS + left + lane, from_left, 0)
    b = tl.load(PACKET_BITS + right + lane - CUT, ~from_left, 0)
    tl.store(OUT + row * BLOCK + lane, tl.where(from_left, a, b))


@triton.jit
def _finish_winner(
    PACKET, BOUNDARY_SCORES, BOUNDARY_IDS, OUT,
    ROWS: tl.constexpr, WORDS: tl.constexpr, BOUNDARY_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    left = row * WORDS
    right = (ROWS + row) * WORDS
    val0 = tl.load(PACKET + left).to(tl.float64, bitcast=True)
    id0 = tl.load(PACKET + left + 1)
    val1 = tl.load(PACKET + right).to(tl.float64, bitcast=True)
    id1 = tl.load(PACKET + right + 1)
    use_right = _better(val1, id1, val0, id0)
    best = tl.where(use_right, val1, val0)
    token = tl.where(use_right, id1, id0)
    if BOUNDARY_SCORES is not None:
        score = tl.load(BOUNDARY_SCORES + row).to(tl.float64)
        boundary_id = tl.load(BOUNDARY_IDS + row)
        key = (tl.cast(BOUNDARY_TILE, tl.int64) << 32) | (boundary_id & 0xFFFFFFFF)
        token = tl.where(_better(score, key, best, token), key, token)
    tl.store(OUT + row, token.to(tl.int32).to(tl.int64))


def _gumbel_blocks(logits, slots, temperature, seeds, pos_before, step, use_fp64, offset):
    rows, cols = logits.shape
    parts = triton.cdiv(cols, GUMBEL_BLOCK)
    dtype = torch.float64 if use_fp64 else torch.float32
    scores = logits.new_empty((rows, parts), dtype=dtype)
    ids = logits.new_empty((rows, parts), dtype=torch.int64)
    _gumbel_sample_kernel[(rows, parts)](
        ids, parts, scores, parts, None, 0, 0, None, None, 0, None,
        logits, logits.stride(0), None, 0, slots, seeds, pos_before,
        temperature, cols,
        BLOCK_SIZE=GUMBEL_BLOCK, IS_DRAFTING=True, APPLY_TEMPERATURE=True,
        USE_FP64=use_fp64, PER_TOKEN_COL=False, HAS_BIAS=False,
        VOCAB_OFFSET=offset, DRAFT_STEP=step,
    )
    return scores, ids


def sample_shard(layout, logits, slots, temperature, seeds, pos_before, step,
                 use_fp64, cache):
    """One winner packet per rank and row, in eager and captured Markov loops."""
    rows = logits.shape[0]
    if not rows:
        return logits.new_empty((0,), dtype=torch.int64)
    probabilistic = cache is not None
    slots = slots.contiguous()
    pos_before = pos_before.contiguous()
    if probabilistic:
        block = GUMBEL_BLOCK
        span = layout.gumbel_slice()
        scores, ids = _gumbel_blocks(
            logits[:, span], slots, temperature, seeds, pos_before, step, use_fp64,
            layout.offset + span.start,
        )
        parts = scores.shape[1]
        cache.stage(logits, slots, temperature, step)
        cut = layout.cut
        fragment = layout.fragment_slice()
        payload = max(cut, block - cut) if cut else 0
        id_offset = 0
        tile_offset = (layout.offset + span.start) // block
    else:
        cut = payload = 0
        fragment = slice(0, 0)
        parts = triton.cdiv(layout.valid, GREEDY_BLOCK)
        scores = logits.new_empty((rows, parts), dtype=torch.float32)
        ids = logits.new_empty((rows, parts), dtype=torch.int32)
        _argmax_partial[(rows, parts)](
            logits, scores, ids, layout.valid, logits.stride(0), parts, GREEDY_BLOCK,
            num_warps=4,
        )
        id_offset = layout.offset
        tile_offset = None

    words = 2 + triton.cdiv(payload, 4)
    packet = logits.new_empty((rows, words), dtype=torch.int64)
    _pack_winner[(rows,)](
        scores, ids, logits.view(torch.int16), packet, packet.view(torch.int16),
        parts, triton.next_power_of_2(parts), id_offset, tile_offset, layout.width, words,
        fragment.start, fragment.stop - fragment.start,
        triton.next_power_of_2((words - 2) * 4) if payload else 0,
        num_warps=4 if payload else 1,
    )
    gathered = tensor_model_parallel_all_gather(packet, dim=0)
    boundary_scores = boundary_ids = None
    if cut:
        # Give the stock tile kernel all the original lanes in their dense order.
        boundary = logits.new_empty((rows, GUMBEL_BLOCK))
        _stitch_boundary[(rows,)](
            gathered.view(torch.int16), boundary.view(torch.int16), rows, words,
            cut, GUMBEL_BLOCK,
        )
        boundary_scores, boundary_ids = _gumbel_blocks(
            boundary, slots, temperature, seeds, pos_before, step, use_fp64,
            layout.boundary,
        )
    out = logits.new_empty((rows,), dtype=torch.int64)
    _finish_winner[(rows,)](
        gathered, boundary_scores, boundary_ids, out, rows, words,
        layout.boundary // GUMBEL_BLOCK, num_warps=1,
    )
    return out
'''


def apply(root):
    generated = root / KERNELS
    if generated.exists():
        raise ValueError(f"{generated}: already exists")
    changes = {}
    shard_gumbel = None
    gumbel_notices = []
    for name, expected in EXPECT.items():
        path = root / name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} != r8 input {expected}")
        source = data.decode()
        for old, new in EDITS[name]:
            count = source.count(old)
            if count != 1:
                raise ValueError(f"{path}: anchor matched {count} times: {old[:90]!r}")
            source = source.replace(old, new, 1)
        if name == GUMBEL:
            # Derive a private variant of the stock tiles and keep the installed
            # dense sampler byte-for-byte identical to r8.
            start = "@triton.jit\ndef gumbel_block_argmax("
            end = "@triton.jit\ndef _gumbel_combine_kernel("
            if source.count(start) != 1 or source.count(end) != 1:
                raise ValueError(f"{path}: expected one source block for the sharded sampler")
            lo, hi = source.index(start), source.index(end)
            if lo >= hi:
                raise ValueError(f"{path}: Gumbel kernel anchors are out of order")
            shard_gumbel = source[lo:hi]
            gumbel_notices = list(takewhile(lambda line: line.startswith("#") or not line.strip(),
                                            source.splitlines()))
        else:
            changes[path] = source
    if shard_gumbel is None:
        raise ValueError("missing guarded Gumbel source")
    if not any(line.startswith("# SPDX-License-Identifier:") for line in gumbel_notices):
        gumbel_notices.insert(0, "# SPDX-License-Identifier: Apache-2.0")
    gumbel_notices.append("# Derived from vLLM's Gumbel sampler and modified for Inkling DSpark vocab sharding.")
    changes[generated] = "\n".join(gumbel_notices) + "\n" + KERNEL_SOURCE + "\n\n" + shard_gumbel.rstrip() + "\n"
    for path, source in changes.items():
        ast.parse(source, filename=str(path))
    for path, source in changes.items():
        path.write_bytes(source.encode())
        print(f"draftshard: patched {path}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        apply(Path(sys.argv[1]))
    except (ValueError, OSError, SyntaxError) as exc:
        sys.exit(f"draftshard: {exc}")


if __name__ == "__main__":
    main()
