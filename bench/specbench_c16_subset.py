#!/usr/bin/env python3
"""Build the Spec-Bench c16 cell (inputs/specbench-subset-c16.jsonl) from the c1/c4/c8 subsets.

Same 120 prompts as c1/c4/c8, reshuffled with seed 7310 + 16 the way the other cells are ordered, each prompt labelled "[headline-r1 subset-c16; question N; seed 7326]"
so no cell shares a cached first page with another. Copies the c1/c4/c8 subset files next to it.   usage: specbench_c16_subset.py SRC_DATA_DIR DEST_DIR"""
import json
import random
import sys
from pathlib import Path

src, dest = Path(sys.argv[1]), Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
cells = {c: [json.loads(l) for l in (src / f"specbench-subset-c{c}.jsonl").read_text().splitlines()] for c in (1, 4, 8)}
ids = {c: sorted(r["question_id"] for r in rows) for c, rows in cells.items()}
assert ids[1] == ids[4] == ids[8] and len(ids[1]) == 120, "c1/c4/c8 subsets differ"


def strip(row, c):
    head = f"[headline-r1 subset-c{c}; question {row['question_id']}; seed {7310 + c}]\n\n"
    assert row["turns"][0].startswith(head), (c, row["question_id"])
    return dict(row, turns=[row["turns"][0][len(head):]] + list(row["turns"][1:]))


# headline-r1 shuffles the selection sorted by (category, str(question_id)), then labels each prompt.
base = sorted((strip(r, 1) for r in cells[1]), key=lambda r: (r["category"] if r["category"] not in
              {"writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"}
              else "multi_turn", str(r["question_id"])))
seed = 7310 + 16
random.Random(seed).shuffle(base)
out = []
for row in base:
    row = dict(row, turns=list(row["turns"]))
    row["turns"][0] = f"[headline-r1 subset-c16; question {row['question_id']}; seed {seed}]\n\n" + row["turns"][0]
    out.append(row)
(dest / "specbench-subset-c16.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for c in (1, 4, 8):
    (dest / f"specbench-subset-c{c}.jsonl").write_bytes((src / f"specbench-subset-c{c}.jsonl").read_bytes())
print(f"{dest}: c16 {len(out)} prompts, first ids {[r['question_id'] for r in out[:5]]}")
