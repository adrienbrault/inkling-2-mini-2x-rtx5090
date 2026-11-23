# Spec-Bench subsets

The four `specbench-subset-c{1,4,8,16}.jsonl` files contain modified 120-prompt subsets of [Spec-Bench](https://github.com/hemingkx/Spec-Bench), from `data/spec_bench/question.jsonl`. The [Apache-2.0 text](../../LICENSE-APACHE) covers Spec-Bench. The `specbench-manifest.json` file lists the selected IDs, prompt hashes and selection parameters. Prompts were selected by taking the smallest seeded ID hash in each character-length quartet within the six categories. The subsets add benchmark labels and use seeded per-cell ordering. The separate c16 label and order are produced by [the subset script](../specbench_c16_subset.py).

The underlying datasets retain their own terms:

| Spec-Bench workload | Underlying material |
| --- | --- |
| Multi-turn conversation | MT-Bench |
| Translation | WMT German–English |
| Summarization | CNN/Daily Mail |
| Question answering | Natural Questions |
| Math reasoning | GSM8K |
| Retrieval-augmented generation | Natural Questions-style queries with Wikipedia passages |

**TO CONFIRM** the exact Spec-Bench revision, the WMT edition, the RAG dataset and passage provenance, all underlying revisions and their redistribution terms, and any upstream notices before publishing the prompts or detailed results. See the [third-party credits](../../THIRD_PARTY.md).
