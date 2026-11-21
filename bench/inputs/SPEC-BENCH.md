# Spec-Bench subset

`specbench-subset-c{1,4,8}.jsonl` are 120-prompt subsets of Spec-Bench (github.com/hemingkx/Spec-Bench, Apache-2.0,
`data/spec_bench/question.jsonl`): within each of its six categories, prompts are grouped into character-length quartets
and the member with the smallest seeded ID hash is taken, so each subset is stratified by category and length.
`specbench-manifest.json` lists the selected IDs, prompt hashes and the selection parameters.
