"""Eval harness: product-level extraction-quality benchmarks.

Layout:
- runner.py           (Phase 4): orchestrator -- iterates a dataset, runs the
                                 Extractor, scores outputs, emits a run report
- graders/            (Phase 4): pluggable scorers (code, rule, model)
- datasets/           (Phase 0): benchmark datasets, one directory per benchmark

Each dataset directory contains:
- schema.yaml      ehrextract schema definition
- prompt.txt       instructional prompt
- notes.jsonl      input notes ({note_id, note_text} per line)
- expected.jsonl   expected outputs ({note_id, <schema fields>} per line)

See `.claude/evals/README.md` for methodology and `.claude/evals/ehrextract-product-extraction.md`
for the active product-eval definition.
"""
