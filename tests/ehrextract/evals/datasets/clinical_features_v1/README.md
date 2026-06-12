# clinical_features_v1

Five hand-written synthetic clinical-style notes, each annotated with the
expected output of a 4-field schema. Designed to exercise integer, string,
boolean, and enum field types with realistic-but-non-PHI content.

## Files

- `schema.yaml` — 4-field schema (age, primary_complaint, has_diabetes, smoking_status)
- `prompt.txt` — instructional prompt
- `notes.jsonl` — 5 input notes
- `expected.jsonl` — 5 expected outputs

## Notes generation

Notes are deliberately:

- Short (1–2 sentences) so eval cost stays low
- Free of PHI (no real identifiers, no real institutions)
- Free of internal taxonomy markers (compound terms in the leakage banlist
  are not present)
- Varied across age groups, presenting complaints, smoking history

## Usage (from Phase 4 onward)

```python
from ehrextract.evals import runner

results = runner.run(
    dataset_dir="tests/ehrextract/evals/datasets/clinical_features_v1",
    provider="openai",
    model="gpt-4o-mini",
)
print(results.summary)  # parse_rate, exact_match, per-field accuracy
```

## When to extend

Add a sixth note when a real-world failure mode is identified that the
existing five don't cover. Don't grow the dataset for its own sake — five
focused examples beat fifty redundant ones for a fast iteration loop.

## When to fork (clinical_features_v2)

Fork to a v2 directory when the schema or prompt changes in a
breaking way. v1 stays frozen for historical comparison.
