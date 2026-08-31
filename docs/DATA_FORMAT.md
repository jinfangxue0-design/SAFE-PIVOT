# Data format

## Judgment JSONL

One JSON object per line:

```json
{
  "id": "model-a::question-001",
  "question_id": "question-001",
  "question": "...",
  "grounding": "...",
  "flat_answer": "...",
  "generic_answer": "...",
  "flat_correct": 1,
  "generic_correct": 0
}
```

`grounding` may be replaced by `evidence` or `context`. `question_id` defaults to
`original_id`, then to the suffix after `::` in `id`, then to `id` itself.

## Verifier JSONL

```json
{
  "id": "model-a::question-001",
  "decision": "A",
  "support_flat": 0.85,
  "support_evidence": 0.30,
  "answer_conflict": 1,
  "evidence_reliability": 0.20,
  "answer_equivalent": 0,
  "delta_hat": -0.55,
  "harm_hat": 0.44,
  "rationale": "..."
}
```

`delta_hat` and `harm_hat` are recomputed from canonicalized primitive fields by
the verifier runner. Rows with an `error` field should be rerun before analysis.

## Labels and leakage

`flat_correct` and `generic_correct` are used only by local evaluation and
calibration. They are not included in the verifier prompt.

