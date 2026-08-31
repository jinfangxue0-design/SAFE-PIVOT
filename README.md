# SAFE-PIVOT

[中文说明](README_zh-CN.md)

SAFE-PIVOT is a post-hoc framework for deciding whether to keep an answer
generated without external evidence (`Y(0)`) or switch to an evidence-grounded
answer (`Y(1)`). This release contains the paper's core method only:

1. a structured verifier that scores both candidate answers and evidence reliability;
2. a conservative tie-to-flat arbitration rule;
3. Scalar and Dual-R threshold families;
4. grouped CGC/LTT calibration with a pre-registered fixed sequence;
5. net-gain conservation: deploy always-flat when the best certified calibration gain is not positive.

This repository contains the core SAFE-PIVOT implementation and synthetic
examples. Benchmark data, generated model outputs, experimental artifacts, and
manuscript assets are not included.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Input contract

Each judgment row must contain:

- `id`, `question`, `grounding` (or `evidence`), `flat_answer`, `generic_answer`;
- `flat_correct` and `generic_correct` for offline evaluation/calibration;
- `question_id` when multiple model records share the same underlying question.

The labels are never sent to the verifier. Full schemas are documented in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

## 1. Run the structured verifier

```powershell
$env:SAFE_PIVOT_BASE_URL="https://api.openai.com/v1"
$env:SAFE_PIVOT_API_KEY="..."
$env:SAFE_PIVOT_MODEL="..."

safe-pivot-verifier `
  --input examples/sample_judgments.jsonl `
  --output outputs/sample_verifier.jsonl `
  --resume
```

The endpoint only needs to implement an OpenAI-compatible chat-completions API.
No credentials are stored in files or printed by the runner.

## 2. Reproduce the basic tie-to-flat gate

The basic gate accepts Answer B only when the verifier chooses B, the estimated
benefit clears `tau_delta`, estimated harm stays below `tau_harm`, and the two
answers are not equivalent.

```powershell
safe-pivot-basic `
  --judgments examples/sample_judgments.jsonl `
  --verifier examples/sample_verifier_outputs.jsonl `
  --tau-delta 0.10 `
  --tau-harm 0.25 `
  --out-detail outputs/basic_detail.jsonl `
  --out-summary outputs/basic_summary.json
```

## 3. Run grouped CGC calibration

```powershell
safe-pivot-cgc `
  --judgments examples/sample_judgments.jsonl `
  --verifier examples/sample_verifier_outputs.jsonl `
  --alphas 0.10 0.20 `
  --delta 0.10 `
  --n-seeds 20 `
  --calibration-fraction 0.50 `
  --methods scalar dual_r `
  --out-seed-csv outputs/cgc_seed_rows.csv `
  --out-summary-csv outputs/cgc_summary.csv
```

For the paper protocol, use `--n-seeds 200`. Records with the same
`question_id` remain on the same side of every calibration/held-out split.

## Definitions that must not be mixed

- `certified_set_nonempty`: at least one candidate passed the risk test.
- `fallback_selected`: the final deployment is always-flat, including the case
  where candidates were certified but none had positive calibration net gain.
- `heldout_fallback_rate`: fraction of held-out samples for which the selected
  gate keeps the flat answer. This is not the run-level fallback rate.
- certificate violation: the already selected configuration has full-target
  harmful-switch risk above `alpha`. Held-out risk is reported separately.

## Test

```bash
python -m unittest discover -s tests -v
```

## Data and model outputs

This repository ships only synthetic examples. Download benchmark data from its
official source and verify its license before redistribution. Generated answers
and verifier outputs may also be subject to provider terms and should be released
separately from the framework code.
