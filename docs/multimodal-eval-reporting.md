# Multimodal Evaluation Reporting

This reporting layer is conservative by design.

Core framing:

- transcript artifacts remain primary
- audio alignment, visual features, and NLP sidecars are supporting evidence
- current multimodal expansion is about breadth, coverage, and instrumentation
- this reporting layer does not claim model quality or predictive improvement

## Current Repo Status

The generated summary currently reports:

- `12` source calls
- `7` source/layout groups
- `25` manifest segments
- `0` visually usable segments in the current generated report
- `0` audio-aligned segments in the current generated report
- `0` segments with NLP support in the current generated report

That should be read as current coverage status in this checkout, not as evidence
that the multimodal layers have been disproven or that the transcript baseline
has changed.

## What The Summary Reports

The multimodal evaluation summary focuses on:

- how many source calls are tracked
- how many source/layout groups are represented
- how many manifest segments exist
- where alignment, visual, and NLP sidecars exist
- how many segments look visually usable
- how many segments have alignment or NLP support
- failure or absence counts for supporting sidecars
- which companies and layouts currently have coverage

It is intentionally not a benchmark-quality statement about whether the
multimodal layers improve outcomes.

## What This Does Not Prove

- no proven predictive improvement
- no proof that alignment, visual, or NLP sidecars improve benchmark results
- no claim of visual model training success
- no proof that external datasets establish finance-specific performance
- no replacement of transcript-first deterministic artifacts

## Output Artifacts

Run:

```bash
PYTHONPATH=src python scripts/build_multimodal_eval_summary.py
```

This writes:

- `data/processed/multimodal/eval/multimodal_eval_summary.json`
- `data/processed/multimodal/eval/multimodal_source_coverage.csv`

## Interpretation Notes

- missing sidecar artifacts mean supporting evidence is absent in the current
  checkout
- visually unusable counts should be read as coverage or extraction limits, not
  as negative findings about a call
- NLP-support counts mean sidecar model outputs exist for inspection; they do
  not override deterministic transcript outputs
- alignment counts reflect sidecar timing coverage only; they do not replace the
  default transcript generation path
