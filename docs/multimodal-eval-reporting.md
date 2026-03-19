# Multimodal Evaluation Reporting

This reporting layer is conservative by design.

Core framing:

- transcript artifacts remain primary
- audio alignment, visual features, and NLP sidecars are supporting evidence
- current multimodal expansion is about breadth, coverage, and instrumentation
- this reporting layer does not claim model quality or predictive improvement

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
