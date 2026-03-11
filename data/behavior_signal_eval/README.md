# Behavior Signal Eval Set

This package is a small sentence-level gold set for Phase 1 behavioral signal evaluation.

It is separate from the guidance-change benchmark packages and does not modify any canonical guidance labels.

Files:
- `uncertainty_labels.csv`: sentence spans labeled `absent`, `present`, or `strong`
- `reassurance_labels.csv`: sentence spans labeled `absent` or `present`
- `skepticism_labels.csv`: analyst question spans labeled `low`, `medium`, or `high`
- `source_manifest.csv`: unique source files used by the eval set

Conventions:
- Labels are tied to exact local text spans with 0-based `[start_char, end_char)` offsets.
- `source_path` always points to an existing local transcript or excerpt file.
- Excerpt files remain clearly separate from full transcripts via `source_kind` in `source_manifest.csv`.
- This is an auditable mini eval set for reviewer-facing behavior signals, not a production benchmark.
