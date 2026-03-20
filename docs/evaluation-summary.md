# Evaluation Summary

## What Is Evaluated Today
- Functional pipeline behavior through unit/integration-style checks in `tests/`.
- Artifact completeness/health using `scripts/verify_outputs.py`.
- Deterministic signal generation for:
  - guidance extraction
  - guidance revision labeling
  - tone-change detection
- Reproducible metadata capture in `run_meta.json` and summary reporting in `metrics.json` / `report.md`.
- Committed multimodal coverage reporting from `data/processed/multimodal/eval/multimodal_eval_summary.json`.
- Canonical review truth still comes from transcript-first deterministic outputs such as `transcript.json`, `transcript.txt`, `guidance*.csv`, `tone_changes.csv`, and `report.md`.
- Supporting sidecar run visibility comes from `data/processed/multimodal/eval/curated_slice_run_status.json` and `data/processed/multimodal/visual/curated_clip_run_status.json`.
- Committed source-level visual summaries live under `data/processed/multimodal/visual/<source_id>/segment_visual_features.{csv,json}` and `visual_coverage_summary.json`.

Current committed multimodal summary counts:
- `sources_with_visual = 4`
- `sources_with_nlp = 1`
- `sources_with_any_multimodal_sidecar = 5`
- `sources_with_alignment = 0`
- `visually_usable_segments = 1`
- `audio_aligned_segments = 0`
- `segments_with_nlp_support = 561`
- Current visual slice interpretation: `1` usable clip window, `3` completed but unusable windows due to face-too-small/intermittent framing.

## What Is Optional / Exploratory
- Question-shift analysis (`--question-shifts`) is available but should be treated as heuristic analysis.
- LLM narrative mode in `scripts/run_eval.py` is optional; deterministic mode is preferred for validation.
- Backtest tooling is available for study, but quality depends on run metadata and local market data inputs.
- Visual sidecars are currently most practical through short clip-based runtime checks rather than full-webcast OpenFace passes on local hardware.
- Video-backed outputs are real in the current repo state, but they remain supporting evidence only.

## What Remains Unproven
- Predictive performance consistency across symbols/time periods.
- Statistical significance for any return-linked signal claim.
- Practical trading edge after realistic costs and execution assumptions.
- Any benchmark or decision-quality lift from the current multimodal sidecars.
- Any claim that the current clip-based visual workflow validates full-call visual behavior at scale.
- Any claim that alignment is materially exercised in the current committed repo state.

## Honest Interpretation for Capstone Review
The project currently demonstrates a credible **signal extraction and evidence organization workflow** plus partial multimodal instrumentation. The committed multimodal outputs show real exercised visual coverage on a narrow curated slice and limited NLP support, but they remain supporting layers around a transcript-first deterministic review path rather than a replacement truth source or a proven forecasting, trading-performance, or validated multimodal-performance advantage.

## Next Evaluation Steps (Safe and Reasonable)
- Run repeated out-of-sample backtests across multiple events/symbols.
- Pre-register simple acceptance criteria before comparing variants.
- Treat any positive findings as preliminary until replicated.
