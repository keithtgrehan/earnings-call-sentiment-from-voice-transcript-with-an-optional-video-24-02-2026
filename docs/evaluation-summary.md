# Evaluation Summary

## What Is Evaluated Today
- Functional pipeline behavior through unit/integration-style checks in `tests/`.
- Artifact completeness/health using `scripts/verify_outputs.py`.
- Deterministic signal generation for:
  - guidance extraction
  - guidance revision labeling
  - tone-change detection
- Reproducible metadata capture in `run_meta.json` and summary reporting in `metrics.json` / `report.md`.
- Multimodal breadth reporting via manifests and sidecar artifact presence using `scripts/build_multimodal_eval_summary.py`.
- Manual source-pair and manifest validation for multimodal collection planning.

## What Is Optional / Exploratory
- Question-shift analysis (`--question-shifts`) is available but should be treated as heuristic analysis.
- LLM narrative mode in `scripts/run_eval.py` is optional; deterministic mode is preferred for validation.
- Backtest tooling is available for study, but quality depends on run metadata and local market data inputs.
- WhisperX alignment, optional pyannote diarization, OpenFace summaries, and NLP-assist sidecars are exploratory support layers rather than core evaluation targets.
- External dataset scaffolding for MAEC, MELD, and RAVDESS is preparatory only and does not constitute finance-specific evaluation.

## What Remains Unproven
- Predictive performance consistency across symbols/time periods.
- Statistical significance for any return-linked signal claim.
- Practical trading edge after realistic costs and execution assumptions.
- Any review-quality lift from the new multimodal sidecars.
- Any proven visual model quality or predictive value from the current repo state.
- Any finance-specific performance claim from external dataset scaffolding.

## Honest Interpretation for Capstone Review
The project currently demonstrates a credible **signal extraction and evidence organization workflow**. The multimodal additions currently extend breadth, instrumentation, and metadata tracking. They do not yet demonstrate a proven forecasting, trading-performance, or multimodal-performance advantage.

## Next Evaluation Steps (Safe and Reasonable)
- Run repeated out-of-sample backtests across multiple events/symbols.
- Pre-register simple acceptance criteria before comparing variants.
- Treat any positive findings as preliminary until replicated.
