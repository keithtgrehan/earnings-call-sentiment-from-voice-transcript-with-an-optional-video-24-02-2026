# Evaluation Summary

## What Is Evaluated Today
- Functional pipeline behavior through unit/integration-style checks in `tests/`.
- Artifact completeness/health using `scripts/verify_outputs.py`.
- Deterministic signal generation for:
  - guidance extraction
  - guidance revision labeling
  - tone-change detection
- Reproducible metadata capture in `run_meta.json` and summary reporting in `metrics.json` / `report.md`.

## What Is Optional / Exploratory
- Question-shift analysis (`--question-shifts`) is available but should be treated as heuristic analysis.
- LLM narrative mode in `scripts/run_eval.py` is optional; deterministic mode is preferred for validation.
- Backtest tooling is available for study, but quality depends on run metadata and local market data inputs.

## What Remains Unproven
- Predictive performance consistency across symbols/time periods.
- Statistical significance for any return-linked signal claim.
- Practical trading edge after realistic costs and execution assumptions.

## Honest Interpretation for Capstone Review
The project currently demonstrates a credible **signal extraction and evidence organization workflow**. It does not yet demonstrate a proven forecasting or trading-performance advantage.

## Next Evaluation Steps (Safe and Reasonable)
- Run repeated out-of-sample backtests across multiple events/symbols.
- Pre-register simple acceptance criteria before comparing variants.
- Treat any positive findings as preliminary until replicated.
