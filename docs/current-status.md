# Current Status

## Implemented Now
- Transcript/audio-first pipeline from call input to structured artifacts.
- Deterministic post-processing for guidance extraction, guidance revision comparison, and tone-change detection.
- Standard artifact outputs (`transcript.*`, `sentiment_segments.csv`, `chunks_scored.jsonl`, `guidance*.csv`, `tone_changes.csv`, `metrics.json`, `report.md`, `run_meta.json`).
- CLI stage controls (`--download-only`, `--transcribe-only`, `--score-only`) and output checks (`--strict`, `scripts/verify_outputs.py`).

## Optional / Experimental
- Question-shift analysis (`--question-shifts`) is optional and heuristic-driven.
- LLM narrative layer in `scripts/run_eval.py` is optional; deterministic mode (`--llm none`) is the safest baseline.
- Backtest harness exists (`scripts/backtest_signals.py`) but depends on local run metadata quality and external price history inputs.

## Unproven
- No proven predictive edge from current repository artifacts.
- No statistical-significance claim should be made without dedicated out-of-sample evaluation.
- No claim of autonomous trading capability.

## Practical Positioning
This project should be presented as a **signal-extraction prototype** for faster, more structured earnings-call review. The strongest current value is deterministic, evidence-linked output generation rather than return prediction.
