# Current Status

## Reviewer Quick Links
- [Evidence map](evidence-map.md)
- [End-to-end demo path](demo-path.md)
- [Evaluation summary](evaluation-summary.md)

## Implemented Now
- Transcript-first pipeline from call input to structured artifacts, with audio/video remaining optional supporting layers.
- Deterministic post-processing for guidance extraction, guidance revision comparison, and tone-change detection.
- Standard artifact outputs (`transcript.*`, `sentiment_segments.csv`, `chunks_scored.jsonl`, `guidance*.csv`, `tone_changes.csv`, `metrics.json`, `report.md`, `run_meta.json`).
- CLI stage controls (`--download-only`, `--transcribe-only`, `--score-only`) and output checks (`--strict`, `scripts/verify_outputs.py`).
- Committed multimodal summary artifacts under `data/processed/multimodal/` for curated visual coverage and merged eval reporting.

## Reviewer Status Surfaces
- Canonical transcript-first review truth still lives in the deterministic run outputs: `transcript.json`, `transcript.txt`, `guidance*.csv`, `tone_changes.csv`, `metrics.json`, and `report.md`.
- Sidecar execution status for the curated slice lives in `data/processed/multimodal/eval/curated_slice_run_status.json`.
- Clip-based visual runtime status lives in `data/processed/multimodal/visual/curated_clip_run_status.json`.
- Committed source-level visual summaries live under `data/processed/multimodal/visual/<source_id>/segment_visual_features.{csv,json}` and `visual_coverage_summary.json`.
- Rolled-up multimodal breadth lives in `data/processed/multimodal/eval/multimodal_eval_summary.json`.
- Current `main` keeps summary-level multimodal status plus source-level visual summaries. Supporting sidecars should not be treated as canonical review truth.

## Optional / Experimental
- Question-shift analysis (`--question-shifts`) is optional and heuristic-driven.
- LLM narrative layer in `scripts/run_eval.py` is optional; deterministic mode (`--llm none`) is the safest baseline.
- Backtest harness exists (`scripts/backtest_signals.py`) but depends on local run metadata quality and external price history inputs.
- Multimodal sidecars remain supporting and partial. The current exercised path is a narrow curated slice rather than broad production-scale coverage.

## Current Multimodal Breadth
Current committed multimodal summary outputs report:
- `4` sources with visual sidecar artifacts
- `1` source with NLP sidecar artifacts
- `5` sources with any multimodal sidecar artifacts
- `0` sources with alignment artifacts
- `1` visually usable segment
- `0` audio-aligned segments
- `561` segments with NLP support

Practical status:
- visual support has been exercised on a real but small curated clip-based slice
- full-webcast OpenFace was not practical on local hardware, so short runtime-check clips are the current workable path
- video remains secondary supporting evidence, not the review source of truth
- of the `4` committed clip windows, `1` is visually usable and `3` are explicitly recorded as unusable because the face is too small or intermittent
- committed NLP support is visible through the rolled-up eval summary for one source rather than broad source-level NLP artifacts on `main`
- alignment exists as a target area, but the current committed outputs still show zero aligned sources
- The repo also now includes `4` committed Codex prototype artifact-review rows in `data/media_support_eval/multimodal_review_results_codex_proto.csv` plus a saved descriptive summary in `outputs/media_support_eval/multimodal_review_summary.json`.
- Those rows are prototype artifact reviews rather than human-subject study results, so they do not establish multimodal lift or statistical significance.

## Unproven
- No proven predictive edge from current repository artifacts.
- No statistical-significance claim should be made without dedicated out-of-sample evaluation.
- No claim of autonomous trading capability.
- No claim that current multimodal sidecars improve benchmark quality or decision quality.
- No claim that clip-based visual summaries prove full-call visual validity.
- No claim that the current repo state demonstrates alignment coverage at useful scale.

## Practical Positioning
This project should be presented as a **transcript-first validated prototype** and deterministic evidence-backed review tool. The strongest current value is deterministic, evidence-linked output generation plus strong internal agreement checkpoints rather than return prediction.
