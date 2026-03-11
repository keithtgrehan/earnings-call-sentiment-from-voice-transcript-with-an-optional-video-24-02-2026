# Earnings Call Signal Engine
Transcript-first AI tool for extracting structured signals from earnings call audio and video sources using NLP.

This project is a local decision-support system for earnings-call review. It is not a live trading system, does not execute orders, and does not claim predictive edge or statistical significance.

## Project Summary
The repo converts one earnings call into structured, auditable artifacts that help a reviewer inspect:
- guidance statements and guidance-change evidence
- tone-change moments
- transcript-backed supporting spans
- report and metrics outputs for local review

The current workflow is transcript/audio-first and deterministic-first. Optional narrative layers exist, but the deterministic artifacts remain the source of truth.

## What Is Implemented Now
- End-to-end CLI pipeline via `earnings-call-sentiment` for:
  - YouTube or local media ingestion
  - audio normalization
  - transcription
  - segment and chunk scoring
  - deterministic guidance extraction
  - deterministic guidance revision comparison
  - deterministic tone-change detection
  - report, metrics, and artifact generation
- Primary local review UI shell served by `app/site_server.py`
- Benchmark subset shown in the UI and now sourced from canonical gold labels, not draft labels
- Frozen 9-call gold guidance benchmark under `data/gold_guidance_calls/`
- Gold benchmark summary utility:
  - `python scripts/summarize_gold_benchmark.py`

Artifacts currently produced by the core pipeline include:
- `transcript.json`, `transcript.txt`
- `chunks_scored.jsonl`, `sentiment_segments.csv`
- `guidance.csv`, `guidance_revision.csv`, `tone_changes.csv`
- `metrics.json`, `report.md`, `run_meta.json`

Optional heuristic outputs still exist, but they are not the current benchmark focus:
- question-shift artifacts
- optional summary/eval utilities
- offline backtest scripts

## UI / Local Review Shell
The active local review shell is the primary interface served by:

```bash
PORT=7872 python app/site_server.py
```

Current shell structure:
- hero and project positioning
- left-side configuration flow:
  - input source
  - transcript/media/document input
  - metadata and transcription settings
  - deterministic vs additive summary mode
  - primary run action
- dossier / outputs area below the configuration flow
- compact right rail with:
  - recent local runs
  - benchmark subset
  - workflow notes

This is a local review product shell for analyst-style inspection. It is not a production SaaS app.

## Gold Benchmark
The frozen gold benchmark lives under:

```text
data/gold_guidance_calls/
```

Frozen scope:
- `call01`
- `call02`
- `call03`
- `call04`
- `call05`
- `call06`
- `call07`
- `call08`
- `call09`

Canonical files:
- `data/gold_guidance_calls/labels.csv`
- `data/gold_guidance_calls/call_manifest.csv`

`labels.csv` is now the canonical source of truth for benchmark labels.

The UI benchmark subset reads from canonical gold `labels.csv` and uses manifest data only for supporting display metadata such as source URL, quality flag, and notes.

Current frozen gold label distribution:
- `raised`: 1
- `maintained`: 1
- `lowered`: 1
- `withdrawn`: 0
- `unclear`: 6

This benchmark is intentionally conservative. If a transcript contains explicit forward guidance but no explicit direction change versus prior guidance, the gold label stays `unclear`.

## Utility Scripts
### Gold benchmark summary
Run:

```bash
python scripts/summarize_gold_benchmark.py
```

It prints:
- total benchmark row count
- label distribution
- one compact per-call table with:
  - `call_id`
  - `ticker`
  - `company`
  - `quarter`
  - `event_date`
  - `guidance_change_label`

It exits nonzero if:
- `labels.csv` is missing
- duplicate `call_id` rows exist
- any label falls outside:
  - `raised`
  - `maintained`
  - `lowered`
  - `withdrawn`
  - `unclear`

### Benchmark evaluator
Run:

```bash
python scripts/evaluate_gold_benchmark.py
python scripts/evaluate_gold_benchmark.py --benchmark-root data/gold_guidance_calls_holdout --output-dir outputs/holdout_eval --benchmark-name "Expanded Holdout Benchmark Evaluation"
```

Useful companion docs:
- `docs/evaluation-baseline.md`
- `docs/capstone-evaluation-summary.md`

### Evaluation evidence
- Frozen benchmark agreement on canonical gold labels: `9/9`
- Expanded unseen holdout agreement on current labeled rows: `7/7`
- The holdout remains small and excerpt-heavy.
- These are benchmark-agreement results only, not predictive or statistical-significance results.

## What Remains Unproven
- No proven predictive edge
- No statistical significance claim
- No live trading claim
- No claim that the current deterministic rule set is validated beyond the current frozen benchmark and local review workflow

The current baseline is useful for structured review, but it still needs broader unseen evaluation before stronger claims are justified.

## Immediate Next Steps
- Add more defensible unseen holdout rows, especially `lowered` cases
- Keep rerunning frozen and holdout evaluation as new unseen rows are added
- Keep the benchmark package separate from any predictive or backtest claims

## Usage / Validation
### Compile check
```bash
python -m py_compile app/server.py app/site_server.py src/earnings_call_sentiment/web_backend.py
```

### Test suite
```bash
pytest -q
```

### Gold benchmark sanity check
```bash
python scripts/summarize_gold_benchmark.py
```

### Launch the active local review shell
```bash
PORT=7872 python app/site_server.py
```

### Run the CLI pipeline
```bash
earnings-call-sentiment \
  --youtube-url "<earnings-call-url>" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model small \
  --chunk-seconds 30 \
  --symbol <TICKER> \
  --event-dt "2024-08-01T16:00:00" \
  --verbose
```

## Notes On Scope
- Deterministic artifacts remain the source of truth
- The benchmark is frozen at 9 calls for the current guidance-change evaluation pass
- Draft benchmark files still exist for review history, but they are no longer the canonical label source
- Any predictive or trading-performance claims require separate out-of-sample evaluation work
