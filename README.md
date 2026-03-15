# Earnings Call Signal Engine
Transcript-first deterministic review tool for extracting structured, auditable earnings-call signals from transcript, audio, and optional video inputs.

This project is a local decision-support system for earnings-call review. It is not a live trading system, does not execute orders, and does not claim predictive edge or statistical significance.

## Project Summary
The repo converts one earnings call into structured, auditable artifacts that help a reviewer inspect:
- guidance statements and guidance-change evidence
- tone-change moments
- deterministic behavior signals such as uncertainty, reassurance, and analyst skepticism
- deterministic Q&A shift summaries
- deterministic audio behavior summaries for pause / hesitation review support
- optional observational visual-behavior summaries for video-capable runs
- deterministic scorecard presentation for reviewer-friendly ranking and triage
- transcript-backed supporting spans
- report and metrics outputs for local review

The current workflow is transcript-first and deterministic-first across audio, video, and text inputs. Optional narrative layers exist, but the deterministic artifacts remain the source of truth.

## What Is Implemented Now
- End-to-end CLI pipeline via `earnings-call-sentiment` for:
  - YouTube or local media ingestion
  - audio normalization
  - transcription
  - segment and chunk scoring
  - deterministic guidance extraction
  - deterministic guidance revision comparison
  - deterministic tone-change detection
  - deterministic behavior-signal extraction:
    - uncertainty / hedging
    - management reassurance
    - analyst skepticism
  - deterministic Q&A shift summary artifacts
  - deterministic audio behavior monitoring for answer-level pause / hesitation support:
    - pause-before-answer
    - answer-onset delay
    - filler density
    - lightweight hesitation summaries
  - optional visual behavior monitoring for video-capable runs:
    - frame sampling
    - face visibility / face presence analysis
    - motion and head-shift proxies
    - segment-level visual stability summaries
  - deterministic scorecard presentation layer that ranks current review evidence into reviewer-friendly categories
  - report, metrics, and artifact generation
- Primary local review UI shell served by `app/site_server.py`
- Benchmark subset shown in the UI and now sourced from canonical gold labels, not draft labels
- Frozen 9-call gold guidance benchmark under `data/gold_guidance_calls/`
- Gold benchmark summary utility:
  - `python scripts/summarize_gold_benchmark.py`

## Behavior Layer
Phase 1 behavioral monitoring is implemented with deterministic, auditable rules only:
- uncertainty / hedging
- management reassurance
- analyst skepticism / hostility

The behavior layer is observational. It is not emotion-truth inference, not deception detection, and not a claim about hidden intent.

Current measured result on the small internal behavior eval set:
- overall: `58/58`
- `uncertainty`: `20/20`
- `reassurance`: `20/20`
- `skepticism`: `18/18`

This is a deterministic rule-QA check on a small curated set, not a statistical validation set.

## Q&A Shift
The repo now includes a deterministic Q&A shift layer that summarizes:
- prepared remarks vs Q&A differences
- analyst question pressure context
- management answer uncertainty relative to the prepared baseline

Outputs include:
- `qa_shift_segments.csv`
- `qa_shift_summary.json`

## Audio Behavior
The repo now includes a deterministic audio behavior support layer for answer-level review. It is observational only and is meant to help reviewers inspect pauses and hesitation patterns around management answers.

Outputs include:
- `audio_behavior_segments.csv`
- `audio_behavior_summary.json`

This layer is not emotion inference, not lie detection, and not a truth detector.

## Review Scorecard
The repo now includes a deterministic scorecard presentation layer derived from existing guidance, behavior, and Q&A artifacts.

It adds:
- six reviewer-facing category scores on a `1-10` scale
- green / amber / red bands
- an overall review signal
- a review-confidence percentage
- short explanations and strongest evidence snippets

The scorecard is presentation-only. It does not replace raw artifacts, does not change benchmark labels, and does not add trading or alpha claims.

Review confidence means confidence in the tool's interpretation of the available deterministic evidence, not investment confidence.

## Current Review Layers
- guidance extraction and guidance revision comparison
- behavioral text signals: uncertainty, reassurance, analyst skepticism
- deterministic Q&A shift
- deterministic audio behavior support
- optional visual behavior support
- deterministic scorecard presentation layer

Audio and visual layers are supporting, confidence-tagged review aids. They are not truth detectors and should not be presented as hidden-state inference.

## Repo-Native Media Support Set
- committed media-support labels now cover 102 segments:
  - 84 audio
  - 18 video
- 9 of the 23 downstream comparison cases now carry source-level media-support targets
- visual tension still covers 12 labeled rows across 2 independent source groups, so the visual layer remains calibration-only and underpowered relative to the transcript/audio path

## Current Review Outputs
Artifacts currently produced by the core pipeline include:
- `transcript.json`, `transcript.txt`
- `chunks_scored.jsonl`, `sentiment_segments.csv`
- `guidance.csv`, `guidance_revision.csv`, `tone_changes.csv`
- `uncertainty_signals.csv`, `reassurance_signals.csv`, `analyst_skepticism.csv`
- `behavioral_summary.json`, `qa_shift_segments.csv`, `qa_shift_summary.json`
- `audio_behavior_segments.csv`, `audio_behavior_summary.json` when audio support is available
- `visual_behavior_frames.csv`, `visual_behavior_segments.csv`, `visual_behavior_summary.json` for video-capable runs
- `metrics.json`, including a `review_scorecard` block with:
  - `overall_review_signal`
  - `review_confidence_pct`
  - ranked categories, scores, bands, explanations, and strongest evidence
- `report.md`, which surfaces the scorecard alongside the raw deterministic summaries
- `run_meta.json`

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

## Evaluation Evidence
- Frozen benchmark agreement on canonical gold labels: `9/9`
- Active unseen holdout agreement on current labeled rows: `7/7`
- Watchlist-derived unseen holdout agreement on current labeled rows: `7/7`
- Behavior mini-eval agreement on current labeled rows: `58/58`

These are benchmark-agreement and rule-QA results only. They do not establish predictive edge, statistical significance, or finance-specific generalization.

## Benchmark And Package Layout
The repo now contains six distinct benchmark or benchmark-adjacent data areas:

- `data/gold_guidance_calls/`
  - canonical frozen guidance-change benchmark
  - active source of truth for the `9/9` frozen evaluation result
- `data/gold_guidance_calls_holdout/`
  - active unseen holdout benchmark
  - separate from the frozen benchmark and used for current unseen agreement checks
- `data/gold_guidance_calls_holdout_watchlist/`
  - second unseen holdout built from watchlist-derived candidate rows
  - current agreement checkpoint: `7/7`
  - separate from both the frozen benchmark and the original active holdout
- `data/watchlist_earnings_candidates/`
  - metadata-first future candidate pool sourced from the supplied watchlist
  - not an active benchmark and not a canonical label source
  - may include official-source excerpts where full transcript collection was blocked
- `data/nvda_2025_historical_calls/`
  - separate NVIDIA calendar-year 2025 call-history pack
  - preserves NVIDIA's official fiscal-quarter labels for the four earnings calls that occurred during calendar year 2025
  - not part of the frozen benchmark or active holdout benchmark
- `data/behavior_signal_eval/`
  - small internal evaluation set for the behavior layer
  - separate from the guidance-change benchmarks
  - used to measure uncertainty, reassurance, and skepticism rules

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
python scripts/evaluate_gold_benchmark.py --benchmark-root data/gold_guidance_calls_holdout_watchlist --output-dir outputs/holdout_watchlist_eval --benchmark-name "Watchlist Holdout Benchmark Evaluation"
```

Useful companion docs:
- `docs/evaluation-baseline.md`
- `docs/capstone-evaluation-summary.md`
- `data/watchlist_earnings_candidates/README.md`
- `data/nvda_2025_historical_calls/README.md`

### Behavior mini-eval
Run:

```bash
python scripts/summarize_behavior_eval_set.py
python scripts/evaluate_behavior_signal_set.py
```

## What Remains Unproven
- No proven predictive edge
- No statistical significance claim
- No live trading claim
- No claim that the current deterministic rule set is validated beyond the current frozen benchmark and local review workflow
- No emotion-truth, intent, or deception claim from the behavior layer

The current baseline is useful for structured review, but it still needs broader unseen evaluation before stronger claims are justified.

## Immediate Next Steps
- Add more defensible unseen holdout rows, especially `lowered` cases
- Keep rerunning frozen, active holdout, and watchlist-holdout evaluation as new unseen rows are added
- Keep the benchmark package separate from any predictive or backtest claims
- Keep multimodal visual monitoring observational and optional rather than changing the transcript-first baseline

## Run Instructions
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

## Limitations
- Deterministic artifacts remain the source of truth
- The guidance benchmark is still small, and both unseen sets remain limited and excerpt-heavy
- The behavior eval set is small and internal; it is useful for rule QA, not for broad claims
- Behavior outputs are observational review aids, not emotion/deception inference
- Visual behavior outputs are low-confidence under poor framing, low face visibility, or sparse usable frames
- Any predictive or trading-performance claims require separate out-of-sample evaluation work

## Notes On Scope
- The frozen guidance benchmark is fixed at 9 calls for the current core evaluation pass
- Draft benchmark files still exist for review history, but they are no longer the canonical label source
- Candidate and history packages are separate from benchmark packages and should not be treated as active gold sets
