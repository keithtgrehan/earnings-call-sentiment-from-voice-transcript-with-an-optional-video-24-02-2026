# AI Earnings Call Signal Engine
Converts earnings call audio into deterministic trading signals plus a reproducible backtest harness.

## 1) Why this exists (Problem)
- Earnings calls are noisy, unstructured, and time-sensitive.
- Manual analysis does not scale across companies or quarters.
- Without standardized artifacts, comparing quarter-to-quarter signal movement is difficult.

## 2) What it does (Solution)
This repo builds deterministic, auditable outputs from call audio:
- Sentiment timeline (`sentiment_timeline.png`)
- Tone change detection (`tone_changes.csv`)
- Guidance extraction (`guidance.csv`)
- Guidance revision vs prior run (`guidance_revision.csv`)
- Run-level metrics and report (`metrics.json`, `report.md`)
- Run metadata for backtesting (`run_meta.json`)
- Optional question-shift artifacts (`question_sentiment_shifts.csv`, `question_shifts.png`)

Notes:
- Core scoring is deterministic and traceable from output columns.
- Optional narrative layer exists in `scripts/run_eval.py` (`--llm none|ollama`), but the default pipeline does not require LLM calls.
- Backtest supports pressure/confidence features if provided in run outputs (`management_confidence.csv`, `analyst_pressure.csv`), and safely handles them as missing otherwise.

## 3) How it works (Pipeline)
```text
YouTube/local audio
  -> audio normalization (ffmpeg, mono 16k wav)
  -> transcription (faster-whisper)
  -> segment sentiment scoring (transformers)
  -> chunk-level feature table (signed sentiment + probs)
  -> deterministic heuristics
       - guidance extraction
       - guidance revision matching/classification
       - tone change z-score detection
  -> artifacts (csv/json/jsonl/png/md)
  -> backtest harness (returns + statistical tests)
```

Conceptually:
- NLP chunking produces time-indexed text windows.
- Feature extraction creates a compact feature vector per call.
- Deterministic defaults minimize LLM token spend and maximize reproducibility.

## 4) Technology stack
- Speech-to-text: `faster-whisper`
- NLP: `transformers`
- Compute: `PyTorch`
- Data/statistics: `pandas`, `numpy`, `scipy`
- Audio/data acquisition: `ffmpeg`, `librosa`, `yt-dlp`
- Tooling: `ruff`, `pytest`, `git`

Free best-practice alternatives note:
- `ffmpeg` is the standard for reliable media conversion.
- `yt-dlp` is best-in-class for local extraction workflows.
- `librosa` is a standard Python audio utility library.
- `ruff` + `pytest` + `git` are standard lightweight Python dev tooling.

## 5) Install
### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ruff pytest
```

### Clean reinstall (avoid stale console script)
```bash
pip uninstall -y earnings-call-sentiment earnings_call_sentiment || true
pip install -e .
hash -r
```

## 6) Quickstart
### Minimal run (no VAD)
```bash
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=nlsBt74HCug" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model tiny \
  --chunk-seconds 20 \
  --min-chars 20 \
  --symbol AAPL \
  --event-dt "2024-08-01T16:00:00" \
  --verbose
```

### Run with VAD enabled
```bash
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=nlsBt74HCug" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model tiny \
  --chunk-seconds 20 \
  --min-chars 20 \
  --vad \
  --symbol AAPL \
  --event-dt "2024-08-01 16:00" \
  --verbose
```

### Prior-guidance comparison (two-pass)
```bash
# prior run
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=nlsBt74HCug" \
  --cache-dir ./cache_prior \
  --out-dir ./outputs_prior \
  --model tiny --chunk-seconds 20 --min-chars 20 \
  --symbol AAPL --event-dt "2024-05-01T16:00:00"

# current run compared to prior guidance.csv
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=nlsBt74HCug" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model tiny --chunk-seconds 20 --min-chars 20 \
  --prior-guidance ./outputs_prior/guidance.csv \
  --symbol AAPL --event-dt "2024-08-01T16:00:00"
```

## 7) CLI reference (key flags)
- `--vad`: enable VAD filtering during transcription.
- `--force`: rerun post-score deterministic stages even if outputs exist.
- `--resume` / `--no-resume`: reuse or recompute post-score artifacts.
- `--tone-change-threshold`: z-score threshold for tone-change detection.
- `--prior-guidance`: prior `guidance.csv` path for revision labeling.
- `--symbol`: ticker stored in `run_meta.json`.
- `--event-dt`: event timestamp stored in `run_meta.json`.

Note: a `--strict` CLI flag is not currently present. Contract checks are enforced via `scripts/verify_outputs.py` (plus `--require-run-meta` when needed).

## 8) Outputs Contract
Core contract artifacts (full scoring run):
- `run_meta.json` (strict keys)
- `metrics.json`
- `report.md`
- `guidance.csv`
- `guidance_revision.csv`
- `tone_changes.csv`
- `transcript.json`
- `transcript.txt`
- `sentiment_segments.csv`
- `chunks_scored.jsonl`

Optional artifacts (flag-dependent):
- `question_sentiment_shifts.csv`
- `question_shifts.png`

### `run_meta.json` schema (strict keys)
```json
{
  "symbol": "<STRING>",
  "event_dt": "<ISO8601 timestamp>",
  "source_url": "<youtube_url>",
  "run_id": "<string>",
  "generated_at": "<ISO8601 timestamp>",
  "version": "<package version or git sha if available>"
}
```

### CSV/JSONL required columns
`sentiment_segments.csv`
- `start`, `end`, `text`, `sentiment`, `score`

`chunks_scored.jsonl` (one JSON object per line)
- `start`, `end`, `text`, `sentiment`, `score`, `signed_score`, `positive_prob`, `negative_prob`

`guidance.csv`
- `start`, `end`, `text`, `sentiment`, `score`, `topic`, `period`, `numbers`, `numeric_signature`, `midpoint_hint`, `guidance_strength`, `count_numbers`, `has_percent`, `has_range`, `has_currency`, `matched_cues`

`guidance_revision.csv`
- `row_id`, `is_matched`, `revision_label`, `topic`, `period`, `current_start`, `current_end`, `prior_start`, `prior_end`, `current_text_snippet`, `prior_text_snippet`, `current_numbers`, `prior_numbers`, `current_midpoint`, `prior_midpoint`, `diff`, `overlap_score`, `current_match_key`, `prior_match_key`

`tone_changes.csv`
- `start`, `end`, `sentiment_score`, `rolling_mean_5`, `rolling_std_5`, `tone_change_z`, `is_change`, `text`

`question_sentiment_shifts.csv` (optional)
- `question_time`, `question_text`, `sentiment_before`, `sentiment_after`, `sentiment_shift`

Contract verification:
```bash
python scripts/verify_outputs.py --out-dir ./outputs
python scripts/verify_outputs.py --out-dir ./outputs --require-run-meta
```

## 9) Backtesting + Statistical significance
Workflow:
1. Generate one or more run folders with valid `run_meta.json` and output artifacts.
2. Convert broker export into canonical `prices.csv`.
3. Run deterministic backtest harness.

### Convert IBKR export
```bash
python scripts/ibkr_prices_to_prices_csv.py \
  --in ./ibkr_export.csv \
  --out ./prices.csv \
  --timezone UTC
```

### Run backtest
```bash
python scripts/backtest_signals.py \
  --runs-dir ./runs \
  --prices-csv ./prices.csv \
  --event-window "0h:1h,0h:1d,close:close" \
  --out-dir ./outputs
```

Backtest outputs:
- `backtest_results.csv` (per-event rows and feature columns)
- `backtest_summary.json` (correlations with p-values, regression coefficients, high-vs-low t-test p-values, bootstrap CI)
- `backtest_report.md`

Acceptance criteria for alpha claims:
- Treat predictive claims as valid only when out-of-sample tests remain stable and p-value < 0.05.
- Prefer repeated results across symbols/windows over one-off wins.

## 10) Course alignment (AI Master themes)
- NLP preprocessing and chunking
- Feature engineering from unstructured transcripts
- Deterministic scoring and auditable heuristics
- Evaluation metrics and statistical testing
- Experiment design (train/test split by event date)
- CI/testing discipline (`ruff`, `pytest`)
- Reproducible artifact-first pipelines

## 11) Roadmap (next 3 weeks)
Week 1
- Tighten output contract checks and add run manifest/index.
- Add quarter-over-quarter tone delta summaries.

Week 2
- Build multi-company feature dataset builder from run artifacts.
- Add benchmark slices by sector/symbol.

Week 3
- Expand IBKR-driven backtests: p-values, multiple-hypothesis correction (Benjamini-Hochberg), and walk-forward evaluation.

## 12) Contributing / Dev
```bash
ruff check src tests
pytest -q
```

Branch naming:
- Use `codex/<feature-name>` for Codex-driven branches.
