# Evaluation Baseline

## Frozen benchmark
- Source of truth: `data/gold_guidance_calls/labels.csv`
- Scope: 9 calls
- Current result: `9/9`

## Holdout benchmark
- Source of truth: `data/gold_guidance_calls_holdout/labels.csv`
- Initial labeled scope: 2 calls
- Initial result before the first narrow raised-guidance refinements: `2/2`
- Current labeled scope: 7 calls
- Current result on the expanded holdout: `7/7`
- Current holdout composition remains small and excerpt-heavy:
  - `raised`: 4
  - `maintained`: 1
  - `lowered`: 0
  - `withdrawn`: 0
  - `unclear`: 2

## Watchlist-derived unseen holdout
- Source of truth: `data/gold_guidance_calls_holdout_watchlist/labels.csv`
- Current labeled scope: 7 calls
- Current result: `7/7`
- Current composition:
  - `raised`: 3
  - `maintained`: 1
  - `lowered`: 0
  - `withdrawn`: 0
  - `unclear`: 3
- This package is separate from the active holdout and was built from watchlist-derived candidate rows that were defensible enough to promote into a second unseen check.

## Evaluator
- Script: `scripts/evaluate_gold_benchmark.py`
- Benchmark sanity utility: `scripts/summarize_gold_benchmark.py`
- Method: current deterministic transcript-to-guidance extraction path plus a fixed closed-set sentence mapper over extracted guidance text

## Evaluation evidence
- Frozen benchmark agreement: `9/9`
- Expanded holdout agreement: `7/7`
- Watchlist-derived unseen holdout agreement: `7/7`
- This is evidence of closed-label agreement on the current benchmark packages only.
- It is not evidence of predictive edge, alpha, or statistical significance.

## Limits
- This is a label-agreement baseline only.
- It is not evidence of predictive edge, alpha, or statistical significance.
- The holdout remains small and uses a mix of transcript excerpts and official-source excerpts where direct media collection was blocked.
- The watchlist-derived unseen holdout is also small and excerpt-heavy.
- The current results justify transcript-first decision-support positioning, not broader claims about generalization or trading performance.
