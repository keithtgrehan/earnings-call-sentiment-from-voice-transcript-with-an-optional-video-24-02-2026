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

## Behavior mini eval
- Source of truth: `data/behavior_signal_eval/`
- Current labeled scope: 58 rows
- Evaluator: `scripts/evaluate_behavior_signal_set.py`
- Current overall agreement: `58/58 = 100.0%`
- Current family-level agreement:
  - `uncertainty`: `20/20 = 100.0%`
  - `reassurance`: `20/20 = 100.0%`
  - `skepticism`: `18/18 = 100.0%`
- This eval set is separate from the guidance benchmark packages and is used only for deterministic behavior-signal QA.
- The current result reflects narrow measured refinements over explicit caution and probing phrase families; it is not evidence that the full Q&A behavior layer is complete.

## Current review stack
- Transcript-first guidance extraction and revision comparison remain the benchmark backbone.
- Deterministic behavioral text signals remain measured by the current `58/58` behavior mini eval.
- Deterministic `Q&A Shift` outputs are part of the current review product and support reviewer interpretation.
- Deterministic audio behavior outputs are also part of the current review product, but they are support layers rather than benchmark labels.
- Optional visual behavior outputs remain observational support for video-capable runs.
- The reviewer scorecard is a deterministic presentation layer built from existing artifacts; it does not change benchmark labels or replace raw evidence.
- `review_confidence_pct` means confidence in the tool's interpretation of the deterministic evidence, not investment confidence.
- Audio and visual outputs should be presented as supporting review aids, not truth detection, deception detection, or alpha claims.

## Evaluator
- Script: `scripts/evaluate_gold_benchmark.py`
- Benchmark sanity utility: `scripts/summarize_gold_benchmark.py`
- Method: current deterministic transcript-to-guidance extraction path plus a fixed closed-set sentence mapper over extracted guidance text

## Evaluation evidence
- Frozen benchmark agreement: `9/9`
- Expanded holdout agreement: `7/7`
- Watchlist-derived unseen holdout agreement: `7/7`
- Behavior mini-eval agreement: `58/58`
- This is evidence of closed-label agreement on the current benchmark packages only.
- It is not evidence of predictive edge, alpha, or statistical significance.

## Limits
- This is a label-agreement baseline only.
- It is not evidence of predictive edge, alpha, or statistical significance.
- The holdout remains small and uses a mix of transcript excerpts and official-source excerpts where direct media collection was blocked.
- The watchlist-derived unseen holdout is also small and excerpt-heavy.
- The behavior eval set is also small and manually selected for deterministic QA, not for broad statistical claims.
- The current results justify transcript-first decision-support positioning, not broader claims about generalization or trading performance.
