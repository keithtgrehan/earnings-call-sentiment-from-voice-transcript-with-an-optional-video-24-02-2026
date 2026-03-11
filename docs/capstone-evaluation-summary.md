# Capstone Evaluation Summary

## Setup
- Frozen benchmark: `data/gold_guidance_calls/labels.csv` with 9 canonical gold rows
- Holdout benchmark: `data/gold_guidance_calls_holdout/labels.csv`
- Evaluator: `scripts/evaluate_gold_benchmark.py`
- Label set: `raised`, `maintained`, `lowered`, `withdrawn`, `unclear`

## Results
- Frozen benchmark: `9/9`
- Initial holdout before the first narrow refinement: `1/2`
- Initial holdout after the comparative-update refinement: `2/2`
- Expanded holdout after adding more unseen rows: `4/7`

## What the expanded holdout showed
- The engine stayed conservative on the `unclear` rows it saw.
- It still missed several explicit `raised` cases on unseen official-source excerpts.
- The main new failure pattern is straightforward: sentences that say management is `raising` guidance or outlook were still mapped to `unclear`.

## Practical judgment
- The deterministic baseline is strong enough to justify the current transcript-first capstone framing.
- It is not stable enough yet to call the guidance-change mapper finished.
- The next step should be one more narrow refinement pass targeted only at explicit `raising` phrasing, followed by another holdout rerun.

## Limits
- The sample sizes are still small.
- The expanded holdout uses a mix of transcript excerpts and official-source excerpts where direct media collection was blocked.
- These results support decision-support positioning only. They do not support any predictive, alpha, or statistical-significance claim.
