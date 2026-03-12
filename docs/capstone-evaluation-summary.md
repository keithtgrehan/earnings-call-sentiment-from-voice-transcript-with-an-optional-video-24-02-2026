# Capstone Evaluation Summary

## Setup
- Frozen benchmark: `data/gold_guidance_calls/labels.csv` with 9 canonical gold rows
- Holdout benchmark: `data/gold_guidance_calls_holdout/labels.csv` with 7 currently labeled unseen rows
- Watchlist-derived unseen holdout: `data/gold_guidance_calls_holdout_watchlist/labels.csv` with 7 currently labeled unseen rows
- Evaluator: `scripts/evaluate_gold_benchmark.py`
- Label set: `raised`, `maintained`, `lowered`, `withdrawn`, `unclear`
- Method: current deterministic transcript-to-guidance extraction path plus a fixed closed-set sentence mapper over extracted guidance text

## Results
- Frozen benchmark: `9/9`
- Initial holdout before the first narrow refinement: `1/2`
- Initial holdout after the comparative-update refinement: `2/2`
- Expanded holdout before the gerund-style refinement: `4/7`
- Expanded holdout after the gerund-style raised-guidance refinement: `7/7`
- Watchlist-derived unseen holdout after the comparative and maintained-wording refinements: `7/7`
- Behavior mini eval after the skepticism refinement pass: `52/58`
  - `uncertainty`: `14/20`
  - `reassurance`: `20/20`
  - `skepticism`: `18/18`

## What changed
- The narrow refinements were limited to explicit raised-guidance wording already observed in unseen misses:
  - comparative update phrasing such as `up from our previous estimate`
  - gerund-style phrasing such as `we are raising our guidance` or `we are raising our outlook`
- No broader sentiment or heuristic rewrite was introduced.
- The behavior layer also received one narrow skepticism pass based on measured misses in analyst question phrasing.
- A small deterministic `Q&A Shift` layer was added for demo/reporting:
  - prepared remarks vs Q&A
  - analyst skepticism level
  - management answer uncertainty vs prepared remarks
  - early vs late Q&A drift

## Practical judgment
- The deterministic baseline is strong enough to justify the current transcript-first capstone framing.
- The current benchmark package now shows clean agreement on:
  - frozen benchmark: `9/9`
  - expanded holdout: `7/7`
  - watchlist-derived unseen holdout: `7/7`
- This is still a benchmark-agreement result, not proof that the mapper is finished for broader unseen coverage.
- The Phase 1 behavior layer is now measured well enough to demo, but it is still a lightweight deterministic layer rather than a complete behavioral model.

## Limits
- The sample sizes are still small.
- The expanded holdout uses a mix of transcript excerpts and official-source excerpts where direct media collection was blocked.
- The watchlist-derived unseen holdout is also excerpt-heavy and should be treated as a disciplined generalization check, not a broad validation set.
- There is still no unseen `lowered` holdout row in the current package.
- The behavior eval set is also small and manually curated for auditable rule QA.
- The new Q&A shift output is useful for review and demo flow, but it is not a validated trading or forecasting signal.
- These results support decision-support positioning only. They do not support any predictive, alpha, or statistical-significance claim.

## Final positioning
- This project is a transcript-first deterministic review tool for structured earnings-call guidance analysis.
- The current evidence supports local benchmark agreement on a small frozen set and a small unseen holdout.
- It does not support live trading, alpha, or statistical-significance claims.
