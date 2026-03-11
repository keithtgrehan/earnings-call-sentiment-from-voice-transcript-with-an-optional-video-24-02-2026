# Evaluation Baseline

## Frozen benchmark
- Source of truth: `data/gold_guidance_calls/labels.csv`
- Scope: 9 calls
- Current result: `9/9`

## Initial holdout benchmark
- Source of truth: `data/gold_guidance_calls_holdout/labels.csv`
- Initial labeled scope: 2 calls
- Current result on that initial 2-row holdout: `2/2`

## Evaluator
- Script: `scripts/evaluate_gold_benchmark.py`
- Benchmark sanity utility: `scripts/summarize_gold_benchmark.py`
- Method: current deterministic transcript-to-guidance extraction path plus a fixed closed-set sentence mapper over extracted guidance text

## Limits
- This is a label-agreement baseline only.
- It is not evidence of predictive edge, alpha, or statistical significance.
- The initial holdout remained very small even after the first unseen pass, so broader generalization still needed a larger unseen set.
