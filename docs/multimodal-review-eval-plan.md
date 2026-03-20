# Multimodal Review Evaluation Plan

## Purpose
This plan defines the smallest honest next-step evaluation for testing whether supporting multimodal sidecars help reviewers work faster, more consistently, or with better evidence traceability than transcript-first review alone.

This is a prototype-level user-study plan. It is not market proof, not a trading study, and not a claim that multimodal already improves outcomes.

## Evaluation Question
Does a transcript-first review workflow with supporting multimodal sidecars help reviewers complete earnings-call review tasks faster, more consistently, or with more traceable evidence than a transcript-first workflow without those sidecars?

## Conditions
- `transcript_first_core`
  Same canonical transcript-first review package used today: transcript-backed deterministic outputs remain the review truth.
- `transcript_first_plus_supporting_sidecars`
  The same transcript-first core plus supporting multimodal artifacts when they exist, such as clip-based visual summaries, rolled-up multimodal status, or other sidecar status surfaces.

The sidecar condition should only add supporting evidence. It should not change the canonical transcript-backed deterministic outputs.

## Recommended Case Source
- Primary pilot case list: `data/media_support_eval/task_impact_eval_cases.csv`
- Optional later extension: subset to downstream comparison or curated slice cases only when those rows have real repo-local supporting artifacts

## Proposed Metrics
### Reviewer speed
- `completion_seconds`

### Reviewer consistency
- `label_accuracy` against the existing gold guidance label
- within-condition case agreement using repeated reviews of the same `case_id`

### Evidence traceability
- `evidence_traceability_rating`
- `clarity_rating`
- citation behavior via `cited_artifact_paths`
- treatment-only descriptive metric: how often reviewers actually cited supporting multimodal artifacts

## Minimal Study Shape
1. Use the existing pilot case list.
2. Assign reviewers to `transcript_first_core` or `transcript_first_plus_supporting_sidecars`.
3. Capture one guidance label, supporting evidence text, and a short summary.
4. Record completion time, clarity, and evidence traceability.
5. Repeat enough cases so at least some `case_id` values are reviewed by multiple people per condition.

## Smallest Scaffold In This Repo
- Blank results sheet: `data/media_support_eval/multimodal_review_results_template.csv`
- Descriptive summarizer: `python3 scripts/summarize_multimodal_review_results.py`

## Current Interpretation Boundary
- Treat results as descriptive until enough counterbalanced observations exist.
- Do not claim statistical significance from a small pilot.
- Do not claim predictive improvement, trading edge, or market usefulness from this scaffold alone.
- Do not treat any positive result as proof that sidecars should replace transcript-first review truth.

## What This Still Does Not Prove
- No market edge
- No predictive lift
- No statistical significance from small samples
- No proof that visual or NLP sidecars improve decisions in production
- No reason to weaken transcript-first discipline
