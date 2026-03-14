# Media Support Eval

This package is a small repo-native seed set for calibrating the earnings-call
audio and video support layers.

It is intentionally conservative:
- transcript remains the primary evidence layer
- audio/video labels are reviewer-support labels, not truth or deception labels
- the current set is strong enough for an initial audio model pass
- the visual set is still best treated as calibration data until more unique
  webcast sources are added

## Current Contents
- `media_manifest.csv`: local media and feature-artifact sources already present in this repo
- `segment_labels.csv`: seed labels for audio hesitation, delivery confidence,
  visual tension, media quality, and multimodal support direction
- `runtime_smoke_manifest.csv`: runtime/status coverage rows for the repo-local
  webcast variants currently available
- `downstream_decision_eval_cases.csv`: fixed downstream case pack built from
  the repo's labeled transcript assets
- `task_impact_eval_cases.csv`: human-task case pack for the lightweight pilot
- `task_impact_results_template.csv`: blank submission template for the pilot
- `task_impact_assignment_template.csv`: blank participant/case assignment grid

## Scope Notes
- Sources are real repo-local earnings-call runs only.
- No gated external datasets were downloaded for this package.
- The current committed media labels now cover 70 rows total:
  - 52 audio rows
  - 18 video rows
  - 12 nonblank visual-tension rows across 2 source groups
- Visual labeling is now strong enough for a basic grouped check across two
  source groups, but it is still short of a more defensible three-group
  evaluation target.
- The downstream/task-impact package now covers 23 independent labeled calls
  from the repo's frozen, holdout, and watchlist-derived benchmark assets.
- Of those 23 downstream cases, only 5 currently carry source-level
  media-support targets from `segment_labels.csv`; the rest remain transcript-
  first packaged cases until more media-support labels are added.
- Missing treatment bundles in the task-impact package are explicit transcript-
  only placeholders, not hidden full reruns.

## Label Meaning
- `media_quality_label`: whether the segment quality is poor, usable, or strong
- `hesitation_pressure_label`: low / medium / high audio hesitation under questioning
- `visual_tension_label`: low / medium / high visible tension or motion pressure
- `delivery_confidence_label`: low / medium / high delivery confidence support
- `multimodal_support_direction`: supportive / cautionary / neutral / unavailable

These labels are for reviewer support only and should not be interpreted as
emotion truth, deception detection, or trading recommendation signals.
