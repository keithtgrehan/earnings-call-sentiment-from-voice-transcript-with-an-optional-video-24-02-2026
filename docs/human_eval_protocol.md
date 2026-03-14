# Human Evaluation Protocol

## Purpose
This protocol defines a lightweight comparison test to evaluate whether the MVP helps retail users review guidance-change signals faster and more consistently.

## Runnable Package In This Repo
- Cases: `data/media_support_eval/task_impact_eval_cases.csv`
- Blank results sheet: `data/media_support_eval/task_impact_results_template.csv`
- Blank assignment grid: `data/media_support_eval/task_impact_assignment_template.csv`
- Summarizer: `PYTHONPATH=src .venv/bin/python scripts/summarize_task_impact_results.py`

Package status in this checkout:
- 23 independent labeled cases are packaged for the pilot.
- Cases that already had repo-local deterministic outputs point to their real treatment bundles.
- Cases that did not have a rerunnable treatment bundle now point to explicit transcript-only placeholder bundles under `outputs/downstream_decision_eval/`.
- Those placeholders are suitable for logistics/setup verification, but they are not equivalent to a fresh full deterministic review run.

## Evaluation Conditions
- A = retail trader alone
- B = retail trader with tool output

## Required Captured Fields
participant_id, condition, call_id, predicted_label, evidence_text, summary_text, start_time, submit_time, clarity_rating

## Protocol Steps
1. Assign each participant a call transcript and one condition (A or B).
2. Start timing at first transcript/tool exposure (`start_time`).
3. Participant submits one guidance-change label and supporting evidence text.
4. Participant adds a short summary and a clarity rating.
5. Record `submit_time` and calculate completion time.
6. Repeat across calls with balanced condition assignment.

Practical control/treatment mapping for the packaged cases:
- Control uses `baseline_transcript_path`.
- Treatment uses `treatment_report_path` plus `treatment_transcript_path` if the reviewer needs to inspect raw text alongside the report.
- If `treatment_report_path` points into `outputs/downstream_decision_eval/`, treat that row as a verified placeholder rather than a full regenerated treatment report.

## Scoring
- label accuracy: Compare `predicted_label` to the gold label.
- completion time: Measure elapsed time from `start_time` to `submit_time`.
- clarity rating: Use participant-provided rating as a usability signal.

## Current Interpretation
- No real participant submissions are committed in this repo today.
- The pilot scaffold is therefore made runnable and verifiable, not actually run end-to-end with human participants.
