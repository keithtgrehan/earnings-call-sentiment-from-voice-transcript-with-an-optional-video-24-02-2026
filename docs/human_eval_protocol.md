# Human Evaluation Protocol

## Purpose
This protocol defines a lightweight comparison test to evaluate whether the MVP helps retail users review guidance-change signals faster and more consistently.

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

## Scoring
- label accuracy: Compare `predicted_label` to the gold label.
- completion time: Measure elapsed time from `start_time` to `submit_time`.
- clarity rating: Use participant-provided rating as a usability signal.
