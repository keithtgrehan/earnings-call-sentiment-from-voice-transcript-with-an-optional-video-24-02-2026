# A/B Test Plan

## Objective
This experiment evaluates whether the Earnings Call Signal Extraction Engine helps retail traders review earnings-call information more effectively than manual review alone. The goal is to measure practical decision-support value in a lightweight capstone setting, focusing on speed, clarity, and evidence-backed interpretation.

## Hypothesis
Primary hypothesis: Participants using the Earnings Call Signal Extraction Engine identify meaningful guidance shifts and tone-change moments faster than participants using manual transcript/material review.

Secondary hypothesis: Participants using the tool report higher confidence and provide better evidence-backed reasoning for their conclusions.

## Test Design
- Control: Participants review a raw transcript and standard earnings materials manually.
- Treatment: Participants review the structured output from the Earnings Call Signal Extraction Engine.
- Target participant type: Self-directed retail traders or active private investors who follow earnings announcements.
- Test task: For a single earnings call, identify key guidance shifts, tone-change moments, and the evidence snippets that support why each signal matters.

## Success Metrics
- Primary metric 1: Time-to-completion for producing a signal summary.
- Primary metric 2: Rubric score for correctly identifying key guidance shifts, key tone-change moments, and supporting transcript evidence (simple checklist-based scoring).
- Secondary metric 1: Self-reported confidence in conclusions (post-task survey).
- Secondary metric 2: Evidence quality score based on whether conclusions are supported by relevant transcript snippets.

## Method
- Run a small capstone validation session with participants who match the target user profile.
- Have each participant complete the same earnings-call task in either control or treatment (or in counterbalanced order if both are used).
- Record completion time, submitted conclusions, and cited evidence snippets for each task.
- Collect a short post-task survey for confidence and perceived usefulness.
- Score each submission with a simple rubric and compare control vs treatment patterns.

## Risks and Confounds
- Differences in participant baseline investing experience may affect speed and quality.
- Learning effects can bias results if participants see similar materials across conditions.
- Call-specific complexity can influence outcomes independent of tool quality.
- Small capstone sample sizes may limit statistical confidence and generalizability.

## Success Criteria
- Treatment shows a clear reduction in time-to-completion versus control in the proposed test design.
- Treatment improves conclusion quality and evidence support scores on the rubric.
- Treatment participants report higher confidence and usefulness, supporting further product iteration.
