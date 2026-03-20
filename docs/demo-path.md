# Demo Path

## Best Current Demo Case

Use [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md) as the main reviewer walk-through.

Why this case:

- it has a complete committed deterministic bundle on `main`
- it also has committed audio support, a saved multimodal support summary, and the restored NLP sidecar example
- it appears in the committed prototype review slice, so a reviewer can inspect both the case artifacts and the saved prototype-review evidence

## Open These Files In Order

1. [report.md](../outputs/MSFT_2026_Q2_call05/report.md)
   Start here for the reviewer-friendly summary. It shows the overall signal, confidence note, strongest evidence snippets, and a clear separation between deterministic findings and multimodal support.
2. [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
   Open this next if you want the same assessment in machine-readable form. It exposes the scorecard structure, counts, and category explanations without having to parse the markdown report.
3. [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt)
   Use the transcript to sanity-check the quoted guidance language and uncertainty evidence from the report. This is the underlying review truth for the case.
4. [labels.csv](../data/gold_guidance_calls/labels.csv)
   Check the `call05` row to see the frozen benchmark label and why it remains `unclear`: explicit revenue guidance exists, but the committed benchmark notes do not find an explicit raised, maintained, lowered, or withdrawn change verb.
5. [audio_behavior_summary.json](../outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json)
   Read this only after the transcript-first pass. It shows that usable audio support exists and stays observational, with notes about hesitation, pauses, and why the audio support is considered supportive.
6. [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
   This is the compact sidecar roll-up. It keeps the transcript assessment primary, records audio as supportive, and shows that video support was unavailable because quality gates were not met.
7. [nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json)
   This is the restored NLP sidecar example on `main`. It is useful for showing that one committed source has inspection-only NLP coverage, but it is still explicitly secondary to the deterministic label path.
8. [multimodal_review_results_codex_proto.csv](../data/media_support_eval/multimodal_review_results_codex_proto.csv) and [multimodal_review_summary.json](../outputs/media_support_eval/multimodal_review_summary.json)
   End here if the reviewer wants to see how this case participated in the first committed prototype artifact-review slice and what the saved descriptive summary actually contains.

## Deterministic Findings

- The current committed MSFT report is `amber` with `55%` review confidence.
- The deterministic package surfaces explicit revenue guidance, but the prior-guidance comparison block is empty in the committed bundle.
- The strongest deterministic caution comes from high uncertainty / hedging rather than from analyst hostility or a definitive guidance-revision signal.
- The frozen benchmark label for `call05` remains `unclear`, which is consistent with the transcript-first conservative labeling rule.

## Supporting Multimodal Sidecars

- Audio support is present, usable, and marked supportive in the committed summary.
- The multimodal roll-up keeps video support unavailable because quality gates were not met.
- The restored NLP sidecar exists for this same Microsoft case and covers `561` rows, but its own notes keep deterministic labels as the source of truth.
- The committed prototype review row for `call05` still landed on `unclear`, which helps show that sidecars were treated as supporting context rather than label-changing proof.

## Limitations

- This path is intentionally transcript-first. Review the sidecars only after reading the deterministic report, metrics, transcript, and frozen label.
- The committed demo bundle is minimal. Use the files linked above rather than assuming every runtime byproduct from a full local run is present on `main`.
- The saved prototype review summary is descriptive only. It is not a human-subject study, does not prove multimodal lift, and does not establish statistical significance.
- This case does not show alignment coverage, and its video support is explicitly unavailable in the committed multimodal summary.
