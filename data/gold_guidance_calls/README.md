# Gold Guidance Calls Benchmark

This folder holds the local benchmark materials for the guidance-change MVP.

## Files
- `call_manifest.csv`: Source-of-truth manifest for benchmark call metadata and source URLs.
- `raw_calls/*.txt`: Local transcript text files copied from the repo CLI transcription path.
- `transcription_status.csv`: Current transcript completion status and basic file stats.
- `draft_labels.csv`: Provisional one-row-per-call label proposals for reviewer inspection only.
- `draft_label_review.md`: Human-readable rationale for each provisional draft label.
- `transcript_inventory.csv`: Lightweight keyword inventory to speed manual review.
- `labels.csv`: Final gold labels. This is the authoritative benchmark label file once rows are manually reviewed and approved.

## Review rule
`draft_labels.csv` is provisional and must not be treated as final gold truth until manually reviewed against the rubric in `docs/labeling_rubric.md`.
