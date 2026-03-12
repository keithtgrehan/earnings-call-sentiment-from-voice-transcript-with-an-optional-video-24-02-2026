# Gold Guidance Holdout Pack

This folder holds the first unseen holdout package for guidance-change evaluation.

Purpose:
- keep the frozen 9-call benchmark unchanged
- stage a separate transcript-collection and labeling queue for generalization testing
- preserve the same local package conventions used by `data/gold_guidance_calls/`

Working rules:
- `labels.csv` is empty until holdout rows are actually transcribed and labeled
- `call_manifest.csv` is the candidate queue for collection
- `official_source_manifest.csv` records the best locally available primary-source metadata
- `transcription_status.csv` tracks collection state and should stay `pending` until transcripts exist
- do not mix these holdout rows into the frozen benchmark evaluation until they are fully labeled

Selection criteria used for the initial queue:
- target 6-10 calls
- prefer explicit guidance language when the primary source already suggests raised / maintained / lowered language
- include likely `unclear` controls where local source material is clean enough to support conservative labeling later
- prefer official or primary-source investor materials when locally available
- if event date or direction is not well supported locally, keep the row and mark the risk explicitly instead of guessing
