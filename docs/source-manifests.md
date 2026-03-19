# Source Manifest Scaffolding

This package defines a conservative, metadata-first structure for tracking real
earnings-call collection targets without scraping, downloading, or auto-labeling.

Canonical files:

- `data/source_manifests/earnings_call_sources.csv`
- `data/source_manifests/earnings_call_segments.csv`

Validation entry point:

- `python scripts/validate_source_manifests.py`

## Purpose

- track call-level source candidates before collection
- track segment-level extraction targets before clipping or labeling
- keep planned multimodal collection separate from benchmark labels and runtime assets

The template rows in these manifests are examples only. Their URLs are
placeholders and are not treated as verified in-repo sources.

## Call-Level Fields

`earnings_call_sources.csv` contains one row per planned source candidate.

Core fields:

- `source_id`: stable manifest key
- `company`, `ticker`
- `event_title`, `fiscal_period`, `event_date`
- `source_family`, `layout_type`
- `video_url`, `transcript_url`
- `transcript_source_type`, `video_source_type`
- `has_prepared_remarks`, `has_qa`
- `language`
- `face_visibility_expectation`
- `notes`, `status`, `license_or_usage_notes`

## Segment-Level Fields

`earnings_call_segments.csv` contains one row per planned extraction target.

Core fields:

- `segment_id`: stable segment key
- `source_id`: foreign key into `earnings_call_sources.csv`
- `start_time`, `end_time`: blank until collection or annotation
- `segment_type`
- `speaker_name`, `speaker_role`
- `transcript_ref`: text anchor, span id, or planning note
- `face_expected`
- `visual_usability_label`
- `audio_usability_label`
- `labeling_status`
- `notes`

## Allowed Values

### `source_family`

- `official_investor_relations`
- `official_youtube`
- `official_results_page`
- `third_party_repost`
- `transcript_vendor`
- `transcript_only`

Use `official_*` values whenever the planned source comes directly from the
company or its investor-relations publishing surface.

### `layout_type`

- `single_speaker_camera`
- `single_speaker_with_slides`
- `multi_speaker_grid`
- `slides_only`
- `audio_only`
- `transcript_only`
- `unknown`

This field is intentionally coarse. It is for planning expected visual utility,
not frame-level description.

### `segment_type`

- `prepared_remarks`
- `q_and_a_question`
- `q_and_a_answer`
- `intro_or_safe_harbor`
- `closing_remarks`

Keep this set small so segment planning stays auditable and consistent.

### `visual_usability_label`

- `face_visible`
- `single_speaker_visible`
- `multi_speaker_visible`
- `slides_only_or_face_too_small`
- `visual_unusable`

Leave this field blank until the segment is actually reviewed. The label should
describe practical visual usefulness, not emotion or inferred intent.

## Conservative Labeling Notes

- blank `start_time` and `end_time` values are valid for planned rows
- blank `visual_usability_label` and `audio_usability_label` values are valid
  until review
- `labeling_status` should communicate progress instead of guessing labels early
- these manifests are planning and tracking artifacts, not evidence of collection

## Intentionally Out Of Scope

- no automatic source discovery
- no automatic downloads
- no transcript parsing
- no model inference
- no benchmark labeling
- no edits to existing deterministic signal logic
