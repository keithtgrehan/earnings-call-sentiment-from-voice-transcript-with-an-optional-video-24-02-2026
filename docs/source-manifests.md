# Source Manifest Scaffolding

This package defines a conservative, metadata-first structure for tracking real
earnings-call collection targets without scraping, downloading, or auto-labeling.

Canonical files:

- `data/source_manifests/earnings_call_sources.csv`
- `data/source_manifests/earnings_call_segments.csv`

Validation entry point:

- `python scripts/validate_source_manifests.py`
- `python scripts/validate_source_pairs.py`

## Purpose

- track call-level source candidates before collection
- track segment-level extraction targets before clipping or labeling
- keep planned multimodal collection separate from benchmark labels and runtime assets

The template rows in these manifests are examples only. Their URLs are
placeholders and are not treated as verified in-repo sources.

## Source Pairing Workflow

This repo uses a manual-entry plus validation workflow for source pairing.

Principles:

- the manifest is the source of truth
- official IR transcript metadata is preferred whenever available
- official replay video or official YouTube video can be paired as supporting
  visual metadata
- third-party transcript pairs must stay clearly marked as third-party
- this repo does not try to crawl or auto-discover all investor-relations sites

The source-pair helper validates and summarizes:

- required pair fields
- URL formatting
- allowed `source_family` and `layout_type` values
- whether transcript provenance is clearly marked as official, third-party, or
  missing
- whether video provenance is clearly marked as official, third-party, or
  missing
- pairing status summaries such as complete pair, missing transcript, and
  missing video

Optional URL reachability checks are available, but they are explicitly
lightweight and are not a crawler.

Run:

```bash
PYTHONPATH=src python scripts/validate_source_pairs.py
```

Optional lightweight URL checks for non-template rows:

```bash
PYTHONPATH=src python scripts/validate_source_pairs.py --check-urls
```

Template example rows are handled conservatively. They remain valid planning
examples even when they use placeholder URLs or intentionally incomplete
transcript/video status markers.

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

## Example Seed Coverage

The current example manifest is only a curated seed list. It does not imply
automatic retrieval from all investor-relations sites.

Recommended manual-curation seed examples include:

- Microsoft FY26 Q2
- NVIDIA quarterly results
- Bank of America Q4 2025
- Disney Q1 FY26
- Disney Q4 FY25
- HSBC results
- AstraZeneca results
- Starbucks investor/prepared remarks
- Alphabet Q1 2025
- Alphabet Q4 2025
- Apple Q1 2025
- Amazon Q4 2024
- Intel Q2 2025
- Airbnb Q2 2025
- Oracle Q4 2025

## Intentionally Out Of Scope

- no automatic source discovery
- no automatic downloads
- no transcript parsing
- no model inference
- no benchmark labeling
- no edits to existing deterministic signal logic
