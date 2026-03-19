# External Multimodal Datasets

This repo keeps external dataset support conservative and explicitly secondary to
the transcript-first capstone workflow.

What this scaffolding does:

- defines local staging roots and registry entries
- validates whether a dataset is present locally
- loads basic metadata into a shared normalized record shape
- documents how each dataset should be framed

What this scaffolding does not do:

- download datasets automatically
- add training code or notebooks
- change benchmark assets or capstone claims
- treat external datasets as replacements for deterministic repo outputs

## Priority Framing

### MAEC

- Priority: primary external reference
- Why it exists here:
  it is the only external dataset in this scaffold treated as domain-relevant to
  earnings-call style analysis
- What it should be used for:
  future calibration, metadata inspection, and cautious comparison work when the
  dataset has been staged locally
- What it should not be used for:
  replacing repo-native benchmark evidence, replacing deterministic labels, or
  inflating capstone claims

### MELD

- Priority: secondary calibration only
- Why it exists here:
  it can help with generic conversation-style metadata normalization and sanity
  checks, but it is not finance-domain evidence
- What it should be used for:
  secondary calibration or sanity-check work only
- What it should not be used for:
  finance-ground-truth claims, earnings-call benchmarking, or product truth

### RAVDESS

- Priority: secondary calibration only
- Why it exists here:
  it can help sanity-check audio/video feature plumbing and staged media parsing
- What it should be used for:
  secondary calibration or feature sanity checks only
- What it should not be used for:
  earnings-call inference, finance-domain benchmarking, or management-behavior
  claims

## Local Placement

The registry uses existing environment variables if they are set:

- `EARNINGS_CALL_MAEC_ROOT`
- `EARNINGS_CALL_MELD_ROOT`
- `EARNINGS_CALL_RAVDESS_ROOT`

If an env var is not set, the scaffold falls back to these repo-local staging
paths:

- `data/external_datasets/maec/`
- `data/external_datasets/meld/`
- `data/external_datasets/ravdess/`

The datasets do not need to exist locally. Missing roots are reported cleanly by
the validator.

## Expected Local Staging Layout

These are the layouts accepted by this repo's placeholder adapters. They are
staging conventions for this codebase, not claims that upstream releases always
arrive in exactly this shape.

### MAEC

Expected root:

- configured `EARNINGS_CALL_MAEC_ROOT`, or `data/external_datasets/maec/`

Expected staged files:

- one metadata file such as `metadata/records.csv`
- or `metadata/records.jsonl`
- or `records.csv`
- or `records.jsonl`

Optional supporting folders:

- `transcripts/`
- `audio/`
- `video/`

Normalization notes:

- normalized as domain-relevant external records
- preserves text/audio/video/transcript path fields when available
- keeps labels as metadata only

### MELD

Expected root:

- configured `EARNINGS_CALL_MELD_ROOT`, or `data/external_datasets/meld/`

Expected staged files:

- `train_sent_emo.csv`
- `dev_sent_emo.csv`
- `test_sent_emo.csv`

Optional supporting folders:

- `train_splits/`
- `dev_splits_complete/`
- `output_repeated_splits_test/`

Normalization notes:

- normalized as non-finance dialogue records
- keeps speaker, text, and optional inferred video path
- should remain secondary calibration/sanity-check material only

### RAVDESS

Expected root:

- configured `EARNINGS_CALL_RAVDESS_ROOT`, or `data/external_datasets/ravdess/`

Expected staged files:

- `Actor_01/`, `Actor_02/`, ... directories
- `.wav` and/or `.mp4` media files inside actor directories
- standard file names such as `03-01-05-01-02-01-12.wav`

Normalization notes:

- normalized as acted-expression media metadata
- parses actor id, modality code, and emotion code from file names
- remains secondary calibration/sanity-check material only

## Shared Normalized Record Shape

Where feasible, the adapters normalize records into these common fields:

- `dataset_id`
- `record_id`
- `split`
- `modality`
- `text`
- `audio_path`
- `video_path`
- `transcript_path`
- `speaker_id`
- `speaker_role`
- `label_namespace`
- `original_label`
- `normalized_label`
- `source_context`
- `is_finance_domain`
- `notes`

This shape is meant for future inspection and cautious calibration work, not
for immediate model training.

## Validation Script

Run:

```bash
PYTHONPATH=src python scripts/validate_external_datasets.py
```

Validate one dataset only:

```bash
PYTHONPATH=src python scripts/validate_external_datasets.py --dataset maec
```

Override a root without editing config:

```bash
PYTHONPATH=src python scripts/validate_external_datasets.py \
  --dataset meld \
  --meld-root /path/to/local/MELD
```

The script prints a JSON report describing:

- whether the root exists
- whether the staged file structure looks usable
- which metadata files or media folders were detected
- a few normalized sample records when loading is possible
