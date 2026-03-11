# NVIDIA 2025 Historical Calls

This package tracks the four NVIDIA earnings calls that occurred during calendar year 2025.

## Scope note
NVIDIA's fiscal-quarter labels cross calendar years. To avoid guessing, the `quarter` column preserves NVIDIA's official fiscal-quarter naming:
- `Q4_FY2025` on 2025-02-26
- `Q1_FY2026` on 2025-05-28
- `Q2_FY2026` on 2025-08-27
- `Q3_FY2026` on 2025-11-19

## Purpose
- keep NVIDIA historical call coverage separate from the active frozen benchmark and current holdout benchmark
- preserve official event dates and source references for the four 2025 call events
- leave `labels.csv` empty until any future quarter-specific labeling is deliberately performed

## Current package contents
- `call_manifest.csv`: official quarter/date/source rows for the 2025 call-history pack
- `official_source_manifest.csv`: event-page verification for each call date
- `transcription_status.csv`: official outlook excerpt collection state for each quarter
- `labels.csv`: header-only placeholder; no labels were appended in this run
