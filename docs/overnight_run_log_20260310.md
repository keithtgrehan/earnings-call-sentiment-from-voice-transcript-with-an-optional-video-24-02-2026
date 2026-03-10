# Overnight Run Log - 2026-03-10

## Start
- UTC start: 2026-03-10T22:27:55Z
- Repository: `/Users/keith/GitHub/earnings-call-sentiment-from-voice-transcript-with-an-optional-video-24-02-2026`
- Starting branch: `mvp-guidance-benchmark`
- Working branch for this run: `codex/overnight/guidance-benchmark-batch`

## Goals
1. Verify benchmark call metadata (ticker/company/quarter/date confidence).
2. Transcribe benchmark calls with existing CLI (`--transcribe-only`, `--model small`, `--chunk-seconds 30`).
3. Build draft benchmark artifacts (`call_manifest.csv`, `transcription_status.csv`, `draft_labels.csv`, `draft_label_review.md`, `transcript_inventory.csv`).
4. Preserve ambiguity explicitly and avoid scope expansion.

## Pre-existing working tree state (before this run)
- Modified: `data/gold_guidance_calls/labels.csv` (existing manual row)
- Untracked: `data/gold_guidance_calls/raw_calls/PLTR_2025_Q4_call01.txt`
- Action: preserved as-is; no overwrite/deletion.

## Action log
- Confirmed repo path, active virtualenv usage, branch, and git status.
- Inspected pre-existing `labels.csv` diff.
- Created overnight branch from current HEAD: `codex/overnight/guidance-benchmark-batch`.

## Phase 1 - Metadata verification
- Pulled metadata for 7 benchmark URLs via yt-dlp (skip_download).
- Created `data/gold_guidance_calls/call_manifest.csv` with call_id call01..call07.
- Used explicit title/channel/upload metadata and conservative confidence flags.
- Marked third-party repost uploads as `benchmark_quality_flag=caution`.
- Event-date assumptions recorded in manifest notes when taken from description instead of upload date.
- Started batch transcription loop from manifest using CLI transcribe-only path.
- Initial run began with call01, which was already present in raw transcript set; process was stopped to avoid redundant spend.
- Batch then continued to call02 (`LEU_2025_Q4_call02`) with model=small.
- call02 transcription completed successfully.
  - raw file: data/gold_guidance_calls/raw_calls/LEU_2025_Q4_call02.txt
  - size_bytes: 45097
  - line_count: 1148
  - sanity_preview: Welcome to Centrist Energy Fourth Quarter and Four-Year 2025 Earnings Squall.
- call03 transcription started (GOOGL_2025_Q4_call03).
- call03 transcription completed successfully.
  - raw file: data/gold_guidance_calls/raw_calls/GOOGL_2025_Q4_call03.txt
  - size_bytes: 53870
  - line_count: 1668
  - sanity_preview: Welcome everyone. Thank you for standing by for the Alphabet 4th Quarter 2025 earnings conference call.
- call04 transcription started (IBM_2025_Q4_call04).
- Runtime note: `--model small` proved materially slow on this machine for long calls (roughly 10 to 15 minutes per call observed on call02/call03).
- Fallback decision: switched remaining missing calls (starting with call04) to `--model tiny` using the same repo CLI path to maximize overnight benchmark coverage while preserving deterministic reproducibility.
- Created current benchmark control artifacts from completed transcripts:
  - `data/gold_guidance_calls/transcription_status.csv`
  - `data/gold_guidance_calls/draft_labels.csv`
  - `data/gold_guidance_calls/draft_label_review.md`
  - `data/gold_guidance_calls/transcript_inventory.csv`
- Current draft label state:
  - `call01` (PLTR): `unclear`, high-confidence unclear
  - `call02` (LEU): `maintained`, based on explicit phrase "our guidance is flat"
  - `call03` (GOOGL): `unclear`, outlook commentary without explicit change verb
- `call04` (IBM) completed successfully under the `tiny` fallback.
  - raw file: data/gold_guidance_calls/raw_calls/IBM_2025_Q4_call04.txt
  - size_bytes: 57400
  - line_count: 633
  - sanity_preview: Welcome and thank you for standing by. At this time, participants are in a listen only mode.
- Added `call04` draft label as `unclear` based on explicit 2026 outlook language without an explicit guidance-action verb.
- `call05` (MSFT) started under the same `tiny` fallback path after IBM completed.
- `call05` (MSFT) completed successfully under the `tiny` fallback.
  - raw file: data/gold_guidance_calls/raw_calls/MSFT_2026_Q2_call05.txt
  - size_bytes: 52105
  - line_count: 561
  - sanity_preview: Greetings and welcome to the Microsoft Fiscal Year 2026 second quarter earnings conference call.
- `call06` (AAPL) completed successfully under the `tiny` fallback.
  - raw file: data/gold_guidance_calls/raw_calls/AAPL_2026_Q1_call06.txt
  - size_bytes: 48050
  - line_count: 527
  - sanity_preview: Good afternoon and welcome to the Apple Q1 fiscal year 2026 earnings conference call.
- Refreshed benchmark support artifacts after the sixth completed transcript:
  - `data/gold_guidance_calls/transcription_status.csv`
  - `data/gold_guidance_calls/draft_labels.csv`
  - `data/gold_guidance_calls/draft_label_review.md`
  - `data/gold_guidance_calls/transcript_inventory.csv`
- Added draft labels:
  - `call05` (MSFT): `unclear`, explicit Q3 revenue range without an explicit guidance-action verb
  - `call06` (AAPL): `unclear`, explicit March-quarter revenue range without an explicit guidance-action verb
- Quality-control pass completed on current draft set.
  - All CSV artifacts open with expected headers.
  - Evidence offsets validated for `call01` through `call06`.
  - Spot-checked transcript previews for PLTR, MSFT, and AAPL.
- `call07` (NVDA) initial `tiny` run stopped without writing `transcript.txt`; restarted the same repo CLI command for a clean retry.
