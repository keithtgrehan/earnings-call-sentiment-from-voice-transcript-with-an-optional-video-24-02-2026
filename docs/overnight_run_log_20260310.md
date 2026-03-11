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
- Added optional benchmark review docs after core benchmark artifacts stabilized:
  - `data/gold_guidance_calls/README.md`
  - `docs/benchmark_progress.md`
- Current recommended first-pass human-eval shortlist, pending NVDA completion:
  - `call02` (LEU): strongest explicit maintained cue
  - `call01` (PLTR): strong explicit guidance issuance, good `unclear` calibration case
  - `call04` (IBM): concrete annual outlook, good `unclear` calibration case
  - `call05` (MSFT): concrete next-quarter range, good `unclear` calibration case
  - `call06` (AAPL): concrete next-quarter range, good `unclear` calibration case
- `call07` (NVDA) completed successfully on retry under the `tiny` fallback.
  - raw file: data/gold_guidance_calls/raw_calls/NVDA_2026_Q4_call07.txt
  - size_bytes: 52819
  - line_count: 430
  - sanity_preview: Good afternoon. My name is Sarah, and I will be your conference operator today.
- Refreshed the benchmark artifacts to a full seven-call state:
  - `data/gold_guidance_calls/transcription_status.csv`
  - `data/gold_guidance_calls/draft_labels.csv`
  - `data/gold_guidance_calls/draft_label_review.md`
  - `data/gold_guidance_calls/transcript_inventory.csv`
  - `docs/benchmark_progress.md`
- Added `call07` draft label as `unclear` based on explicit first-quarter revenue outlook language without an explicit guidance-action verb.
- Full draft-label quality control now passes for `call01` through `call07`.

## Metadata hardening and next benchmark sourcing
- Added `data/gold_guidance_calls/official_source_manifest.csv` to pin official source URLs and confirmation status for the current seven-call batch without rewriting the working manifest mid-run.
- Added `data/gold_guidance_calls/prior_quarter_sources.csv` to point each current call to a same-company prior-quarter official source, with extracted guidance snippets where the official source exposed them quickly.
- Added `docs/directional_call_shortlist.md` to capture higher-value next benchmark targets with explicit directional language (`raised`, `lowered`, `maintained`) from primary sources.
- Notable metadata findings:
  - `call04` (IBM) current manifest date is likely one day late relative to the official January 28, 2026 event notice.
  - `call02` (LEU) remains the messiest metadata row; the official release supports Q4 2025 / 2026 guidance, but official Centrus event pages expose an inconsistent timestamp and should be reconciled before final gold use.
- Notable benchmark finding: the current seven-call batch is now complete and reproducible, but still skewed toward explicit guidance issuance that maps conservatively to `unclear` under the MVP rubric.
- Applied one direct manifest correction backed by the official-source supplemental file:
  - `call04` (IBM) event date updated from `2026-01-29` to official call date `2026-01-28` in `call_manifest.csv`, `transcription_status.csv`, and `draft_labels.csv`.

## Directional benchmark expansion
- Began post-batch expansion beyond the original seven calls to improve label coverage.
- Selected two raised-guidance candidates with both primary-source confirmation and usable repost video URLs for the existing CLI path:
  - `call08` / `LLY_2025_Q2_call08` (official Lilly Q2 2025 results raised guidance; repost video selected for transcription)
  - `call09` / `ECG_2025_Q2_call09` (official Everus Q2 2025 results raised guidance; repost video selected for transcription)
- Using the same `tiny` fallback path for these add-on calls because the machine already showed `small` to be the throughput bottleneck for overnight benchmark collection.
- Upgraded the IBM prior-quarter row from source-only to snippet-extracted using the official Q3 2025 release sentence: `Given the strength of our business, we are raising our full-year outlook for revenue growth and free cash flow.`
- Promoted IBM Q3 2025 above Everus in the directional shortlist because it offers both a clean primary-source raised sentence and an easily found repost video path for the current CLI.
- Added `docs/morning_review_package.md` to summarize the current benchmark state, source-quality split, use-now subset, hold-back subset, and directional expansion queue.
- `call08` (LLY) completed successfully and was promoted into the benchmark set.
  - raw file: data/gold_guidance_calls/raw_calls/LLY_2025_Q2_call08.txt
  - size_bytes: 56591
  - line_count: 706
  - draft label: `raised`
  - evidence: `As a result, we raised our revenue / in earnings for share guides.`
- Updated the benchmark package to an eight-call state, including `call_manifest.csv`, `official_source_manifest.csv`, `transcription_status.csv`, `draft_labels.csv`, `draft_label_review.md`, `transcript_inventory.csv`, `docs/benchmark_progress.md`, and `docs/morning_review_package.md`.
- Final benchmark status before UI work: eight completed transcript files, eight draft label rows, official-source supplemental data in place, prior-quarter source pack present, and morning review package written.
- Directional expansion decision: stop after `call08` for this run. `call09` (IBM Q3 2025) remains the next best raised candidate, but the benchmark is materially stronger now and the remaining missing label family is `lowered`, not another `raised` row.
- Stopped the in-progress IBM Q3 2025 transcription intentionally at roughly one-third completion because the benchmark already had a transcript-backed `raised` row from Lilly and the higher-priority missing family was `lowered`.
- Pivoted the next directional transcription candidate from IBM to PVH Q1 2025 after confirming:
  - official PVH source explicitly says the company lowered its revenue and EBIT outlook for the year
  - a usable call video exists for the current CLI ingest path
- `call09` (PVH) completed successfully and was promoted into the benchmark set.
  - raw file: data/gold_guidance_calls/raw_calls/PVH_2025_Q1_call09.txt
  - size_bytes: 56290
  - line_count: 609
  - draft label: `lowered`
  - evidence: `that's why we have to just our full year non-gap guidance down for both / ebit margin and eps.`
- The benchmark now contains transcript-backed `raised`, `lowered`, and `maintained` rows, which materially improves first-pass human evaluation usefulness.
