# NLP Sidecar Blockers For Assessed Sources

This note captures the current honest limit of NLP sidecar coverage on `main` after checking current artifacts, branch history, and the local `feat/multimodal-sidecars` worktree.

Current committed truth remains:

- `msft_fy26_q2_example` is the only assessed source with committed NLP sidecar artifacts on `main`
- `data/processed/multimodal/eval/multimodal_eval_summary.json` still reports `sources_with_nlp = 1`
- `data/processed/multimodal/eval/multimodal_eval_summary.json` still reports `segments_with_nlp_support = 561`

## Existing Scorer Requirements

The current repo-local NLP scorer already exists at `scripts/run_nlp_segment_scoring.py`, but it only accepts:

- `chunks_scored.csv`
- `chunks_scored.jsonl`
- or a real `transcript.json` plus a segment manifest with non-placeholder text windows or real timestamps

It does not score raw `transcript.txt` files directly.

## Source Inventory

| source_id | Repo-local inputs found | Directly usable by current NLP scorer? | Minimal conversion needed | Would that stay small-scope? | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `msft_fy26_q2_example` | Committed NLP outputs under `data/processed/multimodal/nlp/msft_fy26_q2_example/`; committed review bundle under `outputs/MSFT_2026_Q2_call05/` | yes | none | yes | already restored |
| `goog_q1_2025_example` | Committed visual artifacts on `main`; branch-local source and segment manifests on `feat/multimodal-sidecars`; local feature-worktree cache contains `audio.wav`, `audio.mp3`, `video.mp4`, `transcript_source.pdf`, and `transcript.txt` under `cache/curated_multimodal_slice/goog_q1_2025_example/` | no | generate `transcript.json` and `chunks_scored.csv` from repo-local audio via the frozen core pipeline | no, this is a heavyweight rerun rather than a tiny restore step | blocked |
| `bac_q4_2025_example` | Committed visual artifacts on `main`; branch-local manifests on `feat/multimodal-sidecars`; local feature-worktree cache contains `audio.wav`, `audio.mp3`, `video.mp4`, `transcript_source.pdf`, and `transcript.txt` under `cache/curated_multimodal_slice/bac_q4_2025_example/` | no | generate `transcript.json` and `chunks_scored.csv` from repo-local audio via the frozen core pipeline | no, same blocker shape as GOOGL | blocked |
| `dis_q1_fy26_example` | Committed visual artifacts on `main`; branch-local manifests on `feat/multimodal-sidecars`; local feature-worktree cache contains `audio.wav`, `audio.mp3`, `audio.webm`, `video.mp4`, `transcript_source.pdf`, and `transcript.txt` under `cache/curated_multimodal_slice/dis_q1_fy26_example/` | no | generate `transcript.json` and `chunks_scored.csv` from repo-local audio via the frozen core pipeline | no, same blocker shape as GOOGL | blocked |
| `sbux_prepared_remarks_example` | Committed visual artifacts on `main`; branch-local manifests on `feat/multimodal-sidecars`; local feature-worktree cache contains `audio.wav`, `audio.mp3`, `video.mp4`, `transcript_source.en.vtt`, `transcript_source.en-orig.vtt`, and `transcript.txt` under `cache/curated_multimodal_slice/sbux_prepared_remarks_example/` | no | either create `chunks_scored.csv` from a fresh frozen-pipeline run or introduce a transcript-text preprocessing path | no, current scorer still does not accept raw `transcript.txt` directly | blocked |

## Restore Candidates Found

- `feat/multimodal-sidecars:data/source_manifests/earnings_call_sources.csv`
- `feat/multimodal-sidecars:data/source_manifests/earnings_call_segments.csv`
- local `feat/multimodal-sidecars` cache files under `cache/curated_multimodal_slice/<source_id>/`

These are real repo-local materials, but they do not by themselves make the four non-MSFT sources NLP-ready:

- the segment manifest rows are still `planned:` placeholders rather than real transcript text
- the non-MSFT sources do not have committed `chunks_scored.csv` or `transcript.json`
- the current scorer does not accept raw `transcript.txt` alone

## Bounded Extension Attempt

I tested the narrowest honest path on `goog_q1_2025_example`:

- use the existing frozen CLI on repo-local cached audio
- write temporary outputs in the current `main` worktree
- only continue to NLP scoring if real transcript and chunk artifacts appeared

That attempt normalized audio but did not emit `transcript.json`, `transcript.txt`, or `chunks_scored.csv` within a short bounded run window, so I stopped there rather than turn this pass into four heavyweight reruns.

## Why Counts Stayed Unchanged

No new NLP artifacts were added on `main`, so the committed summary counts remain the truthful state:

- `sources_with_nlp = 1`
- `segments_with_nlp_support = 561`

## Smallest Honest Next Step

If a later pass explicitly allows a heavier restore path, the next honest move would be:

1. run the existing frozen CLI on the repo-local cached audio for one source at a time
2. confirm that real `transcript.json` and `chunks_scored.csv` artifacts are produced
3. run `scripts/run_nlp_segment_scoring.py` on those chunk files
4. refresh multimodal coverage counts only after the new NLP sidecars are real and reviewable

Until then, MSFT remains the only honestly supported assessed NLP source on `main`.
