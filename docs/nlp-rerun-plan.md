# NLP Rerun Plan For Remaining Assessed Sources

This note maps the exact heavier rerun path that would be required to extend NLP sidecar support beyond `msft_fy26_q2_example`.

One bounded GOOGL rerun attempt was executed on 2026-03-21 after this plan was first written. It confirmed that the existing frozen CLI can start real transcription work from the cached `audio.wav`, but it did not emit `transcript.json`, `transcript.txt`, or `chunks_scored.csv` within the bounded watch window used for that pass.

## Current Truth

- `msft_fy26_q2_example` is still the only assessed source with committed NLP sidecar artifacts on `main`
- `goog_q1_2025_example`, `bac_q4_2025_example`, `dis_q1_fy26_example`, and `sbux_prepared_remarks_example` still lack scorer-ready transcript/chunk inputs on `main`
- the current blocker map remains in [nlp-sidecar-blockers.md](nlp-sidecar-blockers.md)

## Existing Supported Path

The current repo-local NLP path is still:

1. create `transcript.json` and `transcript.txt`
2. create `chunks_scored.csv`
3. run [scripts/run_nlp_segment_scoring.py](../scripts/run_nlp_segment_scoring.py) against `chunks_scored.csv`

The smallest existing code path that does this without changing frozen logic is the existing CLI:

- `PYTHONPATH=src python3 -m earnings_call_sentiment.cli ... --transcribe-only`
- `PYTHONPATH=src python3 -m earnings_call_sentiment.cli ... --score-only`
- `PYTHONPATH=src python3 scripts/run_nlp_segment_scoring.py --source-id ... --chunks-csv ...`

## Shortcuts That Are Not Recommended

- Do not score raw `transcript.txt` directly. The current scorer does not support that input.
- Do not use the document-review path in [review_workflow.py](../src/earnings_call_sentiment/review_workflow.py) as a substitute for source-level reruns. It can synthesize segment timings from plain text, but that would invent chunk boundaries and would not be cleanly comparable to the current MSFT path.
- Do not rely on the removed curated multimodal runner. It is not kept on canonical `main`.

## Shared Setup

Run from the canonical `main` repo root and point at the sibling local feature worktree that still holds the cached source files:

```bash
export FEAT_REPO=../earnings-call-sentiment-from-voice-transcript-with-an-optional-video-24-02-2026
```

Use stable repo-relative staging directories so any successful rerun stays attributable:

- cache: `cache/nlp_rerun/<source_id>/`
- deterministic output bundle: `outputs/nlp_rerun/<source_id>/`
- NLP sidecar output: default `data/processed/multimodal/nlp/<source_id>/`

## Per-Source Recipes

### `goog_q1_2025_example`

Current repo-local inputs:

- feature-worktree-only cache files:
  - `cache/curated_multimodal_slice/goog_q1_2025_example/audio.wav`
  - `cache/curated_multimodal_slice/goog_q1_2025_example/audio.mp3`
  - `cache/curated_multimodal_slice/goog_q1_2025_example/video.mp4`
  - `cache/curated_multimodal_slice/goog_q1_2025_example/transcript_source.pdf`
  - `cache/curated_multimodal_slice/goog_q1_2025_example/transcript.txt`
- committed visual sidecar artifacts already exist on `main`

Missing inputs to generate:

- `outputs/nlp_rerun/goog_q1_2025_example/transcript.json`
- `outputs/nlp_rerun/goog_q1_2025_example/chunks_scored.csv`

Commands:

```bash
PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/goog_q1_2025_example/audio.wav" \
  --cache-dir cache/nlp_rerun/goog_q1_2025_example \
  --out-dir outputs/nlp_rerun/goog_q1_2025_example \
  --transcribe-only \
  --symbol GOOGL \
  --event-dt 2025-04-24 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/goog_q1_2025_example/audio.wav" \
  --cache-dir cache/nlp_rerun/goog_q1_2025_example \
  --out-dir outputs/nlp_rerun/goog_q1_2025_example \
  --score-only \
  --symbol GOOGL \
  --event-dt 2025-04-24 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 scripts/run_nlp_segment_scoring.py \
  --source-id goog_q1_2025_example \
  --chunks-csv outputs/nlp_rerun/goog_q1_2025_example/chunks_scored.csv
```

Expected outputs:

- `outputs/nlp_rerun/goog_q1_2025_example/transcript.json`
- `outputs/nlp_rerun/goog_q1_2025_example/transcript.txt`
- `outputs/nlp_rerun/goog_q1_2025_example/chunks_scored.csv`
- `data/processed/multimodal/nlp/goog_q1_2025_example/nlp_segment_scores.csv`
- `data/processed/multimodal/nlp/goog_q1_2025_example/nlp_segment_scores.json`
- `data/processed/multimodal/nlp/goog_q1_2025_example/nlp_scoring_summary.json`

Risk: medium  
Effort: heavy  
Dependency scope: feature-worktree cache plus existing frozen CLI and current scorer  
Commit-worthiness: yes, if the rerun finishes cleanly and the new NLP outputs are accompanied by stable repo-relative transcript/chunk staging outputs  
Recommendation: go first if a heavier rerun is approved

Observed bounded attempt on 2026-03-21:

- command used the existing frozen `--transcribe-only` path against a temp copy of `audio.wav`
- early useful output included `cache/tmp_nlp_restore/goog_q1_2025_example/audio_normalized.wav`
- the run reached `113.9s / 3278.6s` of source audio after about `93s` of wall time
- no `transcript.json` or `transcript.txt` appeared before the attempt was stopped
- practical takeaway: the path is real, but on current hardware it is still too heavy for a short bounded pass unless a longer run window is explicitly approved

### `dis_q1_fy26_example`

Current repo-local inputs:

- feature-worktree-only cache files:
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/audio.wav`
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/audio.mp3`
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/audio.webm`
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/video.mp4`
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/transcript_source.pdf`
  - `cache/curated_multimodal_slice/dis_q1_fy26_example/transcript.txt`
- committed visual sidecar artifacts already exist on `main`

Missing inputs to generate:

- `outputs/nlp_rerun/dis_q1_fy26_example/transcript.json`
- `outputs/nlp_rerun/dis_q1_fy26_example/chunks_scored.csv`

Commands:

```bash
PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/dis_q1_fy26_example/audio.wav" \
  --cache-dir cache/nlp_rerun/dis_q1_fy26_example \
  --out-dir outputs/nlp_rerun/dis_q1_fy26_example \
  --transcribe-only \
  --symbol DIS \
  --event-dt 2026-02-02 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/dis_q1_fy26_example/audio.wav" \
  --cache-dir cache/nlp_rerun/dis_q1_fy26_example \
  --out-dir outputs/nlp_rerun/dis_q1_fy26_example \
  --score-only \
  --symbol DIS \
  --event-dt 2026-02-02 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 scripts/run_nlp_segment_scoring.py \
  --source-id dis_q1_fy26_example \
  --chunks-csv outputs/nlp_rerun/dis_q1_fy26_example/chunks_scored.csv
```

Expected outputs:

- `outputs/nlp_rerun/dis_q1_fy26_example/transcript.json`
- `outputs/nlp_rerun/dis_q1_fy26_example/transcript.txt`
- `outputs/nlp_rerun/dis_q1_fy26_example/chunks_scored.csv`
- `data/processed/multimodal/nlp/dis_q1_fy26_example/nlp_segment_scores.csv`
- `data/processed/multimodal/nlp/dis_q1_fy26_example/nlp_segment_scores.json`
- `data/processed/multimodal/nlp/dis_q1_fy26_example/nlp_scoring_summary.json`

Risk: medium  
Effort: heavy  
Dependency scope: feature-worktree cache plus existing frozen CLI and current scorer  
Commit-worthiness: yes, with the same stable-path caveat as GOOGL  
Recommendation: second-best approved rerun candidate

### `bac_q4_2025_example`

Current repo-local inputs:

- feature-worktree-only cache files:
  - `cache/curated_multimodal_slice/bac_q4_2025_example/audio.wav`
  - `cache/curated_multimodal_slice/bac_q4_2025_example/audio.mp3`
  - `cache/curated_multimodal_slice/bac_q4_2025_example/video.mp4`
  - `cache/curated_multimodal_slice/bac_q4_2025_example/transcript_source.pdf`
  - `cache/curated_multimodal_slice/bac_q4_2025_example/transcript.txt`
- committed visual sidecar artifacts already exist on `main`

Missing inputs to generate:

- `outputs/nlp_rerun/bac_q4_2025_example/transcript.json`
- `outputs/nlp_rerun/bac_q4_2025_example/chunks_scored.csv`

Commands:

```bash
PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/bac_q4_2025_example/audio.wav" \
  --cache-dir cache/nlp_rerun/bac_q4_2025_example \
  --out-dir outputs/nlp_rerun/bac_q4_2025_example \
  --transcribe-only \
  --symbol BAC \
  --event-dt 2026-01-14 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/bac_q4_2025_example/audio.wav" \
  --cache-dir cache/nlp_rerun/bac_q4_2025_example \
  --out-dir outputs/nlp_rerun/bac_q4_2025_example \
  --score-only \
  --symbol BAC \
  --event-dt 2026-01-14 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 scripts/run_nlp_segment_scoring.py \
  --source-id bac_q4_2025_example \
  --chunks-csv outputs/nlp_rerun/bac_q4_2025_example/chunks_scored.csv
```

Expected outputs:

- `outputs/nlp_rerun/bac_q4_2025_example/transcript.json`
- `outputs/nlp_rerun/bac_q4_2025_example/transcript.txt`
- `outputs/nlp_rerun/bac_q4_2025_example/chunks_scored.csv`
- `data/processed/multimodal/nlp/bac_q4_2025_example/nlp_segment_scores.csv`
- `data/processed/multimodal/nlp/bac_q4_2025_example/nlp_segment_scores.json`
- `data/processed/multimodal/nlp/bac_q4_2025_example/nlp_scoring_summary.json`

Risk: medium-high  
Effort: heavy  
Dependency scope: feature-worktree cache plus existing frozen CLI and current scorer  
Commit-worthiness: yes, if the rerun completes and the staging outputs are kept stable  
Recommendation: only after a smaller source succeeds

### `sbux_prepared_remarks_example`

Current repo-local inputs:

- feature-worktree-only cache files:
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/audio.wav`
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/audio.mp3`
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/video.mp4`
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/transcript_source.en.vtt`
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/transcript_source.en-orig.vtt`
  - `cache/curated_multimodal_slice/sbux_prepared_remarks_example/transcript.txt`
- committed visual sidecar artifacts already exist on `main`

Missing inputs to generate:

- `outputs/nlp_rerun/sbux_prepared_remarks_example/transcript.json`
- `outputs/nlp_rerun/sbux_prepared_remarks_example/chunks_scored.csv`

Commands:

```bash
PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/sbux_prepared_remarks_example/audio.wav" \
  --cache-dir cache/nlp_rerun/sbux_prepared_remarks_example \
  --out-dir outputs/nlp_rerun/sbux_prepared_remarks_example \
  --transcribe-only \
  --symbol SBUX \
  --event-dt 2026-01-28 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 -m earnings_call_sentiment.cli \
  --audio-path "$FEAT_REPO/cache/curated_multimodal_slice/sbux_prepared_remarks_example/audio.wav" \
  --cache-dir cache/nlp_rerun/sbux_prepared_remarks_example \
  --out-dir outputs/nlp_rerun/sbux_prepared_remarks_example \
  --score-only \
  --symbol SBUX \
  --event-dt 2026-01-28 \
  --model small \
  --chunk-seconds 30

PYTHONPATH=src python3 scripts/run_nlp_segment_scoring.py \
  --source-id sbux_prepared_remarks_example \
  --chunks-csv outputs/nlp_rerun/sbux_prepared_remarks_example/chunks_scored.csv
```

Expected outputs:

- `outputs/nlp_rerun/sbux_prepared_remarks_example/transcript.json`
- `outputs/nlp_rerun/sbux_prepared_remarks_example/transcript.txt`
- `outputs/nlp_rerun/sbux_prepared_remarks_example/chunks_scored.csv`
- `data/processed/multimodal/nlp/sbux_prepared_remarks_example/nlp_segment_scores.csv`
- `data/processed/multimodal/nlp/sbux_prepared_remarks_example/nlp_segment_scores.json`
- `data/processed/multimodal/nlp/sbux_prepared_remarks_example/nlp_scoring_summary.json`

Risk: high  
Effort: heavy  
Dependency scope: feature-worktree cache plus existing frozen CLI and current scorer  
Commit-worthiness: only if the rerun is clean and the resulting transcript/chunk staging looks reviewable; the cached transcript-side assets themselves are not clean enough to use directly  
Recommendation: last candidate on the current frozen-path route

## Ranking

1. `goog_q1_2025_example`
2. `dis_q1_fy26_example`
3. `bac_q4_2025_example`
4. `sbux_prepared_remarks_example`

Reasoning:

- `goog_q1_2025_example` is the best next attempt because it has the smallest observed local WAV, a clean local transcript PDF, and the strongest overlap with already committed visual evidence.
- `dis_q1_fy26_example` is second because it also has a clean local transcript PDF and a narrower transcript text than BAC.
- `bac_q4_2025_example` is likely the heaviest straight audio rerun on current hardware.
- `sbux_prepared_remarks_example` is the riskiest current-path attempt because the transcript-side local artifacts are WebVTT-derived and messy, so there is no clean cheap fallback if the audio rerun is slow or fails.

## Counts Refresh Gap

If any future rerun succeeds, there is still a separate small follow-up issue:

- current `main` does not retain the older curated multimodal runner that originally refreshed `multimodal_source_coverage.csv` and `multimodal_eval_summary.json`
- so the minimal future rerun pass should first focus on creating real scorer-ready inputs and real NLP sidecar outputs
- only after that should a separate approved step refresh the rolled-up counts on `main`
