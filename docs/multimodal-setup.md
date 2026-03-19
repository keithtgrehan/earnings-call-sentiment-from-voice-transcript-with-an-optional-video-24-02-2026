# Multimodal Setup

This page documents conservative setup scaffolding for optional multimodal
components. The current repo remains transcript-first. None of the tools below
are required for the existing deterministic transcript workflow.

## What Each Tool Does

- WhisperX:
  optional word-level alignment and timestamp refinement on top of ASR output
- pyannote.audio:
  optional speaker diarization support for segmenting audio by speaker
- OpenFace:
  optional external facial behavior extraction hook via system-installed binary
- FinBERT:
  optional finance-domain NLP assist model for transcript-side sentiment support
- emotion-english-distilroberta-base:
  optional secondary/supporting NLP assist for generic emotion-style text cues
- MAEC / MELD / RAVDESS:
  optional external dataset roots for future ingestion scaffolding only

## Optional Vs Required

- Required for the repo's current default behavior:
  none of the multimodal components on this page
- Optional Python sidecars:
  WhisperX and pyannote.audio
- Optional system-level tool:
  OpenFace
- Optional model ids:
  FinBERT and emotion-english-distilroberta-base
- Optional external dataset roots:
  MAEC, MELD, and RAVDESS

If you do not install any multimodal extras, the repo should continue to behave
exactly as it does today.

## Install Options

### Base repo only

Use the repo as-is. No multimodal extras are needed.

### Optional Python sidecars

Install from the optional requirements file:

```bash
pip install -r requirements-optional-multimodal.txt
```

Or install via package extras:

```bash
pip install ".[multimodal-sidecars]"
```

## Install Caveats

- WhisperX:
  optional and heavier than the current `faster-whisper` path; CPU use is
  possible but can be slow, and GPU is recommended for practical throughput
- pyannote.audio:
  optional; typically needs `ffmpeg`, a Hugging Face token, and accepted model
  access conditions before diarization can run
- OpenFace:
  optional and not managed as a Python dependency here; point the repo to an
  existing `FeatureExtraction` executable instead of assuming pip install
- FinBERT and emotion-english-distilroberta-base:
  optional model ids only; they reuse the existing `transformers` and `torch`
  setup already present in the repo
- MAEC / MELD / RAVDESS:
  not downloaded automatically; this step only defines config placeholders

## Environment Variables

### Hugging Face / model access

- `EARNINGS_CALL_HF_TOKEN`:
  preferred project-local token override for gated Hugging Face access
- `HF_TOKEN`:
  standard Hugging Face token fallback
- `HUGGINGFACE_HUB_TOKEN`:
  second standard fallback
- `HF_HOME`:
  optional Hugging Face cache root
- `TRANSFORMERS_CACHE`:
  optional Transformers cache directory
- `EARNINGS_CALL_MODEL_CACHE_DIR`:
  optional project-specific cache directory for future multimodal assets

### WhisperX / pyannote scaffolding

- `EARNINGS_CALL_WHISPERX_ENABLED`:
  `0` or `1`; default `0`
- `EARNINGS_CALL_WHISPERX_DIARIZATION_ENABLED`:
  `0` or `1`; default `0`
- `EARNINGS_CALL_PYANNOTE_ENABLED`:
  `0` or `1`; default `0`
- `EARNINGS_CALL_PYANNOTE_MODEL`:
  default `pyannote/speaker-diarization-community-1`
- `EARNINGS_CALL_MULTIMODAL_DEVICE`:
  default `cpu`; may be set to `cuda` later if you explicitly want GPU usage

### OpenFace scaffolding

- `EARNINGS_CALL_OPENFACE_ENABLED`:
  `0` or `1`; default `0`
- `EARNINGS_CALL_OPENFACE_BIN`:
  explicit path to the `FeatureExtraction` executable
- `EARNINGS_CALL_OPENFACE_ROOT`:
  optional directory containing `FeatureExtraction`
- `EARNINGS_CALL_OPENFACE_WORK_DIR`:
  optional scratch/output directory for future OpenFace sidecar calls

If both `EARNINGS_CALL_OPENFACE_BIN` and `EARNINGS_CALL_OPENFACE_ROOT` are set,
the explicit binary path wins.

### Optional NLP assist model ids

- `EARNINGS_CALL_FINBERT_MODEL`:
  default `ProsusAI/finbert`
- `EARNINGS_CALL_EMOTION_MODEL`:
  default `j-hartmann/emotion-english-distilroberta-base`

### External dataset root placeholders

- `EARNINGS_CALL_MAEC_ROOT`
- `EARNINGS_CALL_MELD_ROOT`
- `EARNINGS_CALL_RAVDESS_ROOT`

These are placeholders only. Set them only if you already have the datasets
available locally and are intentionally wiring a future ingestion step.

## What Is Intentionally Not Automated

- no automatic dataset downloads
- no automatic license acceptance or token creation
- no credentials written into files
- no OpenFace installation automation
- no WhisperX / pyannote pipeline wiring into the current CLI path
- no training code
- no changes to transcript-first deterministic outputs

## Alignment Sidecars

The repo now includes optional sidecar scripts for timestamp alignment and
experimental speaker-boundary assistance. These scripts do not replace the
default transcription path and do not rewrite `transcript.json` or
`transcript.txt`.

Default sidecar output root:

```text
data/processed/multimodal/alignment/<source_id>/
```

Artifacts written there include:

- `aligned_transcript.json`
- `aligned_segments.csv`
- `aligned_words.csv`
- `alignment_summary.json`
- `diarization_segments.csv` when diarization is explicitly enabled and applied
- `segment_candidates.csv` if you run the candidate-conversion helper

### Basic alignment using existing transcript artifacts

Align against an existing `transcript.json` from the current repo pipeline:

```bash
PYTHONPATH=src python scripts/run_whisperx_alignment.py \
  --source-id MSFT_2026_Q2_call05 \
  --audio-path cache/MSFT_2026_Q2_call05/audio_normalized.wav \
  --transcript-path outputs/MSFT_2026_Q2_call05/transcript.json \
  --language en
```

Align against a plain-text transcript:

```bash
PYTHONPATH=src python scripts/run_whisperx_alignment.py \
  --source-id EXAMPLE_CALL \
  --audio-path path/to/audio.wav \
  --transcript-path path/to/transcript.txt \
  --language en
```

If no transcript is provided, the sidecar can let WhisperX produce its own
alignment-oriented transcript output without affecting the repo's default
pipeline:

```bash
PYTHONPATH=src python scripts/run_whisperx_alignment.py \
  --source-id EXAMPLE_CALL \
  --audio-path path/to/audio.wav \
  --language en \
  --device cpu \
  --model small \
  --compute-type int8
```

### Experimental diarization

Diarization stays optional and is disabled unless you explicitly:

1. set `EARNINGS_CALL_PYANNOTE_ENABLED=1`
2. configure a Hugging Face token env
3. pass `--enable-diarization`

Example:

```bash
export EARNINGS_CALL_PYANNOTE_ENABLED=1
export EARNINGS_CALL_HF_TOKEN=your_token_here

PYTHONPATH=src python scripts/run_whisperx_alignment.py \
  --source-id EXAMPLE_CALL \
  --audio-path path/to/audio.wav \
  --transcript-path path/to/transcript.json \
  --language en \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 4
```

This adds supporting speaker labels where overlap is available, but it remains a
review aid rather than a source of truth.

### Candidate segment helper

Convert an alignment sidecar into simple candidate rows for later manual review:

```bash
PYTHONPATH=src python scripts/alignment_to_segment_candidates.py \
  --alignment-json data/processed/multimodal/alignment/EXAMPLE_CALL/aligned_transcript.json
```

The helper writes `segment_candidates.csv` next to the alignment JSON by
default.

## OpenFace Visual Sidecars

OpenFace support remains optional and external. These hooks generate only
conservative low-level visual summaries and do not produce psychological,
confidence, stress, deception, or truthfulness labels.

Default sidecar output root:

```text
data/processed/multimodal/visual/<source_id>/
```

Artifacts written there include:

- `segment_visual_features.csv`
- `segment_visual_features.json`
- `visual_coverage_summary.json`
- `openface_raw/<video_basename>.csv` when extraction succeeds

### Run OpenFace feature extraction

```bash
PYTHONPATH=src python scripts/run_openface_features.py \
  --source-id EXAMPLE_CALL \
  --video-path path/to/local_video.mp4 \
  --openface-bin /path/to/FeatureExtraction
```

The script reads source and segment manifests, processes only rows where
`face_expected=true`, and writes conservative per-segment features such as:

- `face_detection_rate`
- `frames_with_face`
- `mean_head_pose_change`
- `gaze_variability_proxy`
- `blink_or_eye_closure_proxy` when supported
- `au_intensity_mean` when supported
- `segment_visual_usability`
- `extraction_errors`

If OpenFace is missing, the script fails clearly and writes no fake outputs.

### Summarize visual coverage

```bash
PYTHONPATH=src python scripts/summarize_visual_coverage.py
```

This summarizes:

- attempted segments
- usable vs unusable
- extraction success rate
- source-group coverage

## NLP Assist Sidecars

The repo now includes an optional transcript-side NLP assist layer for segment
comparison only. It does not replace `sentiment_segments.csv`, deterministic
behavior outputs, or any final review labels.

Default sidecar output root:

```text
data/processed/multimodal/nlp/<source_id>/
```

Artifacts written there include:

- `nlp_segment_scores.csv`
- `nlp_segment_scores.json`
- `nlp_scoring_summary.json`
- `nlp_disagreement_summary.json` if you run the comparison helper

### Guardrails

- deterministic outputs remain the source of truth
- FinBERT is the default primary sidecar scorer
- the generic emotion model is optional and supporting only
- no black-box final decision layer is added here
- no benchmark or gold-label assets are modified

### Run FinBERT over transcript chunks

```bash
PYTHONPATH=src python scripts/run_nlp_segment_scoring.py \
  --source-id MSFT_2026_Q2_call05 \
  --chunks-csv outputs/MSFT_2026_Q2_call05/chunks_scored.csv
```

### Add the optional generic emotion pass

```bash
PYTHONPATH=src python scripts/run_nlp_segment_scoring.py \
  --source-id MSFT_2026_Q2_call05 \
  --chunks-csv outputs/MSFT_2026_Q2_call05/chunks_scored.csv \
  --run-secondary-emotion
```

### Score prepared remarks or Q&A answers only

```bash
PYTHONPATH=src python scripts/run_nlp_segment_scoring.py \
  --source-id MSFT_2026_Q2_call05 \
  --chunks-csv outputs/MSFT_2026_Q2_call05/chunks_scored.csv \
  --chunk-types prepared_remarks q_and_a_answer
```

### Score segment-manifest rows using a timed transcript JSON

```bash
PYTHONPATH=src python scripts/run_nlp_segment_scoring.py \
  --source-id EXAMPLE_CALL \
  --segment-manifest data/source_manifests/earnings_call_segments.csv \
  --transcript-path outputs/EXAMPLE_CALL/transcript.json
```

For manifest-driven scoring, the safest path is a JSON transcript with segment
timings. Placeholder `transcript_ref` notes are skipped rather than scored as
fake text.

### Summarize disagreement against deterministic artifacts

```bash
PYTHONPATH=src python scripts/summarize_nlp_disagreement.py \
  --source-id MSFT_2026_Q2_call05 \
  --deterministic-out-dir outputs/MSFT_2026_Q2_call05
```

This comparison summary is intended for inspection only. It reports side-by-side
label counts and disagreement examples without changing the existing
deterministic outputs.

## Suggested Minimal Setup

If you want conservative scaffolding only:

1. Install the base repo normally.
2. Optionally install `requirements-optional-multimodal.txt`.
3. Set env vars only for the tools you actually plan to test.
4. Leave all multimodal flags disabled unless you are explicitly experimenting.
