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

## Suggested Minimal Setup

If you want conservative scaffolding only:

1. Install the base repo normally.
2. Optionally install `requirements-optional-multimodal.txt`.
3. Set env vars only for the tools you actually plan to test.
4. Leave all multimodal flags disabled unless you are explicitly experimenting.
