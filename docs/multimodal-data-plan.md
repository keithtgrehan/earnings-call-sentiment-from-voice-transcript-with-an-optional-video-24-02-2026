# Multimodal Data Plan

## Purpose
This repo should stay transcript-first. Multimodal work should strengthen reviewer decision support, not replace the deterministic text backbone.

Priority order:
1. transcript + Q&A + behavior signals
2. audio hesitation / pause features
3. video behavior as optional support

This remains a decision-support system for structured earnings-call review. It is not a predictive trading system, not proof of alpha, not emotion-truth inference, and not lie detection.

## Current Readiness

### Text: usable now
- Transcript ingestion already supports YouTube, local media, document upload, and pasted text.
- The repo already produces:
  - guidance extraction and guidance revision artifacts
  - tone-change artifacts
  - behavior artifacts: uncertainty, reassurance, analyst skepticism
  - Q&A shift artifacts
- The guidance benchmark stack is already stable:
  - frozen benchmark: `data/gold_guidance_calls/labels.csv`
  - active holdout: `data/gold_guidance_calls_holdout/labels.csv`
  - watchlist-derived unseen holdout: `data/gold_guidance_calls_holdout_watchlist/labels.csv`

### Audio: partly ready, but not labeled yet
- Transcript chunks and scored segments already carry timing fields that can anchor answer-level audio windows.
- The CLI already normalizes audio and keeps enough run-level structure to add pause and hesitation extraction without redesigning the workflow.
- Q&A shift logic already separates prepared remarks from Q&A and identifies answer windows, which is the right attachment point for audio features.
- What is missing:
  - answer-level pause / onset latency extraction
  - filler density and hesitation labels
  - small audio gold eval set
  - overlap / interruption handling benchmarked on repo-native calls

### Video: now optional support, not core
- Video-capable runs can now emit observational visual behavior outputs.
- Current video layer is best treated as a support layer because source framing, face visibility, and webcast layouts vary widely.
- What is missing:
  - broader smoke coverage on real investor-webcast video sources
  - any finance-specific evaluation set for visual labels
  - stable speaker-on-camera mapping for mixed layouts

### Not worth doing yet
- emotion classification
- deception or intent inference
- large end-to-end multimodal models
- broad public dataset ingestion pipelines
- topic-specific tone modeling before audio answer-level basics are measured

## Intended Multimodal Stack

### 1. Text layer
Primary product layer.
- transcript
- guidance change
- guidance revision comparison
- uncertainty / hedging
- management reassurance
- analyst skepticism
- Q&A shift

Why it stays first:
- every benchmark already depends on transcript evidence
- text coverage is highest across earnings-call sources
- transcript outputs are already auditable and benchmarked

### 2. Audio layer
Next best expansion.
- pause length
- hesitation / filler markers
- speaking rate
- silence before answer
- answer onset latency
- interruption / overlap if supportable
- optional pitch / energy variability only if extraction is stable and cheap

Why audio comes second:
- audio is present even when video is absent or low quality
- hesitation and pause behavior are easier to define conservatively than facial interpretation
- answer-level timing can reuse existing transcript and Q&A windows

### 3. Video layer
Optional support layer only.
- face visibility
- motion / stability proxies
- head movement / gaze-change proxies
- optional posture / hand visibility when framing supports it

Why video stays third:
- many earnings-call videos are poorly framed, static, or presenter-switching
- webcast layouts often reduce face confidence
- visual outputs should support the reviewer, not drive the product

### 4. Fusion layer
Reviewer-facing summary layer, not a model layer.
- answer-level pressure response summary
- multimodal answer-stability review
- text-first, audio-second, video-third weighting

Interpretation rule:
- text evidence remains canonical
- audio can raise or lower reviewer attention on answer stability
- video can add observational context only when confidence is usable

## Labeling Plan By Modality

### Text labels
Already present or directly adjacent to current work.
- guidance change: `raised | maintained | lowered | withdrawn | unclear`
- uncertainty: `absent | present | strong`
- reassurance: `absent | present`
- skepticism: `low | medium | high`
- Q&A shift: summary labels already emitted by deterministic logic

### Audio labels
Planned next.
- hesitation: `low | medium | high`
- pause before answer: `low | medium | high`
- filler density: `low | medium | high`
- answer onset delay: `low | medium | high`
- interruption / overlap: `none | present | strong` only if extraction is supportable

### Video labels
Support-only.
- face visibility: `low | medium | high`
- visual stability: `low | medium | high`
- visual change during answer: `low | medium | high`
- visual confidence: `low | medium | high`

### Fusion labels
Planned after audio is usable.
- answer pressure response: `low | medium | high`
- multimodal change vs prepared remarks: `low | medium | high`

## Data Acquisition Priority

### Tier 1: repo-native earnings-call data
Best next use of time.
- existing full transcripts with Q&A and timing
- locally reproducible earnings-call sources already present in benchmark packages
- future calls with defensible transcript/audio/video paths

Target use:
- answer-level audio hesitation labels
- pause / latency labels
- multimodal answer review examples

### Tier 2: public multimodal calibration resources
Use only for sanity-checking feature design.
- AMI Meeting Corpus
  - modality: audio + video + turns
  - intended use: overlap / pause and turn-boundary sanity checks
  - limitation: meetings, not earnings calls
- NoXi
  - modality: audio + video + interaction
  - intended use: conversational shift and visible-attention proxy sanity checks
  - limitation: tutoring dialogue, not finance
- AVA-ActiveSpeaker
  - modality: video + speech activity
  - intended use: speaker-visible alignment sanity checks if webcast layouts become more complex
  - limitation: web video benchmark, not finance-specific behavior evidence

### Tier 3: optional calibration references
Use only if a specific feature is blocked.
- audio DSP baselines for pause / filler extraction
- open computer-vision references for face visibility and head-motion thresholds

These resources are support material only. They are not earnings-call benchmarks and do not justify finance-specific claims.

## Practical Next Implementation Path
1. add deterministic audio hesitation / pause extraction at answer level
2. create a small labeled audio eval set under a dedicated package
3. build an answer-level pressure-response summary that combines text + audio
4. keep video as optional support where source quality is good enough

## Decision Rules
- If a source has full transcript + clean Q&A + stable audio, prioritize it over a prettier video source.
- If a source has only a short excerpt, use it for candidate scouting, not for answer-level multimodal labeling.
- If video framing is weak, keep visual outputs secondary and confidence-tagged.
- Do not broaden into topic tone, emotion inference, or predictive claims until the audio layer is measured on a small internal eval set.
