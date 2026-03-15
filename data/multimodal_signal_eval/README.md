# Multimodal Signal Eval Planning Package

This package is a metadata-first staging area for the next multimodal acquisition and labeling cycle.

## Purpose
- keep multimodal planning separate from the frozen guidance benchmark and current holdout benchmarks
- define the next small acquisition shortlist for transcript + Q&A + behavior first, audio second, and video third
- avoid creating benchmark labels before the underlying modality is actually collected and supportable

## Current contents
- `acquisition_manifest.csv`: compact shortlist of the next calls / sources to use for multimodal work

## What this package is not
- not a gold benchmark
- not a holdout benchmark
- not a canonical label source
- not a dataset download area

## Intended next use
1. pull the highest-priority rows that already have full local transcripts and Q&A structure
2. add answer-level audio pause / hesitation extraction on those rows first
3. only then add small audio and fusion labeling files if the extracted windows are actually usable
4. treat video rows as optional support and confidence-gated

## Label families planned
### Audio
- hesitation: `low | medium | high`
- pause_before_answer: `low | medium | high`
- filler_density: `low | medium | high`
- answer_onset_delay: `low | medium | high`
- interruption_overlap: `none | present | strong`

### Video
- face_visibility: `low | medium | high`
- visual_stability: `low | medium | high`
- visual_change_during_answer: `low | medium | high`
- visual_confidence: `low | medium | high`

### Fusion
- answer_pressure_response: `low | medium | high`
- multimodal_change_vs_prepared: `low | medium | high`

No label CSVs are created here yet because the next required step is acquisition and feature extraction, not more empty scaffolding.
