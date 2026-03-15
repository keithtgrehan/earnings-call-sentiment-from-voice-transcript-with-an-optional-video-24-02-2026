# Audio Signal Eval

This package is a small staging area for future audio-behavior labeling.

It is not part of the frozen guidance benchmark, the active holdout benchmark,
or the watchlist-derived unseen holdout benchmark.

## Purpose

Use this package to build a small, auditable gold set for deterministic
audio-behavior signals such as:

- hesitation: `low | medium | high`
- pause_before_answer: `low | medium | high`
- filler_density: `low | medium | high`
- answer_onset_delay: `low | medium | high`
- interruption_overlap: `low | medium | high` if supportable later

## Scope

- source rows should come from repo-native earnings-call media where transcript
  timing and normalized audio both exist locally
- rows should be answer-level whenever possible
- labels should stay observational and timing-based
- do not infer emotion, deception, or hidden intent

## Files

- `source_manifest.csv`: candidate source calls with local transcript/audio paths
- `candidate_labels.csv`: header-only label file for future answer-level spans

## Current status

This package is scaffold-only for now. No audio gold labels have been appended.
