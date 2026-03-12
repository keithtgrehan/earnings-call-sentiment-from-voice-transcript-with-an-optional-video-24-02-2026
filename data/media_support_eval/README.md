# Media Support Eval

This package is a small repo-native seed set for calibrating the earnings-call
audio and video support layers.

It is intentionally conservative:
- transcript remains the primary evidence layer
- audio/video labels are reviewer-support labels, not truth or deception labels
- the current set is strong enough for an initial audio model pass
- the visual set is still best treated as calibration data until more unique
  webcast sources are added

## Current Contents
- `media_manifest.csv`: local media and feature-artifact sources already present in this repo
- `segment_labels.csv`: seed labels for audio hesitation, delivery confidence,
  visual tension, media quality, and multimodal support direction

## Scope Notes
- Sources are real repo-local earnings-call runs only.
- No gated external datasets were downloaded for this package.
- The current visual labels come from one usable repo-local webcast source,
  so visual model training may be skipped when the trainer enforces minimum
  source diversity.

## Label Meaning
- `media_quality_label`: whether the segment quality is poor, usable, or strong
- `hesitation_pressure_label`: low / medium / high audio hesitation under questioning
- `visual_tension_label`: low / medium / high visible tension or motion pressure
- `delivery_confidence_label`: low / medium / high delivery confidence support
- `multimodal_support_direction`: supportive / cautionary / neutral / unavailable

These labels are for reviewer support only and should not be interpreted as
emotion truth, deception detection, or trading recommendation signals.
