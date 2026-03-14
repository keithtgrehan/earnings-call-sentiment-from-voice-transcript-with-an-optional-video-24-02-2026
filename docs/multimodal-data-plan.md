# Multimodal Data Plan

## Product Guardrails
This repo stays transcript-first.

Priority order:
1. transcript + guidance + text behavior + Q&A shift
2. audio delivery support
3. optional visual support
4. late fusion that only adjusts interpretation confidence modestly

This is not:
- a deception detector
- a truth detector
- an emotion-product
- a trading or alpha system

Audio and video should help reviewers inspect delivery under questioning. They should never replace the transcript evidence or force a conclusion when media quality is weak.

## Current Repo Stance
- Transcript artifacts remain canonical and benchmarked.
- Audio and visual outputs are support layers only.
- Weak media should be suppressed, not stretched into a confident read.
- Generic emotion datasets are calibration resources, not earnings-call ground truth.

## Download Status In This Repo
- Downloaded in-repo for this pass: none
- Linked and documented for acquisition planning: yes
- Immediate external-dataset use: feature sanity-checking and threshold calibration only
- Repo-native earnings-call media set: active and committed under `data/media_support_eval/`
- Training a finance-specific support model: active for narrow support tasks using repo-native engineered features

## Current Repo-Native Media Support Set
- Source call groups currently represented in the committed labels: 6
- Total labeled segments: 70
- Audio-labeled segments: 52
- Video-labeled segments: 18
- Visual-tension-labeled video rows: 12 across 2 source groups
- Current support tasks:
  - hesitation pressure
  - delivery confidence support
  - visual tension support

This seed set is intentionally small and conservative. It is used for:
- validating repo-native feature extraction
- training narrow support models where label coverage is sufficient
- calibrating multimodal confidence adjustments without changing transcript-first benchmark labels

It is not large enough to justify broad product claims, and weak media should still suppress or downweight support outputs.
The current visual set is enough for a basic two-group grouped check, but it is
still one independent group short of a more defensible visual evaluation target.

## Priority Dataset Table

| Dataset | Modality | Official access | Current access status | Likely usefulness here | Main risks / domain mismatch | Planned use |
| --- | --- | --- | --- | --- | --- | --- |
| [RAVDESS](https://smartlaboratory.org/resources/speech-song-database-ravdess/) | audio + AV + video-only | SMART Lab page links to Zenodo download | Open download for research; non-commercial license; commercial license sold separately | Good for pause, loudness, and coarse delivery-feature sanity checks across audio and AV clips | Acted emotion corpus, not earnings calls; short scripted utterances; poor match for investor Q&A pressure | Feature sanity-checking only |
| [RAVDESS Facial Landmark Tracking](https://zenodo.org/doi/10.5281/zenodo.3255102) | landmark-derived face tracks | Zenodo | Open download | Useful for validating face-motion, head-motion, and landmark aggregation code without building trackers from scratch | Still acted and studio-recorded; not webcast framing; not speaker-pressure ground truth | Visual feature sanity-checking only |
| [IEMOCAP](https://sail.usc.edu/iemocap/index.html) | audio + video + motion capture + transcript | USC SAIL release flow | Gated / agreement-based release via USC | Strong candidate for multimodal turn-taking, answer-latency, and delivery-under-interaction prototypes | Acted dyadic interactions, not management/analyst calls; licensing friction; motion-capture setting unlike webcast video | Validation and limited calibration only |
| [AffectNet / AffectNet+](https://www.mohammadmahoor.com/pages/databases/affectnet/) | face images with expression/valence/arousal labels | AffectNet page / AffectNet+ request flow | Research-only request path; AffectNet is distributed as part of AffectNet+ | Good for stress-testing face-quality gates, in-the-wild landmark robustness, and weak-label visual proxy experiments | Static image dataset; not video, not earnings calls, not answer-pressure behavior | Feature robustness checks only |
| [RAF-DB](https://www.whdeng.cn/RAF/model1.html) | face images + landmarks + attributes | RAF-DB official page | Non-commercial research only; email request and password required | Good for verifying face-size, pose, and expression-landmark pipelines on unconstrained images | Static image dataset; not video, not finance; access is gated and usage restricted | Feature robustness checks only |

## What The Official Sources Say
- RAVDESS: the SMART Lab page says it can be downloaded free of charge from Zenodo and is released under a non-commercial Creative Commons license, with separate commercial licensing available.
- RAVDESS Facial Landmark Tracking: the Zenodo project page marks it as open and links it directly to the underlying RAVDESS material.
- IEMOCAP: the USC SAIL site describes it as an audiovisual and motion-capture corpus with transcripts, while the release flow is agreement-based rather than one-click open download.
- AffectNet: the official page says it is released for research purposes only and is currently requested through the AffectNet+ path.
- RAF-DB: the official page says it is for non-commercial research only and requires an institutional email plus password request before download.

## Project-Specific Recommendation

### Use now
- repo-native earnings-call runs for transcript-aligned audio/video thresholds and narrow support-model training
- RAVDESS for basic audio and face-pipeline sanity checks
- RAVDESS landmark tracking for visual aggregation smoke tests

### Use next, if access is approved
- IEMOCAP for interaction-aware audio/video feature validation
- RAF-DB and AffectNet for visual gate robustness, not product labels

### Do not use as product truth
- emotion classes from any of the above datasets
- acted affect labels as a direct proxy for management honesty, confidence, or earnings-call quality
- any dataset split as a stand-in benchmark for guidance labels

## Planned Evaluation Roles

| Role | Recommended source | Why |
| --- | --- | --- |
| Audio feature sanity-check | RAVDESS + repo-native clips | Easy access, clean recordings, useful for pause/prosody feature validation |
| Visual pipeline sanity-check | RAVDESS landmark tracking + RAF-DB / AffectNet | Helps test face visibility, landmark stability, and motion aggregation under different image conditions |
| Interaction / turn-taking validation | IEMOCAP | Better fit for question/answer contrast and response-latency experimentation |
| Product-level validation | repo-native earnings-call set only | Only source that actually matches the review task and benchmark contract |

## Modeling Plan

### Phase 1
- deterministic engineered features only
- no training required
- strict quality gates
- transcript-first late fusion

### Phase 2
Now active in a narrow, optional support role:
- logistic regression / LightGBM / XGBoost for narrow support tasks
- hesitation under pressure
- delivery confidence support
- visual tension under questioning
- calibrated probabilities only if they improve reviewer trust and stay interpretable

Current repo stance:
- lightweight logistic models are acceptable for support scoring
- transcript remains primary even when model-backed media support is available
- if model artifacts or optional runtimes are unavailable, the app must fall back to deterministic heuristics
- visual model training should stay deferred when group coverage is too small to validate responsibly

### Explicitly deferred
- giant end-to-end multimodal transformers
- generic seven-emotion classifier as the main product signal
- any model that turns weak media into a forced decision

## Practical Next Steps
1. Add at least one more independently labeled visual source group before treating grouped visual evaluation as defensible.
2. Increase the number of downstream cases that have real source-level media-support targets instead of transcript-first packaging only.
3. Use RAVDESS and RAF/AffectNet-style resources only to sanity-check feature extraction and gating thresholds.
4. Improve calibration only after the repo-native visual eval set clears the three-group minimum and the downstream comparison pack has broader support-target coverage.
