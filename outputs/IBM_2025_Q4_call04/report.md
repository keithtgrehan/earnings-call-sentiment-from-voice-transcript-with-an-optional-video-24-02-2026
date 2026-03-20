# Earnings Call Sentiment Report

## Summary
- Overall review signal: Amber
- Review confidence: 60%
- Chunks scored: 633
- Sentiment mean: 0.376198
- Sentiment std: 0.888333

## Reviewer Scorecard
- Confidence note: Confidence reflects how clear and well-supported the deterministic interpretation is, not investment conviction.

| Rank | Category | Score | Band | Explanation |
| --- | --- | --- | --- | --- |
| 1 | Uncertainty / Hedging | 9/10 | red | Management language includes repeated hedging or visibility caveats, which raises uncertainty. |
| 2 | Analyst Skepticism | 3/10 | green | Analyst questions look relatively routine, with limited skeptical pressure. |
| 3 | Guidance Strength | 4/10 | amber | Guidance language is present, but its direction is not explicit enough to score strongly. |
| 4 | Management Confidence | 4/10 | amber | Management confidence blends reassurance cues (medium) with uncertainty and Q&A follow-through. |
| 5 | Q&A Pressure Shift | 5/10 | amber | Insufficient Q&A contrast evidence was available, so pressure shift stays neutral. |
| 6 | Answer Directness | 5/10 | amber | Q&A evidence is unavailable, so answer directness stays neutral by design. |

### Strongest evidence snippets
#### Uncertainty / Hedging
- [modal uncertainty] As well, one of you could maybe outline the path or trajectory you're expecting for the consulting
- [modal uncertainty] Finally, some comments made in this presentation may be considered forward looking under

#### Analyst Skepticism
- _none_

#### Guidance Strength
- and $14.7 billion of free cash flow, growing 16% over last year. This represents our highest free
- robust quarter, growing 17% driven by strength and Z17, which has been outpacing Z16 performance.

#### Management Confidence
- [remain on track] We remain on track to deliver the first large

#### Q&A Pressure Shift
- _none_

#### Answer Directness
- _none_

## Guidance
- Guidance rows: 187
- Mean guidance strength: 0.255806

| start | end | guidance_strength | text |
| --- | --- | --- | --- |
| 852.44 | 862.12 | 0.6698 | and $14.7 billion of free cash flow, growing 16% over last year. This represents our highest free |
| 338.28 | 346.84 | 0.6674 | robust quarter, growing 17% driven by strength and Z17, which has been outpacing Z16 performance. |
| 4098.20 | 4106.36 | 0.6200 | into, I believe if I look out 3 to 5 years, 50% of the enterprise usage of AI is going to be |
| 1091.48 | 1099.88 | 0.6092 | growing more than 30%. And as we expected last quarter given the record Z-17 placement this year, |
| 111.84 | 118.48 | 0.5899 | Delivering 6% revenue growth, our highest level of revenue growth in many years, and 14.7 |

## Guidance Revisions (vs prior)
- Prior guidance: None
- Matched: 0
- Raised: 0
- Lowered: 0
- Reaffirmed: 0
- Unclear: 0
- Mixed: 0

_none_

## Tone & Behavioral Signals
- uncertainty: high
- reassurance: medium
- analyst skepticism: low

### Uncertainty evidence
- [modal uncertainty] strength=1: As well, one of you could maybe outline the path or trajectory you're expecting for the consulting
- [modal uncertainty] strength=1: Finally, some comments made in this presentation may be considered forward looking under

### Management reassurance evidence
- [remain on track] strength=3: We remain on track to deliver the first large

### Analyst skepticism evidence
- _none_

## Q&A Shift
- prepared remarks vs Q&A: stronger
- analyst skepticism: low
- management answers vs prepared remarks uncertainty: more uncertain
- early vs late Q&A: weaker

### Q&A examples
- delta=+1.5111 | Q: Olympia? | A: Thank you.
- delta=+1.4264 | Q: 16% and any percentage on the 15.7 numbers of this year? | A: Thanks, Simon. I appreciate the question. As we've been talking about for five years,

## Audio Behavior Signals
- hesitation: medium
- pauses before answers: low
- prepared remarks audio stability: low
- Q&A hesitation shift: medium
- answer latency pressure: low
- audio confidence support: high
- support mode: model_backed

- model-backed support: supportive | calibrated_score=-0.23
- notable segment: 334 2292.2s-2297.6s | hesitation=high | pause_ms=None | latency_ms=None
- caution: short segment limits audio confidence

## Media Quality
- audio quality ok: True
- video quality ok: False

- Audio and video quality gates did not raise additional caution notes.

## Multimodal Support
- transcript primary assessment: amber
- audio support direction: supportive
- video support direction: unavailable
- fusion mode: hybrid
- calibrated support score: -0.23
- multimodal alignment: medium
- multimodal confidence adjustment: -1

- Transcript-first signal remains amber.
- Audio support used model-backed scoring (supportive, calibrated score -0.23).
- Video support was suppressed because quality gates were not met.
- Q&A transcript shift stayed stronger.

## Outputs
- guidance.csv
- guidance_revision.csv
- uncertainty_signals.csv
- reassurance_signals.csv
- analyst_skepticism.csv
- behavioral_summary.json
- qa_shift_segments.csv
- qa_shift_summary.json
- metrics.json (includes review_scorecard)
- audio_behavior_segments.csv
- audio_behavior_summary.json
- media_quality.json
- multimodal_support_summary.json
- report.md
