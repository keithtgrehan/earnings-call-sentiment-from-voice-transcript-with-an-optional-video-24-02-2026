# Earnings Call Sentiment Report

## Summary
- Chunks scored: 1166
- Sentiment mean: 0.365131
- Sentiment std: 0.877609

## Guidance
- Guidance rows: 192
- Mean guidance strength: 0.298241

| start | end | guidance_strength | text |
| --- | --- | --- | --- |
| 393.00 | 399.00 | 0.7387 | building on the blistering pace of 121% year-over-year in Q3 and 93% year-over-year growth in Q2, |
| 993.00 | 997.00 | 0.6656 | at 4.3 billion up 138% year over year. |
| 443.00 | 450.00 | 0.6626 | while an energy company expanded from $4 million ACV in Q1 2025 to over $20 million ACV by year-end, |
| 218.00 | 225.00 | 0.6098 | In Q4, overall revenue surged 70% year-over-year, our highest growth rate as a public company, |
| 945.00 | 950.00 | 0.5899 | representing a 51% margin and 82% growth year over year. |

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
- reassurance: low
- analyst skepticism: low

### Uncertainty evidence
- [modal uncertainty] strength=1: OS and like where else we could see
- [modal uncertainty] strength=1: again and I might look

### Management reassurance evidence
- _none_

### Analyst skepticism evidence
- _none_

## Q&A Shift
- prepared remarks vs Q&A: weaker
- analyst skepticism: low
- management answers vs prepared remarks uncertainty: more uncertain
- early vs late Q&A: weaker

### Q&A examples
- delta=+1.3525 | Q: are like well how would you shape the problem | A: no one's you know it's like
- delta=+1.2577 | Q: how do you in fact even justify | A: moving into something that's more complicated

## Audio Behavior Signals
- hesitation: low
- pauses before answers: low
- prepared remarks audio stability: low
- Q&A hesitation shift: high

- notable segment: q_and_a 2161.0s-2163.0s | hesitation=high | pause_ms=None
- caution: low transcript word count limits audio confidence

## Outputs
- guidance.csv
- guidance_revision.csv
- uncertainty_signals.csv
- reassurance_signals.csv
- analyst_skepticism.csv
- behavioral_summary.json
- qa_shift_segments.csv
- qa_shift_summary.json
- audio_behavior_segments.csv
- audio_behavior_summary.json
- metrics.json
- report.md
