# April 2 Demo Side-By-Side

## Demo Case

Use [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md) as the strongest current reviewer/demo case on `main`.

For a deeper defense of why this case still lands on the frozen `unclear` label, see [msft-ambiguity-explainer.md](msft-ambiguity-explainer.md).

Why this case:

- Microsoft is recognizable to reviewers without extra context
- the committed bundle is the fullest current package on `main`: deterministic report, metrics, transcript, audio summary, multimodal roll-up, restored NLP sidecar, frozen benchmark label, and a committed prototype-review row
- it contains both a concrete outlook passage and a good example of management answering a hard question indirectly
- it lets the demo show both sides of the product value: extraction when the transcript is dense, and restraint when the answer is vague

Why it was chosen over the other committed prototype cases:

- [outputs/GOOGL_2025_Q4_call03/report.md](../outputs/GOOGL_2025_Q4_call03/report.md) and [outputs/IBM_2025_Q4_call04/report.md](../outputs/IBM_2025_Q4_call04/report.md) are usable, but neither has the same committed NLP sidecar support on `main`
- [outputs/PLTR_2025_Q4_call01/report.md](../outputs/PLTR_2025_Q4_call01/report.md) has a thinner committed bundle and no saved multimodal support summary in the selected minimal package
- Microsoft gives the cleanest end-to-end traceability path without asking the reviewer to cross too many files

## Excerpt 1: Outlook Language That Benefits From Structured Extraction

### Raw Transcript

From [outputs/MSFT_2026_Q2_call05/transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt):

> In productivity in business processes, we expect revenue of $34.25 to $34.55 billion or growth of $14 to 15%.
> In M365 commercial cloud, we expect revenue growth to be between 13 and 14% in constant currency with continued stability and year over year growth rates on a large and expanding base.
> Accelerating co-pilot momentum and ongoing E-5 adoption will again drive RPO growth.

### What The Raw Transcript Makes Hard

- the transcript is readable, but it is still dense, noisy, and surrounded by adjacent caveats about supply, contract mix, and volatility
- the reviewer has to manually separate the explicit outlook statement from the surrounding investor-relations framing
- the passage is clearly forward-looking, but it still does not answer the benchmark question of whether guidance was raised, maintained, or lowered versus prior guidance

### Matching Structured Outputs

- [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md)
  - top guidance row: `guidance_strength=1.0000`
  - strongest evidence snippet: `we expect revenue of $34.25 to $34.55 billion or growth of $14 to 15%`
- [outputs/MSFT_2026_Q2_call05/metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
  - `guidance.row_count = 147`
  - `guidance_strength_score = 4`
  - `overall_review_signal = amber`
- [data/gold_guidance_calls/labels.csv](../data/gold_guidance_calls/labels.csv)
  - `call05` remains `unclear`
  - frozen note: explicit revenue guidance exists, but no action verb establishes raised, maintained, lowered, or withdrawn versus prior guidance

### What The Structured Output Makes Easier

- it pulls the key outlook line to the top immediately instead of making the reviewer hunt through a long earnings-call transcript
- it turns the messy raw text into a traceable scorecard and a machine-readable metrics file
- it keeps the interpretation conservative: explicit guidance is present, but guidance-change direction still stays unresolved

### What Remains Ambiguous

- the structured package helps a reviewer find the outlook quickly
- it still does not prove a benchmark label stronger than `unclear`
- that restraint is part of the product value, not a failure mode

## Excerpt 2: Management Vagueness Under A Hard Q&A Prompt

### Raw Transcript

From [outputs/MSFT_2026_Q2_call05/transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt):

> Thanks Amy on 45% of the backlog being related to OpenAI. I'm curious if you can comment. There's obviously concern about the durability and I know maybe there's not much you can say in this but I think everyone's concerned about the exposure.
>
> I think maybe I would have thought about the question quite differently.
>
> And so then if you're asking about how do I feel about OpenAI and the contract and the health, listen, it's a great partnership. We continue to be their provider of scale. We're excited to do that.

### What The Raw Transcript Makes Hard

- the analyst asks a direct durability-and-exposure question
- the answer sounds calm and polished, but it shifts toward diversification and partnership language rather than directly addressing durability risk
- a human reviewer can feel the indirectness, but the transcript alone does not cleanly convert that into a benchmark label

### Matching Structured Outputs

- [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md)
  - `Uncertainty / Hedging = 9/10`
  - `Management Confidence = 2/10`
  - `Q&A Shift = mixed`
  - `Answer Directness = 5/10` with the explicit note that Q&A evidence stays neutral by design when evidence is insufficient
- [outputs/MSFT_2026_Q2_call05/metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
  - `uncertainty_score = 9`
  - `management_confidence_score = 2`
  - `review_confidence_pct = 55`
- [outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
  - transcript primary assessment remains `amber`
  - audio support is `supportive`
  - multimodal confidence adjustment is `-1`
- [outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json](../outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json)
  - hesitation stays `low`
  - audio confidence support is `high`
  - notes keep these signals observational rather than mental-state claims
- [data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json)
  - `561` transcript chunks scored
  - notes explicitly say deterministic labels remain the source of truth

### What The Structured Output Makes Easier

- it prevents the demo from equating polished delivery with a definitive answer
- it shows that even with supportive audio and committed NLP sidecar coverage, the primary interpretation stays transcript-first and conservative
- it turns a fuzzy “that answer felt vague” reaction into a traceable combination of high uncertainty, low management confidence, mixed Q&A shift, and unchanged benchmark label

### What Remains Ambiguous

- the answer may be strategically indirect rather than explicitly evasive
- the committed artifacts do not justify a stronger raised, maintained, or lowered guidance call
- the system therefore keeps ambiguity visible instead of hallucinating certainty

## Why Deterministic-First Matters

- the canonical review truth still lives in the transcript, the deterministic report, the structured metrics, and the frozen label files
- sidecars help the reviewer inspect delivery and disagreement, but they do not become the truth source
- this Microsoft case is useful precisely because the repo contains supportive audio and a real NLP sidecar, yet the final benchmark interpretation still stays conservative

## What Happens When Management Is Vague?

- the system does not force a confident answer where the underlying evidence is indirect
- explicit outlook language is surfaced cleanly, but the guidance-revision block stays empty and the frozen label remains `unclear`
- the hard OpenAI durability question is not treated as proof of a guidance change
- supportive sidecars do not override the transcript-first assessment

In other words: vagueness is surfaced as ambiguity, not hallucinated into certainty.

## What This Demonstrates / What It Does Not Prove

What this demonstrates:

- a messy raw transcript can be reduced to a small number of traceable reviewer artifacts quickly
- deterministic extraction makes the important outlook passage easy to find and easy to audit
- the system can stay restrained when management language is indirect
- sidecars can add context without replacing the transcript-first review path

What this does not prove:

- statistical significance
- validated multimodal lift
- production-ready multimodal coverage
- predictive edge or trading edge

## Exact Artifact Paths Used

- [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md)
- [outputs/MSFT_2026_Q2_call05/metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
- [outputs/MSFT_2026_Q2_call05/transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt)
- [outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json](../outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json)
- [outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
- [data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json)
- [data/gold_guidance_calls/labels.csv](../data/gold_guidance_calls/labels.csv)
- [data/media_support_eval/multimodal_review_results_codex_proto.csv](../data/media_support_eval/multimodal_review_results_codex_proto.csv)
- [outputs/media_support_eval/multimodal_review_summary.json](../outputs/media_support_eval/multimodal_review_summary.json)
