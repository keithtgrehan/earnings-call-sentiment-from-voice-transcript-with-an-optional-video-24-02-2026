# Microsoft Ambiguity Explainer

## Why This Case Matters

This is the strongest current Microsoft demo case on `main` because it shows both sides of the product value:

- the transcript contains explicit forward-looking revenue language that the deterministic layer can surface quickly
- the same case also contains a hard Q&A moment where management answers indirectly, and the system stays conservative instead of inventing a stronger conclusion

This makes it a good demo for explaining why the frozen label remains `unclear` even when the case has a rich supporting bundle.

## Which Microsoft Call Is This?

Using only repo-local evidence already on `main`, this bundle corresponds to:

- company: Microsoft
- ticker: `MSFT`
- reporting quarter: `Q2_2026`
- approximate call date: `2026-01-28`

Why this mapping is correct:

- the bundle path itself is [outputs/MSFT_2026_Q2_call05/report.md](../outputs/MSFT_2026_Q2_call05/report.md)
- the frozen benchmark row for `call05` in [labels.csv](../data/gold_guidance_calls/labels.csv) identifies `MSFT`, `Microsoft`, `Q2_2026`, and `2026-01-28`
- the benchmark manifest row for `call05` in [call_manifest.csv](../data/gold_guidance_calls/call_manifest.csv) identifies the title as `$MSFT Microsoft Q2 2026 Earnings Conference Call`
- the raw transcript file path named in the frozen label row is [MSFT_2026_Q2_call05.txt](../data/gold_guidance_calls/raw_calls/MSFT_2026_Q2_call05.txt)
- the transcript itself opens with “Microsoft Fiscal Year 206 second quarter earnings conference call,” which contains an obvious ASR typo but still supports the same quarter mapping when read alongside the file names and manifest rows

The only real ambiguity here is the transcript typo `206` instead of `2026`. The path names and frozen rows resolve that cleanly.

## Clear Signal Vs Ambiguous Signal

| Area | Clear signal | Still ambiguous | Exact evidence |
| --- | --- | --- | --- |
| Call identity | This is Microsoft `call05`, quarter `Q2_2026`, dated `2026-01-28` in the frozen benchmark set. | The transcript intro has an ASR typo (`206`). | [call_manifest.csv](../data/gold_guidance_calls/call_manifest.csv), [labels.csv](../data/gold_guidance_calls/labels.csv), [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt) |
| Outlook language | The transcript explicitly states total-company and segment-level revenue expectations. | The transcript does not itself say whether guidance was raised, maintained, lowered, or withdrawn versus prior guidance. | [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt), [report.md](../outputs/MSFT_2026_Q2_call05/report.md) |
| Deterministic extraction | The report and metrics pull the strongest outlook snippets and score uncertainty, guidance strength, and management confidence. | The guidance-revision block stays empty and no prior-guidance comparison path is present. | [report.md](../outputs/MSFT_2026_Q2_call05/report.md), [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json) |
| Hard Q&A moment | The analyst asks directly about OpenAI backlog durability and exposure. | Management answers indirectly and the system does not claim that indirectness resolves guidance direction. | [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt), [report.md](../outputs/MSFT_2026_Q2_call05/report.md) |
| Supporting sidecars | Audio is usable and supportive; NLP sidecar coverage exists for this same case. | Sidecars stay secondary and do not override the deterministic label path. Video remains unavailable in the committed roll-up. | [audio_behavior_summary.json](../outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json), [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json), [nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json) |

## Evidence Pull

### Excerpt A: Explicit Outlook

Transcript excerpt:

> Starting with the total company. We expect revenue of 80.65 to 81.75 billion US dollars are growth of 15 to 17%.

Exact transcript locator:

- file: [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt)
- local locator: lines `282-283`
- matching frozen benchmark quote: [labels.csv](../data/gold_guidance_calls/labels.csv) lines `15-16`

Matching structured artifacts:

- [report.md](../outputs/MSFT_2026_Q2_call05/report.md)
  - `Guidance Strength` strongest evidence and top guidance rows
- [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
  - `guidance.row_count = 147`
  - `guidance_strength_score = 4`
  - `guidance_revision.prior_guidance_path = null`
  - all revision counts remain `0`
- [labels.csv](../data/gold_guidance_calls/labels.csv)
  - frozen label remains `unclear`

What is clear:

- Microsoft gives explicit forward-looking revenue guidance
- the deterministic layer finds that quickly and surfaces it in structured form

What remains ambiguous:

- the current committed bundle does not contain a prior-guidance comparison for this case
- explicit outlook is not the same thing as explicit raised, maintained, lowered, or withdrawn guidance

### Excerpt B: Indirect Q&A Answer

Transcript excerpt:

> Thanks Amy on 45% of the backlog being related to OpenAI. I'm curious if you can comment. There's obviously concern about the durability
>
> I think maybe I would have thought about the question quite differently.
>
> if you're asking about how do I feel about OpenAI and the contract and the health, listen, it's a great partnership. We continue to be their provider of scale.

Exact transcript locator:

- file: [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt)
- local locator: lines `432-445`
- segment type: Q&A

Matching structured artifacts:

- [report.md](../outputs/MSFT_2026_Q2_call05/report.md)
  - `Management Confidence = 2/10`
  - `Uncertainty / Hedging = 9/10`
  - `Q&A Shift = mixed`
  - `Answer Directness = 5/10` with neutral-by-design wording
- [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
  - `management_confidence_score = 2`
  - `uncertainty_score = 9`
  - `qa_pressure_shift_score = 5`
  - `answer_directness_score = 5`
- [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
  - transcript stays primary
  - audio is supportive
  - video is unavailable

What is clear:

- the analyst asks a direct durability-and-exposure question
- the answer pivots toward diversification and partnership framing

What remains ambiguous:

- the answer is indirect, but the committed artifacts do not justify converting that into a stronger guidance-direction label
- the system records ambiguity instead of pretending the answer settles the case

## Why Still Unclear?

The frozen label still remains `unclear` because the repo-local evidence supports this narrower conclusion:

1. Explicit outlook exists.
   The transcript clearly includes forward-looking revenue ranges for the company and segments.
2. Guidance-change direction is still not explicit.
   The deterministic bundle does not contain a prior-guidance comparison path for this case.
3. The structured deterministic artifacts say the same thing.
   The report shows guidance language, but the guidance-revision section is empty and the metrics file keeps `prior_guidance_path` at `null`.
4. The frozen benchmark row uses the same rule.
   The label note says no explicit action verb indicates raised, maintained, lowered, or withdrawn versus prior guidance.
5. Supporting sidecars do not change the canonical truth source.
   Audio and NLP can add context, but they do not create a guidance-change action verb that is missing from the transcript evidence.
6. The committed prototype review slice reached the same conclusion.
   The `call05` row in [multimodal_review_results_codex_proto.csv](../data/media_support_eval/multimodal_review_results_codex_proto.csv) says the Microsoft package was the richest in the slice, but still stayed `unclear` because the sidecars remained secondary to the transcript-first decision.

This is why the current repo state is easier to defend as “explicit outlook, unresolved direction” than as any stronger benchmark claim.

## What Supporting Sidecars Add

Audio adds:

- usable answer-level support with `audio_quality_ok = true`
- supportive model-backed signal with `calibrated_support_score = -0.39`
- high prepared-remarks stability and low overall hesitation
- a small cue that Q&A becomes somewhat more paused, but still at a low level

NLP adds:

- `561` scored transcript chunks for this same Microsoft source
- an inspection layer over the same transcript bundle
- explicit notes that deterministic labels remain the source of truth

Multimodal roll-up adds:

- a compact record that transcript remains primary
- supportive audio as secondary context
- explicit confirmation that video is currently unavailable in the committed roll-up

What sidecars do not add:

- they do not establish a prior-guidance comparison
- they do not justify a stronger frozen label
- they do not turn this into proof of multimodal lift

## Video Requirements For This Case

Current state:

- the selected committed case bundle contains no usable case-specific video artifact in [outputs/MSFT_2026_Q2_call05](../outputs/MSFT_2026_Q2_call05)
- the current roll-up records `video_support_direction = unavailable` in [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
- the report records `video quality ok: False` in [report.md](../outputs/MSFT_2026_Q2_call05/report.md)

If a human later wants optional supporting video for this same Microsoft case, it should satisfy all of the following:

- exact call: the video must correspond to Microsoft `call05`, `Q2_2026`, event date `2026-01-28`
- strongest target moment: the Q&A exchange on OpenAI backlog durability and exposure in [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt) lines `432-445`
- target speaker: Amy Hood is the most likely primary target for that moment because the analyst addresses “Amy” directly in the question
- target segment type: Q&A
- preferred segment length:
  - `45-90` seconds if capturing both the question stem and the answer
  - `20-45` seconds if capturing answer-only
- secondary optional target: the prepared-remarks total-company outlook in [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt) lines `282-283`
- secondary segment type: prepared remarks
- secondary preferred length: `15-30` seconds

Minimum acceptable visual quality:

- face image should be clear enough to inspect gross delivery cues such as stability, visible pauses, and obvious hesitation behavior
- heavy blockiness, severe compression smearing, or large overlays across the face should count as not good enough
- stable speaker framing is preferred over dynamic edit-heavy presentation

Minimum acceptable face visibility:

- target speaker face visible for most of the selected segment
- near-frontal or readable three-quarter angle is acceptable
- face should not be tiny, mostly off-screen, or repeatedly replaced by slides

Acceptable source types:

- official webcast replay from Microsoft investor-relations materials
- clearly attributable repost of the same webcast, as long as the call identity and moment match the frozen benchmark case

Unacceptable source types:

- commentary clips
- unrelated interviews
- montage or highlight edits
- audio-only placeholders
- clips with heavy cuts, reactions, or non-call overlays that break moment continuity

## Candidate Video Usability Checklist

Use this as a future yes/no screen for any candidate Microsoft video:

- same company: Microsoft
- same quarter/call: `Q2_2026` / `call05`
- same approximate call date: `2026-01-28`
- same speaker moment as the transcript excerpt being demonstrated
- face visible for most of the segment
- stable framing with no major visual interruptions
- sufficient duration to preserve the question or answer context
- audio matches the transcript moment closely enough to align the clip
- replay is attributable to the same earnings call
- segment can be isolated cleanly without heavy cuts or overlays
- useful enough to add supporting evidence beyond the transcript alone

Decision rules:

- `usable`: all identity checks are yes, audio matches the transcript moment, the face is visible for most of the segment, and the clip is clean enough to function as supporting evidence
- `borderline`: identity checks are yes, but one quality condition is weak, for example intermittent face visibility or a short but still interpretable segment
- `not usable`: any identity mismatch, no reliable transcript/audio match, no meaningful face visibility, or heavy editing/commentary contamination

## What They Do Not Prove

This case and its supporting layer do not prove:

- that the benchmark label should be changed
- statistical significance
- validated multimodal lift
- predictive edge or trading edge
- that usable video currently exists in the committed repo for this case

## Exact Repo File References

- [report.md](../outputs/MSFT_2026_Q2_call05/report.md)
- [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json)
- [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt)
- [audio_behavior_summary.json](../outputs/MSFT_2026_Q2_call05/audio_behavior_summary.json)
- [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json)
- [nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json)
- [labels.csv](../data/gold_guidance_calls/labels.csv)
- [call_manifest.csv](../data/gold_guidance_calls/call_manifest.csv)
- [MSFT_2026_Q2_call05.txt](../data/gold_guidance_calls/raw_calls/MSFT_2026_Q2_call05.txt)
- [multimodal_review_results_codex_proto.csv](../data/media_support_eval/multimodal_review_results_codex_proto.csv)
