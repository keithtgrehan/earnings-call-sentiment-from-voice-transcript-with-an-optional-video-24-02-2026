# Evidence Map

This map ties the main public claims on `main` to exact repo-local evidence. It is intentionally conservative:

- `demonstrated`: directly visible in committed code or artifacts on `main`
- `partial`: real evidence exists, but coverage is narrow, secondary, or quality-gated
- `unproven`: the repo explicitly does not establish this claim

## Transcript-First Deterministic Core

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| Transcript-backed deterministic outputs remain the canonical review truth. | [docs/freeze-boundaries.md](freeze-boundaries.md), [README.md](../README.md), [report.md](../outputs/MSFT_2026_Q2_call05/report.md), [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json), [transcript.txt](../outputs/MSFT_2026_Q2_call05/transcript.txt) | demonstrated | The boundary doc and committed MSFT bundle both keep transcript-backed artifacts primary and sidecars separate. |
| Reviewer-facing outputs are evidence-backed rather than free-form only. | [report.md](../outputs/MSFT_2026_Q2_call05/report.md), [metrics.json](../outputs/MSFT_2026_Q2_call05/metrics.json) | demonstrated | The report surfaces ranked categories and strongest snippets; metrics keep the structured counts and scores. |

## Frozen Benchmark Assets

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| A frozen 9-call benchmark with canonical labels is committed on `main`. | [labels.csv](../data/gold_guidance_calls/labels.csv), [call_manifest.csv](../data/gold_guidance_calls/call_manifest.csv) | demonstrated | `labels.csv` provides the canonical label rows; `call_manifest.csv` preserves source metadata and provenance notes. |
| Benchmark labeling is intentionally conservative. | [labels.csv](../data/gold_guidance_calls/labels.csv), [README.md](../README.md) | demonstrated | Several rows, including `call05`, keep the label at `unclear` when explicit forward guidance exists but no explicit change verb is present. |

## Current Multimodal Breadth

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| Multimodal sidecars exist on `main`, but coverage is still partial. | [multimodal_eval_summary.json](../data/processed/multimodal/eval/multimodal_eval_summary.json), [current-status.md](current-status.md) | partial | The committed breadth summary shows `4` visual sources, `1` NLP source, and `0` alignment sources. |
| The visual path is real but narrow and quality-gated. | [curated_clip_run_status.json](../data/processed/multimodal/visual/curated_clip_run_status.json), [segment_visual_features.json](../data/processed/multimodal/visual/goog_q1_2025_example/segment_visual_features.json) | partial | The curated clip workflow is committed, but it uses approximate manual runtime-check clips and only `1` visually usable segment is recorded in the rolled-up summary. |

## Prototype Review Evidence

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| The repo now contains real committed prototype review rows for multimodal artifact review. | [multimodal_review_results_codex_proto.csv](../data/media_support_eval/multimodal_review_results_codex_proto.csv), [multimodal_review_summary.json](../outputs/media_support_eval/multimodal_review_summary.json) | partial | There are `4` committed Codex prototype rows and a saved descriptive summary. This is real repo evidence, but it is still a small prototype slice. |
| Prototype review results are descriptive only, not human-subject validation. | [multimodal_review_summary.json](../outputs/media_support_eval/multimodal_review_summary.json), [multimodal-review-eval-plan.md](multimodal-review-eval-plan.md) | partial | The saved summary explicitly says it should be used descriptively and not for statistical-significance claims. |

## Restored NLP Evidence

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| `main` now includes a restored committed NLP sidecar example. | [nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json), [multimodal_eval_summary.json](../data/processed/multimodal/eval/multimodal_eval_summary.json) | demonstrated | The Microsoft example writes `561` rows and the rolled-up summary records `1` source with NLP support. |
| NLP outputs are supporting evidence only. | [nlp_scoring_summary.json](../data/processed/multimodal/nlp/msft_fy26_q2_example/nlp_scoring_summary.json), [multimodal_support_summary.json](../outputs/MSFT_2026_Q2_call05/multimodal_support_summary.json) | partial | The committed NLP notes explicitly keep deterministic labels as the source of truth and treat the sidecar as inspection-only support. |

## Explicit Non-Claims

| Claim | Exact supporting file(s) | Status | Notes |
| --- | --- | --- | --- |
| Statistical significance is not established here. | [README.md](../README.md), [evaluation-summary.md](evaluation-summary.md), [multimodal_review_summary.json](../outputs/media_support_eval/multimodal_review_summary.json) | unproven | Both the public docs and the saved prototype summary explicitly say the current evidence is descriptive only. |
| The repo does not demonstrate trading edge or predictive lift. | [README.md](../README.md), [current-status.md](current-status.md), [evaluation-summary.md](evaluation-summary.md) | unproven | The project is framed as a review tool, not a validated return-prediction system. |
| The current repo state does not demonstrate meaningful alignment coverage. | [multimodal_eval_summary.json](../data/processed/multimodal/eval/multimodal_eval_summary.json), [current-status.md](current-status.md) | unproven | The committed breadth summary still shows `0` sources with alignment artifacts and `0` audio-aligned segments. |
