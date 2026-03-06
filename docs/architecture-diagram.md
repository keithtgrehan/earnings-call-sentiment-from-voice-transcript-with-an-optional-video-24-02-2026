# Architecture Diagram

## Purpose
This architecture shows the high-level product flow of the Earnings Call Signal Extraction Engine from call input to user-facing output. It focuses on how raw call content is transformed into deterministic, auditable signals that help a retail trader review what changed. It also includes the improved flow step that compares current guidance with prior-call guidance.

## High-Level Flow
1. Ingest one earnings call (audio and/or transcript source).
2. Produce analyzable call text and segment it into usable units.
3. Run deterministic signal extraction over the call text.
4. Derive guidance statements, tone-change moments, and prior-call guidance comparisons.
5. Select transcript evidence snippets tied to each detected signal.
6. Generate a structured signal summary for retail trader review.

## Diagram (Mermaid)
```mermaid
flowchart TD
    A[Earnings Call Input\n(Audio / Transcript)] --> B[Transcript / Call Text]
    B --> C[Deterministic Signal Extraction Engine]
    C --> D[Guidance Extraction]
    C --> E[Tone-Change Detection]
    D --> F[Prior Call Guidance Comparison\n(Improved Version)]
    E --> G[Evidence Snippet Selection]
    F --> G
    G --> H[Structured Signal Summary]
    H --> I[Retail Trader Review]
```

## Component Notes
- Earnings Call Input: The single call selected for analysis.
- Transcript / Call Text: The standardized text used for downstream signal processing.
- Deterministic Signal Extraction Engine: Rule-based/scored logic that produces auditable outputs instead of opaque black-box decisions.
- Guidance Extraction: Identifies and structures guidance-related statements from management language.
- Tone-Change Detection: Flags notable changes in sentiment/tone across the call.
- Prior Call Guidance Comparison: Compares current guidance outputs to prior-call guidance to highlight meaningful revision direction.
- Evidence Snippet Selection: Attaches transcript snippets that justify each surfaced signal.
- Structured Signal Summary: Consolidated output that shows what changed, why it may matter, and where to look first.
- Retail Trader Review: Final decision-support step for a human user.

## Scope Notes
- This architecture includes baseline deterministic extraction and the improved comparison/tone-shift flow described in capstone scope.
- It intentionally excludes autonomous trading, portfolio recommendations, and forecasting claims.
- It focuses on logical data/product flow and evidence-backed summaries, not deployment/infrastructure design.
