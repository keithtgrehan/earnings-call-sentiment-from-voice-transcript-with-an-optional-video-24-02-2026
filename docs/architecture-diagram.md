# Architecture Diagram

## Purpose
This architecture shows how the Earnings Call Signal Extraction Engine converts one earnings call into a structured, evidence-backed signal summary for a retail trader. It communicates the current transcript-first capstone MVP, the improved-version comparison flow, and where a later audio layer may fit without changing the current scope.

## High-Level Flow
1. Ingest one earnings call input and derive transcript/call text as the analyzable source.
2. Run deterministic signal extraction to identify guidance statements and tone-change moments.
3. Compare current-call guidance against prior-call guidance when available.
4. Select supporting transcript evidence snippets for each major signal.
5. Produce a structured signal summary for retail trader review.

## Diagram (Mermaid)
```mermaid
flowchart TD
    A["Earnings Call Input"] --> B["Transcript / Call Text"]
    B --> C["Deterministic Signal Extraction"]
    C --> D["Guidance Extraction"]
    C --> E["Tone-Change Detection"]
    D --> F["Prior Call Comparison"]
    E --> G["Evidence Snippet Selection"]
    F --> G
    G --> H["Structured Signal Summary"]
    H --> I["Retail Trader Review"]

    J["Future Audio Layer (Later)"] -.-> K["Multimodal earnings-call audio cues"]
    K -.-> L["Speaking-rate change"]
    K -.-> M["Pause / hesitation detection"]
    K -.-> N["Emphasis / pitch variability"]
    K -.-> O["Alignment with transcript guidance statements"]
    K -.-> P["Speaker-aware financial-audio processing"]
    P -.-> Q["Cleaner Q&A analysis"]
    O -.-> H
    Q -.-> H
## Component Notes
- **Earnings Call Input**: A single call is the starting point for one deterministic analysis run.
- **Transcript / Call Text**: The current MVP is transcript-first and uses this text as the primary analysis input.
- **Deterministic Signal Extraction**: Applies consistent rules to produce auditable signal outputs.
- **Guidance Extraction**: Pulls management guidance-related statements for structured comparison.
- **Tone-Change Detection**: Flags meaningful shifts in language tone across the call timeline.
- **Prior Call Comparison**: Adds improved-version context by contrasting current guidance with prior guidance.
- **Evidence Snippet Selection**: Keeps the output traceable by attaching transcript excerpts.
- **Structured Signal Summary**: Consolidates key findings into a review-ready output.
- **Retail Trader Review**: Supports faster, evidence-backed interpretation by the target user.

## Future Audio Layer (Later Scope)
Future versions may incorporate multimodal earnings-call audio cues and speaker-aware financial-audio processing to improve Q&A reliability, including speaking-rate change, pause and hesitation patterns, emphasis or pitch variability, and alignment of vocal shifts with transcript guidance statements. This is explicitly later-stage and exploratory, and is not part of the current capstone MVP.

## Scope Notes
- This architecture includes the current transcript-first MVP and improved-version additions around guidance comparison and tone-change analysis.
- It intentionally focuses on deterministic, auditable signal extraction for decision support, not autonomous trading.
- The future audio layer is shown as optional and later-stage, and is not implied as implemented or validated in the current project.
- The document describes logical product flow only and intentionally omits infrastructure and deployment details.
