# Labeling Rubric: Guidance-Change MVP

## Purpose
This rubric defines a deterministic, closed-label process for annotating management guidance-change signals in earnings-call transcripts.

## Closed Label Set
- raised
- maintained
- lowered
- withdrawn
- unclear

## Strict Rule
Narrative tone does not count as guidance unless management explicitly changes guidance.

## Label Definitions
- raised: Management explicitly states guidance was increased or revised upward versus prior guidance.
- maintained: Management explicitly states guidance is unchanged, reaffirmed, or maintained.
- lowered: Management explicitly states guidance was reduced or revised downward versus prior guidance.
- withdrawn: Management explicitly states guidance is suspended, removed, or no longer provided.
- unclear: Guidance direction cannot be determined from explicit management language.

## Unclear Examples
- "We remain focused on execution despite macro uncertainty." (No explicit guidance change)
- "Demand trends are improving quarter over quarter." (Narrative performance statement only)
- "We feel confident about the second half." (Tone signal without explicit guidance update)

## Labeling Workflow (Human Annotators)
1. Read the full management guidance section before assigning a label.
2. Highlight the shortest explicit evidence text supporting the label.
3. Record evidence span start/end offsets and assign one closed-set label.
4. If explicit direction is missing or conflicting, assign `unclear`.
5. Add brief notes only when the evidence is ambiguous.
