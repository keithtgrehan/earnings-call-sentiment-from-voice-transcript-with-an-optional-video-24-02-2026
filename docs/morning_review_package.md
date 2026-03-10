# Morning Review Package

## Current benchmark state
- Completed raw transcripts: 8
- Draft label rows: 8
- Label distribution: 6 `unclear`, 1 `maintained`, 1 `raised`
- Source-quality split: 2 `good`, 6 `caution`

## What is immediately usable
- `call08` / LLY: first transcript-backed `raised` example; explicit sentence says Lilly raised revenue and earnings-per-share guidance.
- `call02` / LEU: strongest current `maintained` control case with explicit transcript evidence (`our guidance is flat`).
- `call01` / PLTR: strongest `unclear` control from an official-source upload and explicit forward guidance range.
- `call03` / GOOGL: strongest second `unclear` control from an official IR upload.

## What should be held back for now
- `call04` / IBM
- `call05` / MSFT
- `call06` / AAPL
- `call07` / NVDA

Reason: these rows are reproducible and useful as `unclear` examples, but they come from repost sources and do not improve directional label coverage enough to carry the benchmark alone.

## Source-quality notes
- The benchmark is now materially better because it contains at least one transcript-backed `raised` row and one transcript-backed `maintained` row.
- `call02` / LEU remains the messiest metadata row; use `official_source_manifest.csv` before promoting it into a final benchmark subset.
- `call04` / IBM event date has been corrected to the official January 28, 2026 call date in the working benchmark files.

## Directional expansion status
- `call08` / LLY Q2 2025: completed and promoted into the benchmark as `raised`.
- `call09` / IBM Q3 2025: next best `raised` candidate if more runtime is available.
- Lowered-case gap remains open. Hologic is still the best current target, but I have not yet found a clean ingestible webcast path for the existing CLI.

## Recommended next review order
1. Review `call08` first; it is the biggest overnight improvement.
2. Keep `call02` as the first maintained control case.
3. Keep `call01` and `call03` as the first two high-quality `unclear` controls.
4. Add IBM Q3 2025 next if one more raised row is worth the runtime cost.
5. Do not promote the full eight-call batch as final gold truth until at least one clean `lowered` example is added.
