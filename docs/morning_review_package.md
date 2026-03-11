# Morning Review Package

## Current benchmark state
- Completed raw transcripts: 9
- Draft label rows: 9
- Label distribution: 6 `unclear`, 1 `maintained`, 1 `raised`, 1 `lowered`
- Source-quality split: 2 `good`, 7 `caution`

## What is immediately usable
- `call08` / LLY: transcript-backed `raised` example; explicit sentence says Lilly raised revenue and earnings-per-share guidance.
- `call09` / PVH: transcript-backed `lowered` example; explicit sentence says full-year non-GAAP guidance was taken down for EBIT margin and EPS.
- `call02` / LEU: strongest current `maintained` control case with explicit transcript evidence (`our guidance is flat`).
- `call01` / PLTR: strongest `unclear` control from an official-source upload and explicit forward guidance range.
- `call03` / GOOGL: strongest second `unclear` control from an official IR upload.

## What should be held back for now
- `call04` / IBM
- `call05` / MSFT
- `call06` / AAPL
- `call07` / NVDA

Reason: these rows are reproducible and useful as `unclear` examples, but they do not improve directional label coverage enough to lead the benchmark subset.

## Source-quality notes
- The benchmark is now materially stronger because it contains transcript-backed `raised`, `lowered`, and `maintained` rows.
- `call09` is a mixed case: revenue guidance was reaffirmed while earnings guidance moved down. The closed-set nearest label is still `lowered`, but the note should stay attached during review.
- `call02` / LEU remains the messiest metadata row; use `official_source_manifest.csv` before promoting it into a final benchmark subset.

## Recommended first-pass benchmark subset
1. `call08` / LLY (`raised`)
2. `call09` / PVH (`lowered`)
3. `call02` / LEU (`maintained`)
4. `call01` / PLTR (`unclear`)
5. `call03` / GOOGL (`unclear`)

## Residual risks
- The directional examples come from repost video sources even though their guidance direction is confirmed by primary-source investor materials.
- The benchmark still benefits from more `good` source rows and at least one additional lowered example for redundancy.
