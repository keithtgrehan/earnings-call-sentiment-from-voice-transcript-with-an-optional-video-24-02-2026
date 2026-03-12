# Watchlist-Derived Unseen Holdout

This package is a second unseen evaluation set built from watchlist-derived companies that are not part of the frozen benchmark and are intentionally kept separate from the active holdout benchmark.

## Why this package exists
- preserve the current active holdout checkpoint without mutating it
- evaluate the deterministic label mapper on a fresh watchlist-derived set
- keep excerpt-heavy unseen rows separate from the main holdout until more full transcripts are available

## Scope
- package type: unseen evaluation set
- source style: official-source excerpts and release/transcript snippets
- not part of the frozen benchmark under `data/gold_guidance_calls/`
- not part of the active holdout benchmark under `data/gold_guidance_calls_holdout/`

## Current composition
- raised: 3
- maintained: 1
- unclear: 3
- lowered: 0
- withdrawn: 0

## Limitations
- the set is excerpt-heavy rather than full-transcript-heavy
- no lowered row was added because no clean watchlist-based lowered candidate was supportable in this pass
- labels remain conservative and default to `unclear` when direction is not explicit
