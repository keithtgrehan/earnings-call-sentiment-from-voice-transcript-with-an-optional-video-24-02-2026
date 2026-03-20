# Freeze Boundaries

This cleanup pass is limited to consistency work on the `main` branch. The goal is to keep public docs and committed artifacts honest about the current repo state without changing the frozen core.

## Frozen In This Pass
- Transcript core is frozen.
- Benchmark assets are frozen.
- Transcript-backed deterministic outputs remain the canonical review truth.
- Committed multimodal artifacts remain supporting evidence only.

## Allowed Cleanup
- Restore missing boundary docs.
- Remove multimodal entrypoints that do not run on canonical `main`.
- Update public docs so committed multimodal counts and limitations are described accurately.
- Scrub local absolute-path leakage from committed multimodal artifacts.
- Soften wording that overstates proof, maturity, or demonstrated capability.

## Out Of Scope
- Adding features or widening scope.
- Changing transcript-core behavior or benchmark assets.
- Claiming benchmark lift, predictive lift, or validated multimodal advantage beyond the committed artifacts.
- Creating a new worktree, switching worktrees, or pushing changes.
