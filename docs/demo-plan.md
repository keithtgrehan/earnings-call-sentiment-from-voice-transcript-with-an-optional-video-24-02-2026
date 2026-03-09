# Demo Plan

## Demo Objective
Show that the prototype turns a long earnings call into a structured, auditable signal package that a retail trader can review quickly.

## Recommended Demo Mode
Use the **no-API deterministic path** with pre-generated outputs. This avoids live download/transcription/model calls during presentation.

## Safest Runbook (No External APIs)
1. Verify pre-generated artifacts:
   ```bash
   python scripts/verify_outputs.py --out-dir ./outputs_prior
   ```
2. Generate deterministic narrative summary:
   ```bash
   python scripts/run_eval.py --out-dir ./outputs_prior --llm none
   ```
3. Walk through outputs in this order:
   - `outputs_prior/report.md` (overall narrative)
   - `outputs_prior/metrics.json` (structured summary)
   - `outputs_prior/guidance.csv` and `outputs_prior/guidance_revision.csv` (what changed)
   - `outputs_prior/tone_changes.csv` (where tone moved)
   - `outputs_prior/llm_eval.json` (deterministic recap from step 2)

## Talk Track (Presentation Friendly)
- Start with the user pain: too much unstructured call text to process quickly.
- Show the engine’s structured outputs and direct evidence links.
- Emphasize deterministic behavior and reproducibility.
- Close with evaluation boundary: this is decision support, not a proven alpha engine.

## Backup Plan
If `outputs_prior/` is missing, use any previously generated local output folder that contains `metrics.json`, `report.md`, and `guidance*.csv`, then run the same verification and deterministic summary commands against that folder.
