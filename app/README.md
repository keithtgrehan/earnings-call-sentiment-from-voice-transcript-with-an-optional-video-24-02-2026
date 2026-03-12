# App

This folder contains two local review interfaces for the earnings-call pipeline, both backed by the same deterministic workflow.

## Run

```bash
python app/server.py
```

Open `http://127.0.0.1:7860`.

Backup interface:
- `app/server.py`
- original panel-heavy layout

Primary shell:

```bash
python app/site_server.py
```

Open `http://127.0.0.1:7861`.

The primary shell is the cleaner long-term website surface. The backup interface remains available as a fallback while the shell evolves.

Runs execute as local background jobs. The review page refreshes while a run is active, so long YouTube transcriptions do not hold the browser request open.

## Inputs
- YouTube URL
- Local audio/video upload
- Local document upload: `.doc`, `.docx`, `.txt`, `.md`, `.csv`, `.json`
- Pasted transcript text

## Modes
- `Deterministic only`: transcript, sentiment, guidance, guidance revision, tone changes, metrics, report
- `Deterministic + LLM`: keeps deterministic artifacts as source of truth and adds `llm_summary.json`

## Notes
- Document mode uses extracted text and synthetic relative timing. It writes `document_timing_note.txt` to make that explicit.
- Legacy `.doc` extraction tries `textutil`, then `antiword`, then `soffice` if available.
- The home screen surfaces the strongest current benchmark subset from `data/gold_guidance_calls/`.

## Suggested starting prompt

```text
You are reviewing one earnings call. Stay grounded in the deterministic artifacts only. Identify the clearest guidance changes, the strongest tone-change moments, the evidence snippets that support them, and any places where the evidence is still ambiguous. Prefer conservative language over confident speculation. Do not make live trading claims or claim predictive edge.
```
