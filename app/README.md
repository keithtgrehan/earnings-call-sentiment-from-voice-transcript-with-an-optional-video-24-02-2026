# App

This folder contains a local review interface for the earnings-call pipeline.

## Run

```bash
python app/server.py
```

Open `http://127.0.0.1:7860`.

## Inputs
- YouTube URL
- Local audio/video upload
- Local document upload: `.doc`, `.docx`, `.txt`, `.md`, `.csv`, `.json`
- Pasted transcript text

## Modes
- `Deterministic only`: transcript, sentiment, guidance, guidance revision, tone changes, metrics, report
- `Deterministic + LLM`: keeps deterministic artifacts as source of truth and adds `llm_summary.json`

## Suggested starting prompt

```text
You are reviewing one earnings call. Stay grounded in the deterministic artifacts only. Identify the clearest guidance changes, the strongest tone-change moments, the evidence snippets that support them, and any places where the evidence is still ambiguous. Prefer conservative language over confident speculation. Do not make live trading claims or claim predictive edge.
```
