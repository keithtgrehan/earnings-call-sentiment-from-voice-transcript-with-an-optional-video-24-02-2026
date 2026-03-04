# Earnings Call Sentiment Signal Engine

A local AI pipeline that transforms earnings call audio into structured sentiment analytics and trading signals.

---

## What This Tool Does

This system:

1. Downloads an earnings call (YouTube)
2. Transcribes audio using Whisper (faster-whisper)
3. Splits transcript into time-based chunks
4. Runs transformer-based sentiment scoring
5. Detects structural phases (operator / management / analyst)
6. Computes trading signals:
   - Analyst Pressure
   - Management Confidence
7. Extracts guidance statements
8. Detects tone changes across the call
9. Tracks analyst hostility
10. Detects management evasiveness
11. Generates structured outputs + report

Everything runs locally. No SaaS dependency.

---

## Pipeline Overview

YouTube → Audio → Whisper Transcription → Chunking → Sentiment NLP → Phase Detection → Signal Extraction → Metrics → Report

---

## CLI Usage

### Clean Install

```bash
pip uninstall -y earnings-call-sentiment earnings_call_sentiment || true
pip install -e .
hash -r

earnings-call-sentiment --help | rg -- '--vad|--force|--resume'

Example Run (VAD OFF — Recommended)
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=jNQXAC9IVRw" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model tiny \
  --chunk-seconds 20 \
  --min-chars 20 \
  --question-shifts \
  --verbose

Example Run (VAD ON with fallback)
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=jNQXAC9IVRw" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --model tiny \
  --chunk-seconds 20 \
  --min-chars 20 \
  --vad \
  --question-shifts \
  --verbose

If VAD removes 100% of audio, system automatically retries with VAD OFF.


Resumable Stages

Pipeline stages:
	1.	Download
	2.	Transcribe → segments.jsonl
	3.	Chunk → chunks.jsonl
	4.	Score → chunks_scored.jsonl
	5.	Question shifts (optional)
	6.	Phase detection → phases.jsonl
	7.	Metrics → metrics.json
	8.	Signals → CSV outputs
	9.	Report → report.md

Behavior:
	•	--resume (default ON)
	•	--force forces full recompute


Core Outputs

Core JSONL
	•	segments.jsonl
	•	chunks.jsonl
	•	chunks_scored.jsonl

Structural
	•	phases.jsonl

Signals
	•	analyst_pressure.csv
	•	management_confidence.csv
	•	analyst_hostility.csv
	•	management_evasiveness.csv
	•	guidance.csv
	•	tone_changes.csv

Metrics
	•	metrics.json

Report
	•	report.md

⸻

Trading Signals Implemented

Analyst Pressure

Measures how aggressive analyst questions are.

Components:
	•	Negative sentiment
	•	Adversarial keywords
	•	Question density
	•	Length normalization

⸻

Management Confidence

Measures clarity and conviction of management tone.

Components:
	•	Positive sentiment
	•	Confident keywords
	•	Numeric specificity
	•	Hedging penalties

⸻

Guidance Extraction

Detects forward-looking guidance and scores strength.

Components:
	•	Revenue/EPS/margin cues
	•	Numeric specificity
	•	Range detection
	•	Hedging penalties

⸻

Tone Change Detection

Detects statistically significant shifts in sentiment.

Uses rolling mean/std with z-score threshold.

⸻

Analyst Hostility

Scores adversarial Q&A behavior.

Components:
	•	Negative sentiment
	•	Confrontational keywords
	•	Interruption cues
	•	Question intensity

⸻

Management Evasiveness

Detects non-answers and deflection.

Components:
	•	Evasive language
	•	Deflection phrases
	•	Hedging
	•	Lack of numeric specificity

---

## Optional LLM Narrative Evaluation

This does **not** change the pipeline. It reads existing artifacts in `--out-dir`
and writes:

- `llm_eval.json` (always)
- `llm_eval.md` (human-readable summary)

Deterministic mode (no LLM call):

```bash
python scripts/run_eval.py \
  --out-dir ./outputs \
  --llm none
```

Local Ollama mode (strict JSON response validated):

```bash
python scripts/run_eval.py \
  --out-dir ./outputs \
  --llm ollama \
  --model llama3.2:1b \
  --limit 5 \
  --top-k 10 \
  --timeout 240 \
  --max-tokens 300
```
