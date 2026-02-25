# earnings-call-sentiment-from-voice-transcript-with-an-optional-video-24-02-2026

Analyze earnings call audio and track sentiment dynamics over time.

## Run (YouTube)

```bash
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --audio-format wav \
  --model base \
  --device auto \
  --compute-type int8 \
  --chunk-seconds 30 \
  --verbose
```

Default run artifacts:

- `outputs/transcript.json`
- `outputs/transcript.txt`
- `outputs/sentiment_segments.csv`
- `outputs/sentiment_timeline.png`
- `outputs/risk_metrics.json`

## Question Shifts

Use `--question-shifts` to detect analyst-style questions and measure sentiment
changes after each question.

Example:

```bash
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cache-dir ./cache \
  --out-dir ./outputs \
  --question-shifts \
  --pre-window-s 60 \
  --post-window-s 120 \
  --min-gap-s 30
```

Artifacts:

- `outputs/question_sentiment_shifts.csv`
- `outputs/question_shifts.png`

You can also run stage-specific modes:

```bash
earnings-call-sentiment \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cache-dir ./cache \
  --audio-format wav \
  --download-only
```

Other stage flags:

- `--transcribe-only`
- `--score-only`
