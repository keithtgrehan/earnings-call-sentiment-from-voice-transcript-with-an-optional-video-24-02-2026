"""Core pipeline execution for earnings call transcription."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any, TypedDict

import matplotlib
import matplotlib.pyplot as plt
from transformers import pipeline as hf_pipeline

matplotlib.use("Agg")

from earnings_call_sentiment.downloaders.youtube import download_audio
from earnings_call_sentiment.transcriber import transcribe_audio as _transcribe_audio


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[verbose] {message}")


class SentimentArtifacts(TypedDict):
    sentiment_segments_csv: str
    sentiment_timeline_png: str
    risk_metrics_json: str
    risk_score: float
    sentiment_segments: list[dict]


def _require_file(path: Path, name: str) -> None:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"{name} file not found: {path}")


def _normalize_to_wav(input_audio: Path, output_wav: Path, verbose: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav),
    ]
    _log(verbose, f"ffmpeg command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        message = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"ffmpeg normalization failed:\n{message.strip()}")


def normalize_audio_to_wav(
    input_audio: Path, output_wav: Path, verbose: bool = False
) -> Path:
    """Normalize audio to mono 16k WAV and return output path."""
    _normalize_to_wav(input_audio=input_audio, output_wav=output_wav, verbose=verbose)
    return output_wav


def write_transcript_artifacts(
    segments: list[dict], output_path: Path
) -> tuple[Path, Path]:
    """Write transcript JSON/TXT artifacts and return their paths."""
    transcript_json = output_path / "transcript.json"
    transcript_txt = output_path / "transcript.txt"
    transcript_json.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    transcript_txt.write_text(
        "\n".join(item.get("text", "") for item in segments if item.get("text")),
        encoding="utf-8",
    )
    return transcript_json, transcript_txt


def load_transcript_segments(transcript_json_path: Path) -> list[dict[str, Any]]:
    """Load transcript segments from transcript.json."""
    raw = json.loads(transcript_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"Invalid transcript JSON format: {transcript_json_path}")
    return [item for item in raw if isinstance(item, dict)]


def transcribe_audio(
    audio_path: str,
    verbose: bool = False,
    model: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
    chunk_seconds: float = 30.0,
) -> list[dict]:
    """Transcribe audio using faster-whisper base model in int8 mode."""
    transcribe_kwargs: dict[str, Any] = {}
    if chunk_seconds > 0:
        transcribe_kwargs["chunk_length"] = max(1, int(round(chunk_seconds)))
    return list(
        _transcribe_audio(
            audio_path=audio_path,
            model_name=model,
            device=device,
            compute_type=compute_type,
            verbose=verbose,
            **transcribe_kwargs,
        )
    )


def build_sentiment_pipeline():
    """Create a transformers sentiment analysis pipeline."""
    return hf_pipeline("sentiment-analysis")


def score_segments_with_sentiment(segments: list[dict]) -> list[dict]:
    """Add sentiment label/score to transcript segments."""
    sentiment = build_sentiment_pipeline()
    scored: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if text:
            result = sentiment(text)[0]
            label = str(result.get("label", ""))
            score = float(result.get("score", 0.0))
        else:
            label = "NEUTRAL"
            score = 0.0
        scored.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": text,
                "sentiment": label,
                "score": score,
            }
        )
    return scored


def _signed_score(segment: dict) -> float:
    label = str(segment.get("sentiment", "")).upper()
    score = float(segment.get("score", 0.0))
    if "NEG" in label:
        return -score
    if "POS" in label:
        return score
    return 0.0


def compute_risk_score(sentiment_segments: list[dict]) -> float:
    """Compute a 0-100 risk score from avg sentiment, volatility, and negatives."""
    if not sentiment_segments:
        return 0.0

    signed_scores = [_signed_score(item) for item in sentiment_segments]
    n = len(signed_scores)
    mean_sentiment = sum(signed_scores) / n

    variance = sum((value - mean_sentiment) ** 2 for value in signed_scores) / n
    volatility = math.sqrt(variance)

    negative_count = sum(
        1
        for item in sentiment_segments
        if "NEG" in str(item.get("sentiment", "")).upper()
    )
    negative_ratio = negative_count / n

    avg_component = ((1 - mean_sentiment) / 2) * 100
    vol_component = min(max(volatility, 0.0), 1.0) * 100
    neg_component = negative_ratio * 100

    risk = (0.4 * avg_component) + (0.3 * vol_component) + (0.3 * neg_component)
    return round(max(0.0, min(100.0, risk)), 2)


def plot_sentiment_timeline(scored_segments: list[dict], output_path: Path) -> None:
    """Plot sentiment score over segment start time."""
    x_values = [float(item.get("start", 0.0)) for item in scored_segments]
    y_values = [float(item.get("score", 0.0)) for item in scored_segments]

    plt.figure(figsize=(10, 4))
    if x_values:
        plt.plot(x_values, y_values, marker="o", linewidth=1.5)
    else:
        plt.plot([], [])
        plt.text(
            0.5,
            0.5,
            "No sentiment segments",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Sentiment score")
    plt.title("Sentiment Timeline")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_sentiment_artifacts(
    segments: list[dict], output_path: Path, verbose: bool = False
) -> SentimentArtifacts:
    """Score transcript segments and write sentiment/risk artifacts."""
    _log(verbose, "Scoring transcript segments")
    sentiment_segments = score_segments_with_sentiment(segments)

    sentiment_csv = output_path / "sentiment_segments.csv"
    with sentiment_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "text", "sentiment", "score"],
        )
        writer.writeheader()
        writer.writerows(sentiment_segments)

    sentiment_plot = output_path / "sentiment_timeline.png"
    plot_sentiment_timeline(sentiment_segments, sentiment_plot)

    signed_scores = [_signed_score(item) for item in sentiment_segments]
    segment_count = len(sentiment_segments)
    mean_sentiment = sum(signed_scores) / segment_count if segment_count else 0.0
    volatility = (
        math.sqrt(
            sum((value - mean_sentiment) ** 2 for value in signed_scores)
            / segment_count
        )
        if segment_count
        else 0.0
    )
    negative_ratio = (
        sum(
            1
            for item in sentiment_segments
            if "NEG" in str(item.get("sentiment", "")).upper()
        )
        / segment_count
        if segment_count
        else 0.0
    )
    risk_metrics = {
        "risk_score": compute_risk_score(sentiment_segments),
        "average_sentiment": round(mean_sentiment, 4),
        "volatility": round(volatility, 4),
        "negative_ratio": round(negative_ratio, 4),
        "segments": segment_count,
    }
    risk_metrics_path = output_path / "risk_metrics.json"
    risk_metrics_path.write_text(json.dumps(risk_metrics, indent=2), encoding="utf-8")

    _require_file(sentiment_csv, "Sentiment segments CSV")
    _require_file(sentiment_plot, "Sentiment timeline PNG")
    _require_file(risk_metrics_path, "Risk metrics JSON")

    return {
        "sentiment_segments_csv": str(sentiment_csv),
        "sentiment_timeline_png": str(sentiment_plot),
        "risk_metrics_json": str(risk_metrics_path),
        "risk_score": float(risk_metrics["risk_score"]),
        "sentiment_segments": sentiment_segments,
    }


def run_pipeline(
    youtube_url: str | None,
    audio_path: str | None,
    cache_dir: str | None,
    out_dir: str | None,
    verbose: bool = False,
    audio_format: str = "wav",
    model: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
    chunk_seconds: float = 30.0,
) -> dict:
    """Run audio acquisition, normalization, transcription, and transcript export."""
    cache_path = Path(cache_dir or "./cache")
    output_path = Path(out_dir or "./outputs")
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    _log(verbose, f"cache_dir={cache_path.resolve()}")
    _log(verbose, f"out_dir={output_path.resolve()}")

    source_audio: Path
    if youtube_url:
        print("Stage 1/5: Downloading audio from YouTube")
        source_audio = Path(
            download_audio(youtube_url, cache_path, audio_format=audio_format)
        )
    elif audio_path:
        print("Stage 1/5: Using provided audio path")
        source_audio = Path(audio_path).expanduser()
    else:
        raise ValueError("Provide either youtube_url or audio_path")

    _require_file(source_audio, "Source audio")
    _log(verbose, f"source_audio={source_audio.resolve()}")

    print("Stage 2/5: Normalizing audio to mono 16k WAV")
    # Keep normalization output separate from downloaded input to avoid in-place overwrite.
    normalized_wav = cache_path / "audio_normalized.wav"
    normalize_audio_to_wav(source_audio, normalized_wav, verbose=verbose)
    _require_file(normalized_wav, "Normalized WAV")
    _log(verbose, f"normalized_wav={normalized_wav.resolve()}")

    print("Stage 3/5: Transcribing with faster-whisper (base, int8)")
    segments = transcribe_audio(
        str(normalized_wav),
        verbose=verbose,
        model=model,
        device=device,
        compute_type=compute_type,
        chunk_seconds=chunk_seconds,
    )
    _log(verbose, f"segments_count={len(segments)}")

    print("Stage 4/5: Saving transcript artifacts")
    transcript_json, transcript_txt = write_transcript_artifacts(segments, output_path)
    _require_file(transcript_json, "Transcript JSON")
    _require_file(transcript_txt, "Transcript text")
    _log(verbose, f"transcript_json={transcript_json.resolve()}")
    _log(verbose, f"transcript_txt={transcript_txt.resolve()}")

    print("Stage 5/5: Scoring sentiment and generating postprocess artifacts")
    artifacts = write_sentiment_artifacts(segments=segments, output_path=output_path)
    sentiment_csv = Path(str(artifacts["sentiment_segments_csv"]))
    sentiment_plot = Path(str(artifacts["sentiment_timeline_png"]))
    risk_metrics_path = Path(str(artifacts["risk_metrics_json"]))
    _log(verbose, f"sentiment_segments_csv={sentiment_csv.resolve()}")
    _log(verbose, f"sentiment_timeline_png={sentiment_plot.resolve()}")
    _log(verbose, f"risk_metrics_json={risk_metrics_path.resolve()}")

    print("Pipeline complete")
    return {
        "audio": str(normalized_wav),
        "transcript_json": str(transcript_json),
        "transcript_txt": str(transcript_txt),
        "sentiment_segments_csv": str(sentiment_csv),
        "sentiment_timeline_png": str(sentiment_plot),
        "risk_metrics_json": str(risk_metrics_path),
        "risk_score": artifacts["risk_score"],
    }
