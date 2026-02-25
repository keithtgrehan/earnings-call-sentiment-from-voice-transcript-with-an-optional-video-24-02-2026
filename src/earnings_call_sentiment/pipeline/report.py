"""Report generation helpers for pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _coerce_segments_count(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _build_sentiment_summary(data: dict, segments_value: object) -> dict:
    existing = data.get("sentiment_summary")
    if isinstance(existing, dict):
        return existing

    summary: dict[str, object] = {}
    if isinstance(segments_value, list):
        labels = [
            str(item.get("sentiment_label", "")).upper()
            for item in segments_value
            if isinstance(item, dict)
        ]
        if labels:
            summary["positive"] = labels.count("POSITIVE")
            summary["neutral"] = labels.count("NEUTRAL")
            summary["negative"] = labels.count("NEGATIVE")

    for key in ("mean_sentiment", "std_sentiment"):
        if key in data:
            summary[key] = data[key]
    return summary


def build_report(data: dict, out_dir: str):
    """Create out_dir/report.json from pipeline output data."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_value = data.get("segments", [])
    report = {
        "audio_file": data.get("audio_file") or data.get("audio_path"),
        "transcript_file": data.get("transcript_file") or data.get("transcript_path"),
        "segments": _coerce_segments_count(segments_value),
        "risk_score": float(data.get("risk_score", 0.0)),
        "sentiment_summary": _build_sentiment_summary(data, segments_value),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
