from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from earnings_call_sentiment.pipeline import run as run_module


def test_pipeline_local_audio_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    source_audio = tmp_path / "input.wav"
    source_audio.write_bytes(b"audio-bytes")

    def fake_normalize(input_audio: Path, output_wav: Path, verbose: bool) -> None:
        assert input_audio == source_audio
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"wav-bytes")

    def fake_transcribe(
        audio_path: str,
        verbose: bool = False,
        model: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        chunk_seconds: float = 30.0,
    ) -> list[dict]:
        assert audio_path == str(cache_dir / "audio_normalized.wav")
        assert verbose is False
        assert model == "base"
        assert device == "auto"
        assert compute_type == "int8"
        assert chunk_seconds == 30.0
        return [{"start": 0.0, "end": 1.5, "text": "Hello world"}]

    def fake_sentiment(_: str):
        return [{"label": "POSITIVE", "score": 0.99}]

    monkeypatch.setattr(run_module, "_normalize_to_wav", fake_normalize)
    monkeypatch.setattr(run_module, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(run_module, "build_sentiment_pipeline", lambda: fake_sentiment)

    result = run_module.run_pipeline(
        youtube_url=None,
        audio_path=str(source_audio),
        cache_dir=str(cache_dir),
        out_dir=str(out_dir),
        verbose=False,
    )

    assert out_dir.exists()
    assert out_dir.is_dir()
    assert Path(result["audio"]).exists()
    assert Path(result["transcript_json"]).exists()
    assert Path(result["transcript_txt"]).exists()
    sentiment_csv = Path(result["sentiment_segments_csv"])
    assert sentiment_csv.exists()

    with sentiment_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["start", "end", "text", "sentiment", "score"]
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["sentiment"] == "POSITIVE"

    sentiment_plot = Path(result["sentiment_timeline_png"])
    assert sentiment_plot.exists()

    risk_metrics = Path(result["risk_metrics_json"])
    assert risk_metrics.exists()
    metrics = json.loads(risk_metrics.read_text(encoding="utf-8"))
    assert 0.0 <= float(metrics["risk_score"]) <= 100.0
