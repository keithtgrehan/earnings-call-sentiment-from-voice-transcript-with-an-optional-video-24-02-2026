from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from earnings_call_sentiment import cli as cli_module


def test_cli_transcribe_only_writes_transcript_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    audio_path = tmp_path / "input.wav"

    sample_rate = 16000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    signal = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(audio_path, signal, sample_rate)

    def fake_normalize(
        input_audio: Path, output_wav: Path, verbose: bool = False
    ) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_audio, output_wav)
        return output_wav

    def fake_transcribe(*_args, **_kwargs) -> list[dict]:
        return [{"start": 0.0, "end": 1.0, "text": "Hello earnings call"}]

    monkeypatch.setattr(cli_module, "normalize_audio_to_wav", fake_normalize)
    monkeypatch.setattr(cli_module, "transcribe_audio", fake_transcribe)

    exit_code = cli_module.main(
        [
            "--audio-path",
            str(audio_path),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--transcribe-only",
            "--verbose",
        ]
    )

    assert exit_code == 0
    transcript_json = out_dir / "transcript.json"
    transcript_txt = out_dir / "transcript.txt"
    assert transcript_json.exists()
    assert transcript_txt.exists()
    assert "Hello earnings call" in transcript_txt.read_text(encoding="utf-8")
