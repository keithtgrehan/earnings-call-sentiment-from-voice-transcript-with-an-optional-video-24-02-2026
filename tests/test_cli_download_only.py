from __future__ import annotations

from pathlib import Path

import pytest

from earnings_call_sentiment import cli as cli_module


def test_cli_download_only_with_youtube_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    downloaded_audio = cache_dir / "youtube_audio.wav"
    calls: dict[str, object] = {}

    def fake_download(
        youtube_url: str, cache_dir: Path, audio_format: str = "wav"
    ) -> Path:
        calls["url"] = youtube_url
        calls["cache"] = cache_dir
        calls["audio_format"] = audio_format
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded_audio.write_bytes(b"wav-bytes")
        return downloaded_audio

    monkeypatch.setattr(cli_module, "download_audio", fake_download)
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("run_pipeline should not run")
        ),
    )

    exit_code = cli_module.main(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--cache-dir",
            str(cache_dir),
            "--download-only",
        ]
    )

    assert exit_code == 0
    assert calls["url"] == "https://www.youtube.com/watch?v=test123"
    assert calls["cache"] == cache_dir
    assert calls["audio_format"] == "wav"


def test_cli_download_only_with_audio_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "local.wav"
    audio_path.write_bytes(b"wav-bytes")

    monkeypatch.setattr(
        cli_module,
        "download_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("download_audio should not run")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("run_pipeline should not run")
        ),
    )

    exit_code = cli_module.main(["--audio-path", str(audio_path), "--download-only"])
    assert exit_code == 0
