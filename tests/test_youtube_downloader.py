from __future__ import annotations

from pathlib import Path

import pytest

from yt_dlp.utils import DownloadError

from earnings_call_sentiment.downloaders import youtube as youtube_module


def test_download_youtube_audio_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "audio.wav"
    captured_opts: dict[str, object] = {}
    captured_call: dict[str, object] = {}

    class FakeYDL:
        def __init__(self, opts: dict[str, object]) -> None:
            captured_opts.update(opts)

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            captured_call["url"] = url
            captured_call["download"] = download
            output_file.write_bytes(b"wav-bytes")
            return {"id": "ignored"}

    monkeypatch.setattr(youtube_module.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(youtube_module.shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

    result = youtube_module.download_audio(
        youtube_url="https://www.youtube.com/watch?v=test123",
        cache_dir=tmp_path,
        audio_format="wav",
    )

    assert result == output_file.resolve()
    assert captured_call["url"] == "https://www.youtube.com/watch?v=test123"
    assert captured_call["download"] is True
    assert captured_opts["format"] == "bestaudio/best"
    assert captured_opts["outtmpl"] == str(tmp_path.resolve() / "audio.%(ext)s")
    postprocessors = captured_opts["postprocessors"]
    assert isinstance(postprocessors, list)
    assert postprocessors[0]["key"] == "FFmpegExtractAudio"
    assert postprocessors[0]["preferredcodec"] == "wav"
    assert postprocessors[0]["preferredquality"] == "0"


def test_download_youtube_audio_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeYDL:
        def __init__(self, _opts: dict[str, object]) -> None:
            pass

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        def extract_info(self, _url: str, download: bool) -> dict[str, object]:
            assert download is True
            raise DownloadError("yt-dlp error details")

    monkeypatch.setattr(youtube_module.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(youtube_module.shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

    with pytest.raises(RuntimeError, match="yt-dlp error details"):
        youtube_module.download_audio(
            youtube_url="https://www.youtube.com/watch?v=test123",
            cache_dir=tmp_path,
            audio_format="wav",
        )


def test_download_youtube_audio_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(youtube_module.shutil, "which", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        youtube_module.download_audio(
            youtube_url="https://www.youtube.com/watch?v=test123",
            cache_dir=tmp_path,
            audio_format="wav",
        )
