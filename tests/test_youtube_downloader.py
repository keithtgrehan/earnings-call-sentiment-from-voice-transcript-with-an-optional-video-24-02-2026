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


def test_retryable_youtube_auth_failure_detection() -> None:
    assert youtube_module._is_retryable_youtube_auth_failure(
        "https://www.youtube.com/watch?v=test123",
        "Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies.",
    )
    assert not youtube_module._is_retryable_youtube_auth_failure(
        "https://vimeo.com/12345",
        "Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies.",
    )
    assert not youtube_module._is_retryable_youtube_auth_failure(
        "https://www.youtube.com/watch?v=test123",
        "HTTP Error 500: backend failure",
    )


def test_cookie_retry_sources_use_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cookie_file = tmp_path / "youtube.cookies.txt"
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "firefox:default")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookie_file))

    sources = youtube_module._build_cookie_retry_sources()

    assert [source.label for source in sources] == [
        "browser:firefox:default",
        f"cookies:{cookie_file}",
    ]
    assert sources[0].params == {
        "cookiesfrombrowser": ("firefox", "default", None, None)
    }
    assert sources[1].params == {"cookiefile": str(cookie_file)}


def test_download_youtube_audio_retries_with_browser_cookies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "audio.wav"
    attempt_opts: list[dict[str, object]] = []

    class FakeYDL:
        def __init__(self, opts: dict[str, object]) -> None:
            attempt_opts.append(dict(opts))

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        def extract_info(self, _url: str, download: bool) -> dict[str, object]:
            assert download is True
            if len(attempt_opts) == 1:
                raise DownloadError(
                    "Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies."
                )
            output_file.write_bytes(b"wav-bytes")
            return {"id": "ignored"}

    monkeypatch.delenv("YTDLP_COOKIES_FROM_BROWSER", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_FILE", raising=False)
    monkeypatch.setattr(youtube_module.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(youtube_module.shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

    result = youtube_module.download_audio(
        youtube_url="https://www.youtube.com/watch?v=test123",
        cache_dir=tmp_path,
        audio_format="wav",
    )

    assert result == output_file.resolve()
    assert len(attempt_opts) == 2
    assert "cookiesfrombrowser" not in attempt_opts[0]
    assert attempt_opts[1]["cookiesfrombrowser"] == ("safari", None, None, None)


def test_download_youtube_audio_non_auth_failure_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    class FakeYDL:
        def __init__(self, _opts: dict[str, object]) -> None:
            pass

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        def extract_info(self, _url: str, download: bool) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            assert download is True
            raise DownloadError("HTTP Error 500: backend failure")

    monkeypatch.setattr(youtube_module.yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(youtube_module.shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

    with pytest.raises(RuntimeError, match="HTTP Error 500: backend failure"):
        youtube_module.download_audio(
            youtube_url="https://www.youtube.com/watch?v=test123",
            cache_dir=tmp_path,
            audio_format="wav",
        )

    assert attempts == 1


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
