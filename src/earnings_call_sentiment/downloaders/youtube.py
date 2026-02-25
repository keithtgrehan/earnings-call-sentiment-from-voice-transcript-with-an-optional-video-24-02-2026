"""YouTube downloader utilities."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, cast

import yt_dlp
from yt_dlp.utils import DownloadError

_SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "m4a"}


class _LoggerCapture:
    """Capture yt-dlp error messages for clear exception output."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def debug(self, _message: str) -> None:
        return

    def warning(self, _message: str) -> None:
        return

    def error(self, message: str) -> None:
        self.errors.append(str(message))


def _check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for audio extraction but was not found on PATH. "
            "Install ffmpeg and retry."
        )


def download_youtube_audio(
    youtube_url: str, cache_dir: Path, audio_format: str = "wav"
) -> Path:
    """Download YouTube audio to a deterministic file in cache_dir."""
    return download_audio(
        youtube_url=youtube_url,
        cache_dir=cache_dir,
        audio_format=audio_format,
    )


def download_audio(youtube_url: str, cache_dir: Path, audio_format: str) -> Path:
    """Download best audio and extract to audio_format using yt-dlp Python API."""
    if audio_format not in _SUPPORTED_AUDIO_FORMATS:
        raise ValueError("audio_format must be one of: wav, mp3, m4a")

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    _check_ffmpeg_available()
    target_path = cache_path / f"audio.{audio_format}"
    if target_path.exists():
        target_path.unlink()

    logger = _LoggerCapture()
    params: Dict[str, Any] = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(cache_path / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "0",
            }
        ],
        "prefer_ffmpeg": True,
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
    }

    try:
        with yt_dlp.YoutubeDL(cast(Any, params)) as ydl:
            ydl.extract_info(youtube_url, download=True)
    except DownloadError as exc:
        details = "\n".join(logger.errors).strip()
        if details:
            raise RuntimeError(f"YouTube download failed: {exc}\n{details}") from exc
        raise RuntimeError(f"YouTube download failed: {exc}") from exc
    except Exception as exc:
        details = "\n".join(logger.errors).strip()
        if details:
            raise RuntimeError(f"YouTube download failed: {details}") from exc
        raise RuntimeError(f"YouTube download failed: {exc}") from exc

    if not target_path.exists() or not target_path.is_file():
        candidates = sorted(
            p
            for p in cache_path.glob("audio.*")
            if p.is_file() and p.suffix.lower() == f".{audio_format}"
        )
        if candidates:
            try:
                candidates[0].replace(target_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Download completed but failed to write {target_path}: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                "YouTube download completed but converted audio file was not created: "
                f"{target_path}"
            )

    return target_path.resolve()
