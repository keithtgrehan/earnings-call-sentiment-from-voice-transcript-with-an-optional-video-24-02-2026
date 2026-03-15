"""YouTube downloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, cast
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import DownloadError

_SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "m4a"}
_RETRYABLE_YOUTUBE_FAILURE_MARKERS = (
    "sign in to confirm you're not a bot",
    "sign in to confirm you’re not a bot",
    "use --cookies-from-browser or --cookies",
    "cookies-from-browser",
    "this helps protect our community",
    "captcha",
    "not a bot",
)


@dataclass(frozen=True)
class _CookieRetrySource:
    label: str
    params: dict[str, Any]


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


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.casefold()
    return host in {"youtube.com", "youtu.be"} or host.endswith(".youtube.com")


def _default_cookie_browsers() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("safari", "chrome", "firefox")
    return ("chrome", "firefox", "safari")


def _format_download_error(exc: Exception, logger: _LoggerCapture) -> str:
    details = "\n".join(logger.errors).strip()
    message = str(exc).strip()
    if details and details not in message:
        return f"{message}\n{details}" if message else details
    return message or details or exc.__class__.__name__


def _is_retryable_youtube_auth_failure(youtube_url: str, error_text: str) -> bool:
    if not _is_youtube_url(youtube_url):
        return False
    normalized = error_text.casefold()
    return any(marker in normalized for marker in _RETRYABLE_YOUTUBE_FAILURE_MARKERS)


def _parse_cookies_from_browser(browser_spec: str) -> tuple[str, str | None, str | None, str | None]:
    spec = browser_spec.strip()
    match = re.fullmatch(
        r"""(?x)
        (?P<name>[^+:]+)
        (?:\s*\+\s*(?P<keyring>[^:]+))?
        (?:\s*:\s*(?!:)(?P<profile>.+?))?
        (?:\s*::\s*(?P<container>.+))?
        """,
        spec,
    )
    if match is None:
        raise ValueError(f"invalid cookies-from-browser value: {browser_spec!r}")
    browser_name, keyring, profile, container = match.group("name", "keyring", "profile", "container")
    return (
        browser_name.casefold(),
        profile,
        keyring.upper() if keyring else None,
        container,
    )


def _build_cookie_retry_sources() -> list[_CookieRetrySource]:
    env_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    env_cookie_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    sources: list[_CookieRetrySource] = []

    if env_browser:
        sources.append(
            _CookieRetrySource(
                label=f"browser:{env_browser}",
                params={"cookiesfrombrowser": _parse_cookies_from_browser(env_browser)},
            )
        )
    if env_cookie_file:
        cookie_path = Path(env_cookie_file).expanduser()
        sources.append(
            _CookieRetrySource(
                label=f"cookies:{cookie_path}",
                params={"cookiefile": str(cookie_path)},
            )
        )

    if sources:
        return sources

    return [
        _CookieRetrySource(
            label=f"browser:{browser}",
            params={"cookiesfrombrowser": (browser, None, None, None)},
        )
        for browser in _default_cookie_browsers()
    ]


def _run_yt_dlp_download(youtube_url: str, params: Dict[str, Any]) -> str | None:
    logger = cast(_LoggerCapture, params["logger"])
    try:
        with yt_dlp.YoutubeDL(cast(Any, params)) as ydl:
            ydl.extract_info(youtube_url, download=True)
    except DownloadError as exc:
        return _format_download_error(exc, logger)
    except Exception as exc:
        return _format_download_error(exc, logger)
    return None


def _format_cookie_retry_failure(
    *,
    failure_prefix: str,
    initial_error: str,
    retry_labels: list[str],
    retry_errors: list[str],
) -> str:
    lines = [
        "YouTube blocked anonymous download.",
        f"The app retried with browser cookies: {', '.join(retry_labels)}.",
        "If downloads are still blocked, use Local media or Document mode.",
        "",
        f"Initial error: {initial_error}",
    ]
    if retry_errors:
        lines.append("Retry errors:")
        lines.extend(f"- {item}" for item in retry_errors)
    return f"{failure_prefix}: " + "\n".join(lines)


def _download_with_cookie_retry(
    *,
    youtube_url: str,
    params: Dict[str, Any],
    failure_prefix: str,
) -> None:
    initial_error = _run_yt_dlp_download(youtube_url, params)
    if initial_error is None:
        return
    if not _is_retryable_youtube_auth_failure(youtube_url, initial_error):
        raise RuntimeError(f"{failure_prefix}: {initial_error}")

    try:
        retry_sources = _build_cookie_retry_sources()
    except ValueError as exc:
        raise RuntimeError(
            f"{failure_prefix}: invalid browser cookie configuration: {exc}"
        ) from exc

    retry_labels: list[str] = []
    retry_errors: list[str] = []
    for source in retry_sources:
        retry_labels.append(source.label)
        retry_params = dict(params)
        retry_params.update(source.params)
        retry_params["logger"] = _LoggerCapture()
        retry_error = _run_yt_dlp_download(youtube_url, retry_params)
        if retry_error is None:
            return
        retry_errors.append(f"{source.label}: {retry_error}")

    raise RuntimeError(
        _format_cookie_retry_failure(
            failure_prefix=failure_prefix,
            initial_error=initial_error,
            retry_labels=retry_labels,
            retry_errors=retry_errors,
        )
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


def download_youtube_video(youtube_url: str, cache_dir: Path) -> Path:
    """Download a deterministic MP4 video file when a YouTube video source is available."""
    return download_video(
        youtube_url=youtube_url,
        cache_dir=cache_dir,
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

    _download_with_cookie_retry(
        youtube_url=youtube_url,
        params=params,
        failure_prefix="YouTube download failed",
    )

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


def download_video(youtube_url: str, cache_dir: Path) -> Path:
    """Download best-effort MP4 video for optional visual analysis."""
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    _check_ffmpeg_available()
    target_path = cache_path / "video.mp4"
    if target_path.exists():
        target_path.unlink()

    logger = _LoggerCapture()
    params: Dict[str, Any] = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "noplaylist": True,
        "outtmpl": str(cache_path / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "prefer_ffmpeg": True,
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
    }

    _download_with_cookie_retry(
        youtube_url=youtube_url,
        params=params,
        failure_prefix="YouTube video download failed",
    )

    if not target_path.exists() or not target_path.is_file():
        candidates = sorted(
            p
            for p in cache_path.glob("video.*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        )
        if candidates:
            try:
                candidates[0].replace(target_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Video download completed but failed to write {target_path}: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                "YouTube video download completed but merged video file was not created: "
                f"{target_path}"
            )

    return target_path.resolve()
