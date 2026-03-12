"""Legacy downloader module kept for tooling/type-check compatibility."""

from __future__ import annotations

from pathlib import Path

from earnings_call_sentiment.downloaders.youtube import download_audio as _download_audio


def download_audio(youtube_url: str, cache_dir: Path, audio_format: str) -> Path:
    """Backwards-compatible wrapper around the primary YouTube downloader."""
    return _download_audio(
        youtube_url=youtube_url,
        cache_dir=cache_dir,
        audio_format=audio_format,
    )
