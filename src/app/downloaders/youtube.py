from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class YouTubeDownloadResult:
    url: str
    audio_path: Path
    video_path: Optional[Path] = None
    title: Optional[str] = None
    id: Optional[str] = None


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def download_youtube(
    url: str,
    cache_dir: str | Path = "cache",
    audio_only: bool = True,
    audio_format: str = "wav",
    sample_rate: int = 16000,
) -> YouTubeDownloadResult:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_tmpl = str(cache_dir / "%(id)s.%(ext)s")

    title = None
    vid = None
    try:
        title = _run(["yt-dlp", "--get-title", url])
        vid = _run(["yt-dlp", "--get-id", url])
    except Exception:
        pass

    if audio_only:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", audio_format,
            "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
            "-o", base_tmpl,
            url,
        ]
        subprocess.run(cmd, check=True)

        if vid:
            audio_path = cache_dir / f"{vid}.{audio_format}"
        else:
            matches = sorted(cache_dir.glob(f"*.{audio_format}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                raise RuntimeError("yt-dlp completed but no audio file was found in cache/")
            audio_path = matches[0]

        return YouTubeDownloadResult(url=url, audio_path=audio_path, title=title, id=vid)

    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/best",
        "--merge-output-format", "mp4",
        "-o", base_tmpl,
        url,
    ]
    subprocess.run(cmd, check=True)

    if vid:
        video_path = cache_dir / f"{vid}.mp4"
    else:
        matches = sorted(cache_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise RuntimeError("yt-dlp completed but no mp4 file was found in cache/")
        video_path = matches[0]

    audio_path = cache_dir / (f"{vid}.{audio_format}" if vid else f"audio.{audio_format}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate), str(audio_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    return YouTubeDownloadResult(url=url, audio_path=audio_path, video_path=video_path, title=title, id=vid)
