from pathlib import Path

import yt_dlp


def download_audio(url: str, cache_dir: Path, audio_format: str = "wav") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "audio.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
            }
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return cache_dir / f"audio.{audio_format}"
