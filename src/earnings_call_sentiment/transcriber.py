from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_transcribe_audio(
    audio_path: Path,
    *,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    vad_filter: bool = True,
    beam_size: int = 5,
) -> Iterator[Segment]:
    """
    Stream transcription segments from faster-whisper.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
    )

    for s in segments_iter:
        text = (s.text or "").strip()
        if not text:
            continue
        yield Segment(start=float(s.start), end=float(s.end), text=text)


def transcribe_audio(
    audio_path: Path,
    **kwargs: Any,
) -> list[Segment]:
    """
    Non-stream convenience wrapper.
    """
    # accept legacy kw names from CLI / older code
    if "model" in kwargs and "model_name" not in kwargs:
        kwargs["model_name"] = kwargs.pop("model")
    if "model_size" in kwargs and "model_name" not in kwargs:
        kwargs["model_name"] = kwargs.pop("model_size")

    return list(iter_transcribe_audio(audio_path, **kwargs))
