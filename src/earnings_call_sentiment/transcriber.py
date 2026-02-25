"""Transcription helpers with optional progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Iterator, TypedDict

from faster_whisper import WhisperModel
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


class TranscriptSegment(TypedDict):
    """Normalized segment output used by pipeline code."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    """Compatibility segment type used by chunking helpers/tests."""

    start: float
    end: float
    text: str


def get_audio_duration_s(path: Path) -> float:
    """Return audio duration in seconds using soundfile, falling back to librosa."""
    audio_path = path.expanduser().resolve()
    try:
        info = sf.info(str(audio_path))
        duration = float(info.duration)
        if duration > 0.0:
            return duration
    except Exception:
        pass

    try:
        import librosa
    except Exception as exc:
        raise RuntimeError(
            "Unable to determine audio duration. Install/read dependencies for "
            f"{audio_path}: {exc}"
        ) from exc

    try:
        duration = float(librosa.get_duration(path=str(audio_path)))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to determine audio duration for {audio_path}: {exc}"
        ) from exc

    if duration <= 0.0:
        raise RuntimeError(f"Audio duration is not positive for {audio_path}")
    return duration


def _should_emit_progress(verbose: bool, logger: logging.Logger | None) -> bool:
    return bool(verbose or (logger and logger.isEnabledFor(logging.DEBUG)))


def _emit_progress(message: str, logger: logging.Logger | None) -> None:
    if logger is not None:
        logger.debug(message)
        return
    print(message)


def iter_transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
    verbose: bool = False,
    progress_log_interval_s: float = 15.0,
    logger: logging.Logger | None = None,
    **transcribe_kwargs: Any,
) -> Iterator[TranscriptSegment]:
    """Yield transcription segments with optional progress updates."""
    resolved_audio = Path(audio_path).expanduser().resolve()
    total_s = 0.0
    try:
        total_s = get_audio_duration_s(resolved_audio)
    except RuntimeError as exc:
        if _should_emit_progress(verbose=verbose, logger=logger):
            _emit_progress(
                f"transcribe progress: duration unavailable ({exc})",
                logger=logger,
            )

    progress_enabled = _should_emit_progress(verbose=verbose, logger=logger)
    interval = max(0.1, float(progress_log_interval_s))

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw_segments, _ = model.transcribe(str(resolved_audio), **transcribe_kwargs)

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=total_s if total_s > 0.0 else None,
            unit="s",
            disable=not progress_enabled,
            leave=False,
        )

    processed_s = 0.0
    prev_processed_s = 0.0
    last_log_t = time.monotonic()
    last_logged_processed_s = 0.0

    try:
        for segment in raw_segments:
            seg_start = float(getattr(segment, "start", 0.0) or 0.0)
            seg_end = float(getattr(segment, "end", 0.0) or 0.0)
            seg_text = str(getattr(segment, "text", "") or "").strip()

            processed_s = max(processed_s, seg_end)
            delta = max(0.0, processed_s - prev_processed_s)
            if progress_bar is not None and delta > 0.0:
                progress_bar.update(delta)
            prev_processed_s = processed_s

            now = time.monotonic()
            if progress_enabled and now - last_log_t >= interval:
                pct = (processed_s / total_s * 100.0) if total_s > 0.0 else 0.0
                _emit_progress(
                    "transcribe progress: processed "
                    f"{processed_s:.1f}s / {total_s:.1f}s ({pct:.1f}%)",
                    logger=logger,
                )
                last_log_t = now
                last_logged_processed_s = processed_s

            yield {"start": seg_start, "end": seg_end, "text": seg_text}
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if progress_enabled and processed_s > last_logged_processed_s:
        pct = (processed_s / total_s * 100.0) if total_s > 0.0 else 0.0
        _emit_progress(
            "transcribe progress: processed "
            f"{processed_s:.1f}s / {total_s:.1f}s ({pct:.1f}%)",
            logger=logger,
        )


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
    verbose: bool = False,
    progress_log_interval_s: float = 15.0,
    logger: logging.Logger | None = None,
    **transcribe_kwargs: Any,
) -> list[TranscriptSegment]:
    """Collect and return all transcribed segments."""
    return list(
        iter_transcribe_audio(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            verbose=verbose,
            progress_log_interval_s=progress_log_interval_s,
            logger=logger,
            **transcribe_kwargs,
        )
    )
