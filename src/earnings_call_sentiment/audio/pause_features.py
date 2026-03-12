from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import soundfile as sf

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_FILLER_RULES = (
    ("um", re.compile(r"\bum+\b")),
    ("uh", re.compile(r"\buh+\b")),
    ("you know", re.compile(r"\byou know\b")),
    ("kind of", re.compile(r"\bkind of\b")),
    ("sort of", re.compile(r"\bsort of\b")),
    ("i mean", re.compile(r"\bi mean\b")),
)


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    duration_s: float
    sample_rate: int
    frame_length_s: float
    hop_length_s: float
    silence_threshold_db: float
    frame_count: int


@dataclass(frozen=True)
class AudioEnvelope:
    metadata: AudioMetadata
    frame_times_s: np.ndarray
    rms_db: np.ndarray
    silent_mask: np.ndarray


def load_audio_envelope(
    audio_path: Path,
    *,
    frame_length_s: float = 0.03,
    hop_length_s: float = 0.01,
    silence_threshold_db: float = -35.0,
) -> AudioEnvelope:
    resolved = Path(audio_path).expanduser().resolve()
    waveform, sample_rate = sf.read(str(resolved), always_2d=False)
    if isinstance(waveform, tuple):
        waveform = np.asarray(waveform)
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        raise RuntimeError(f"Audio file is empty: {resolved}")

    frame_length = max(1, int(round(sample_rate * frame_length_s)))
    hop_length = max(1, int(round(sample_rate * hop_length_s)))
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))

    rms_values: list[float] = []
    frame_times: list[float] = []
    for start in range(0, signal.size - frame_length + 1, hop_length):
        frame = signal[start : start + frame_length]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))))
        frame_times.append(float(start / sample_rate))

    if not rms_values:
        rms_values = [float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))]
        frame_times = [0.0]

    rms = np.asarray(rms_values, dtype=np.float32)
    ref = float(np.percentile(rms, 95)) if np.any(rms > 0.0) else 1e-6
    ref = max(ref, 1e-6)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8) / ref)
    silent_mask = rms_db < float(silence_threshold_db)

    metadata = AudioMetadata(
        path=resolved,
        duration_s=float(signal.size / sample_rate),
        sample_rate=int(sample_rate),
        frame_length_s=float(frame_length / sample_rate),
        hop_length_s=float(hop_length / sample_rate),
        silence_threshold_db=float(silence_threshold_db),
        frame_count=int(len(frame_times)),
    )
    return AudioEnvelope(
        metadata=metadata,
        frame_times_s=np.asarray(frame_times, dtype=np.float32),
        rms_db=rms_db.astype(np.float32),
        silent_mask=silent_mask.astype(bool),
    )


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def count_fillers(text: str) -> tuple[int, list[str]]:
    lowered = " ".join(text.lower().split())
    total = 0
    matched: list[str] = []
    for label, pattern in _FILLER_RULES:
        hits = pattern.findall(lowered)
        if hits:
            total += len(hits)
            matched.append(label)
    return total, matched


def silence_ratio(envelope: AudioEnvelope, start_time_s: float, end_time_s: float) -> tuple[float, int]:
    if end_time_s <= start_time_s or envelope.frame_times_s.size == 0:
        return 0.0, 0
    frame_starts = envelope.frame_times_s
    frame_ends = frame_starts + envelope.metadata.frame_length_s
    mask = (frame_starts < float(end_time_s)) & (frame_ends > float(start_time_s))
    frame_count = int(mask.sum())
    if frame_count <= 0:
        return 0.0, 0
    return float(envelope.silent_mask[mask].mean()), frame_count


def leading_silence_before(
    envelope: AudioEnvelope,
    start_time_s: float,
    *,
    lookback_s: float = 2.0,
) -> float:
    if envelope.frame_times_s.size == 0:
        return 0.0
    frame_starts = envelope.frame_times_s
    frame_ends = frame_starts + float(envelope.metadata.frame_length_s)
    idx = int(np.searchsorted(frame_ends, float(start_time_s), side="right") - 1)
    if idx < 0:
        return 0.0
    lower_bound = max(0.0, float(start_time_s) - float(lookback_s))
    pause_s = 0.0
    while idx >= 0:
        frame_start = float(frame_starts[idx])
        if frame_start < lower_bound:
            break
        if not bool(envelope.silent_mask[idx]):
            break
        pause_s += float(envelope.metadata.hop_length_s)
        idx -= 1
    return min(pause_s, float(lookback_s)) * 1000.0


def speech_rate_wpm(text: str, duration_s: float) -> tuple[float, int]:
    word_count = count_words(text)
    if duration_s <= 0.0 or word_count <= 0:
        return 0.0, word_count
    return float(word_count / (duration_s / 60.0)), word_count
