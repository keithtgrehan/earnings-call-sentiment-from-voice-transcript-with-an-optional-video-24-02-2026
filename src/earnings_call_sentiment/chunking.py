from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from .transcriber import Segment


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    start: float
    end: float
    text: str
    segment_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_segments(
    segments: list[Segment],
    *,
    window_seconds: float = 30.0,
    min_chars: int = 40,
) -> list[Chunk]:
    return list(iter_chunk_segments(segments, window_seconds=window_seconds, min_chars=min_chars))


def iter_chunk_segments(
    segments: list[Segment],
    *,
    window_seconds: float = 30.0,
    min_chars: int = 40,
) -> Iterator[Chunk]:
    """
    Deterministic chunking in fixed time windows.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    if not segments:
        return
        yield  # make mypy happy

    chunk_id = 0
    current_text: list[str] = []
    seg_count = 0

    current_start = segments[0].start
    window_end = current_start + window_seconds
    last_end = segments[0].end

    def flush(end_time: float, is_last: bool) -> Chunk | None:
        nonlocal chunk_id, current_text, seg_count
        text = " ".join(t.strip() for t in current_text if t.strip()).strip()
        if text and (len(text) >= min_chars or is_last):
            c = Chunk(
                chunk_id=chunk_id,
                start=float(current_start),
                end=float(end_time),
                text=text,
                segment_count=int(seg_count),
            )
            chunk_id += 1
            current_text = []
            seg_count = 0
            return c
        current_text = []
        seg_count = 0
        return None

    for s in segments:
        while s.start >= window_end:
            c = flush(end_time=last_end, is_last=False)
            if c is not None:
                yield c
            current_start = window_end
            window_end = current_start + window_seconds

        current_text.append(s.text)
        seg_count += 1
        last_end = s.end

    c = flush(end_time=last_end, is_last=True)
    if c is not None:
        yield c


class StreamingChunker:
    """
    Incremental chunker for a stream of Segment.
    Emits chunks as soon as we pass window boundaries.
    """

    def __init__(self, *, window_seconds: float = 30.0, min_chars: int = 40):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = float(window_seconds)
        self.min_chars = int(min_chars)

        self._initialized = False
        self._chunk_id = 0
        self._current_start = 0.0
        self._window_end = 0.0
        self._last_end = 0.0
        self._texts: list[str] = []
        self._seg_count = 0

    def _start(self, first: Segment) -> None:
        self._initialized = True
        self._current_start = float(first.start)
        self._window_end = self._current_start + self.window_seconds
        self._last_end = float(first.end)

    def _flush(self, *, is_last: bool) -> Chunk | None:
        text = " ".join(t.strip() for t in self._texts if t.strip()).strip()
        if text and (len(text) >= self.min_chars or is_last):
            c = Chunk(
                chunk_id=self._chunk_id,
                start=float(self._current_start),
                end=float(self._last_end),
                text=text,
                segment_count=int(self._seg_count),
            )
            self._chunk_id += 1
        else:
            c = None

        self._texts = []
        self._seg_count = 0
        return c

    def push(self, seg: Segment) -> list[Chunk]:
        if not self._initialized:
            self._start(seg)

        out: list[Chunk] = []

        while seg.start >= self._window_end:
            c = self._flush(is_last=False)
            if c is not None:
                out.append(c)
            self._current_start = self._window_end
            self._window_end = self._current_start + self.window_seconds

        self._texts.append(seg.text)
        self._seg_count += 1
        self._last_end = float(seg.end)
        return out

    def finalize(self) -> list[Chunk]:
        if not self._initialized:
            return []
        c = self._flush(is_last=True)
        return [c] if c is not None else []
