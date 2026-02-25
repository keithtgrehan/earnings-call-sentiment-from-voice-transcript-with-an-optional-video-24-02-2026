from collections.abc import Iterable


def chunk_segments(segments: Iterable[dict], chunk_seconds: float = 30.0) -> list[dict]:
    """
    Combine whisper segments into fixed-duration chunks.
    """

    chunks = []
    current = []
    start_time = None
    end_time = None

    for seg in segments:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", 0))
        text = seg.get("text", "").strip()

        if start_time is None:
            start_time = s

        if e - start_time <= chunk_seconds:
            current.append(text)
            end_time = e
        else:
            chunks.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": " ".join(current),
                    "segment_count": len(current),
                }
            )
            current = [text]
            start_time = s
            end_time = e

    if current:
        chunks.append(
            {
                "start": start_time,
                "end": end_time,
                "text": " ".join(current),
                "segment_count": len(current),
            }
        )

    return chunks
