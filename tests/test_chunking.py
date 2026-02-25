from earnings_call_sentiment.chunking import chunk_segments
from earnings_call_sentiment.transcriber import Segment


def test_chunk_segments_groups_by_window():
    segs = [
        Segment(0.0, 5.0, "Hello"),
        Segment(10.0, 12.0, "world"),
        Segment(31.0, 33.0, "next window"),
    ]
    chunks = chunk_segments(segs, window_seconds=30.0, min_chars=1)
    assert len(chunks) == 2
    assert chunks[0].start == 0.0
    assert chunks[0].end == 12.0
    assert "Hello" in chunks[0].text
    assert "world" in chunks[0].text
    assert chunks[1].start == 30.0
