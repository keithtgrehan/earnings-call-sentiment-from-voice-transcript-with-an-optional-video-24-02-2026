from __future__ import annotations

import json
from pathlib import Path

import pytest

from earnings_call_sentiment.review_workflow import (
    build_segments_from_text,
    extract_text_from_document,
    load_artifact_bundle,
    prepare_review_run,
)


def test_build_segments_from_text_creates_monotonic_offsets() -> None:
    text = "\n\n".join(
        [
            "We reaffirm full-year guidance and expect revenue growth in the second half.",
            "Margin pressure remains manageable and we still expect operating leverage.",
            "This quarter includes stronger services demand and better retention.",
        ]
    )
    segments = build_segments_from_text(text, max_chars=60)
    assert len(segments) >= 3
    starts = [float(item["start"]) for item in segments]
    ends = [float(item["end"]) for item in segments]
    assert starts == sorted(starts)
    assert all(end > start for start, end in zip(starts, ends))
    assert segments[0]["text"]


def test_extract_text_from_docx(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("Raised revenue guidance for the fiscal year.")
    document.add_paragraph("Operating margin outlook also improved.")
    document.save(path)

    text = extract_text_from_document(path)

    assert "Raised revenue guidance" in text
    assert "Operating margin outlook" in text


def test_load_artifact_bundle_reads_tables_json_and_text(tmp_path: Path) -> None:
    review_run = prepare_review_run(repo_root=tmp_path, source_label="demo")
    out_dir = review_run.out_dir
    (out_dir / "transcript.txt").write_text("hello transcript", encoding="utf-8")
    (out_dir / "document_timing_note.txt").write_text(
        "relative timing only", encoding="utf-8"
    )
    (out_dir / "report.md").write_text("# Report\n\nSummary", encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"sentiment_mean": 0.1}), encoding="utf-8"
    )
    (out_dir / "guidance.csv").write_text(
        "start,end,text,guidance_strength\n0,10,raised guidance,0.9\n",
        encoding="utf-8",
    )

    bundle = load_artifact_bundle(review_run)

    assert bundle["text"]["transcript.txt"] == "hello transcript"
    assert bundle["text"]["document_timing_note.txt"] == "relative timing only"
    assert bundle["json"]["metrics.json"]["sentiment_mean"] == 0.1
    assert bundle["tables"]["guidance.csv"][0]["text"] == "raised guidance"
