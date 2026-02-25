from __future__ import annotations

from pathlib import Path

import pandas as pd

from earnings_call_sentiment.question_shifts import (
    detect_question_shifts,
    plot_question_shifts,
    run,
)


def test_detect_question_shifts_with_min_gap() -> None:
    segments = [
        {"start": 15.0, "end": 25.0, "text": "Prepared remarks continue."},
        {
            "start": 100.0,
            "end": 108.0,
            "text": "Could you discuss demand trends for next quarter?",
        },
        {
            "start": 112.0,
            "end": 118.0,
            "text": "What about regional growth?",
        },
        {
            "start": 180.0,
            "end": 188.0,
            "text": "Can you explain margin pressure in more detail?",
        },
    ]
    chunks_scored = pd.DataFrame(
        [
            {"start": 40.0, "end": 70.0, "signed_score": 0.2},
            {"start": 70.0, "end": 100.0, "signed_score": 0.1},
            {"start": 100.0, "end": 130.0, "signed_score": -0.4},
            {"start": 130.0, "end": 160.0, "signed_score": -0.2},
            {"start": 160.0, "end": 190.0, "signed_score": -0.1},
            {"start": 190.0, "end": 220.0, "signed_score": 0.0},
        ]
    )

    result = detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=60.0,
        after_window=120.0,
        min_gap_s=30.0,
    )

    assert list(result.columns) == [
        "question_time",
        "question_text",
        "sentiment_before",
        "sentiment_after",
        "sentiment_shift",
    ]
    assert len(result) == 2
    assert float(result.iloc[0]["question_time"]) == 100.0
    assert float(result.iloc[1]["question_time"]) == 180.0
    assert float(result.iloc[0]["sentiment_shift"]) < 0.0


def test_plot_question_shifts_writes_png(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"question_time": 60.0, "sentiment_shift": -0.2},
            {"question_time": 120.0, "sentiment_shift": 0.1},
        ]
    )
    output_path = tmp_path / "question_shifts.png"

    figure = plot_question_shifts(df, output_path=output_path)

    assert output_path.exists()
    assert output_path.is_file()
    figure.clear()


def test_run_writes_artifacts_with_no_detected_questions(tmp_path: Path) -> None:
    segments = [{"start": 1.0, "end": 2.0, "text": "short"}]
    chunks_scored = pd.DataFrame(
        [{"start": 0.0, "end": 30.0, "signed_score": 0.1, "text": "hello"}]
    )

    df = run(segments=segments, chunks_scored=chunks_scored, out_dir=tmp_path)

    assert df.empty
    assert (tmp_path / "question_sentiment_shifts.csv").exists()
    assert (tmp_path / "question_shifts.png").exists()
