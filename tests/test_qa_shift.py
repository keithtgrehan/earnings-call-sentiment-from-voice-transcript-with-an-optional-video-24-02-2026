from __future__ import annotations

import pandas as pd

from earnings_call_sentiment.signals.qa_shift import compute_qa_shift_outputs


def _chunks(*rows: tuple[float, float, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": start,
                "end": end,
                "text": text,
                "signed_score": score,
            }
            for start, end, text, score in rows
        ]
    )


def test_qa_shift_pairs_questions_with_following_answers() -> None:
    payload = compute_qa_shift_outputs(
        _chunks(
            (0.0, 5.0, "We remain confident in the quarter.", 0.4),
            (10.0, 15.0, "What changed in demand?", -0.2),
            (16.0, 22.0, "We may see some timing uncertainty in the near term.", -0.3),
            (30.0, 35.0, "What gives you confidence that margins recover?", -0.2),
            (36.0, 42.0, "We remain on track and are well positioned for the second half.", 0.3),
        )
    )
    df = payload["segments_df"]
    qa_rows = df[df["phase"] == "q_and_a"]
    assert list(qa_rows["qa_pair_id"]) == [1, 1, 2, 2]
    assert qa_rows.iloc[1]["question_text"] == "What changed in demand?"
    assert qa_rows.iloc[3]["question_text"] == "What gives you confidence that margins recover?"


def test_qa_shift_reports_weaker_q_and_a_and_more_uncertain_answers() -> None:
    payload = compute_qa_shift_outputs(
        _chunks(
            (0.0, 5.0, "We remain confident and demand remains strong.", 0.5),
            (6.0, 11.0, "We are well positioned for the year.", 0.4),
            (20.0, 26.0, "What changed in demand?", -0.2),
            (27.0, 35.0, "We may see timing uncertainty and visibility remains limited.", -0.4),
            (40.0, 47.0, "How sustainable do you think those are?", -0.3),
            (48.0, 56.0, "It is difficult to predict when supply and demand will balance.", -0.5),
        )
    )
    summary = payload["summary"]
    assert summary["prepared_remarks_vs_q_and_a"]["label"] == "weaker"
    assert summary["management_answers_vs_prepared_uncertainty"]["label"] == "more uncertain"
    assert summary["analyst_skepticism"]["level"] in {"medium", "high"}


def test_qa_shift_reports_stable_flow_when_sentiment_is_balanced() -> None:
    payload = compute_qa_shift_outputs(
        _chunks(
            (0.0, 5.0, "We remain on track for the quarter.", 0.2),
            (10.0, 14.0, "Could you just talk a little bit about channel mix?", 0.1),
            (15.0, 20.0, "We remain confident in the current demand trends.", 0.22),
        )
    )
    summary = payload["summary"]
    assert summary["prepared_remarks_vs_q_and_a"]["label"] == "mixed"
    assert summary["management_answers_vs_prepared_uncertainty"]["label"] == "mixed"
