from __future__ import annotations

import pandas as pd

from earnings_call_sentiment.signals.behavior import compute_behavioral_outputs


def _chunks(*texts: str) -> pd.DataFrame:
    rows = []
    for idx, text in enumerate(texts):
        rows.append(
            {
                "start": float(idx * 10),
                "end": float((idx * 10) + 5),
                "text": text,
                "sentiment": "POSITIVE",
                "score": 0.7,
                "signed_score": 0.7,
            }
        )
    return pd.DataFrame(rows)


def test_uncertainty_positive_matches_management_hedging() -> None:
    payload = compute_behavioral_outputs(
        _chunks("We may see timing uncertainty this quarter and visibility remains limited.")
    )
    df = payload["uncertainty_df"]
    assert not df.empty
    assert set(df["matched_phrase"]) >= {"modal uncertainty", "timing uncertainty", "limited visibility"}
    assert payload["summary"]["uncertainty_score_overall"]["level"] in {"medium", "high"}


def test_uncertainty_negative_keeps_direct_statement_clean() -> None:
    payload = compute_behavioral_outputs(
        _chunks("Revenue grew 20% and the company shipped all committed units in the quarter.")
    )
    assert payload["uncertainty_df"].empty
    assert payload["summary"]["uncertainty_score_overall"]["score"] == 0


def test_reassurance_positive_matches_management_language() -> None:
    payload = compute_behavioral_outputs(
        _chunks("We remain confident, demand remains strong, and we are well positioned for the year.")
    )
    df = payload["reassurance_df"]
    assert not df.empty
    assert set(df["matched_phrase"]) >= {"remain confident", "demand remains strong", "well positioned"}
    assert payload["summary"]["reassurance_score_management"]["level"] in {"medium", "high"}


def test_reassurance_negative_ignores_analyst_question_language() -> None:
    payload = compute_behavioral_outputs(
        _chunks("What gives you confidence that demand remains strong?")
    )
    assert payload["reassurance_df"].empty


def test_skepticism_low_medium_high_examples() -> None:
    low_payload = compute_behavioral_outputs(_chunks("What gives you confidence here?"))
    medium_payload = compute_behavioral_outputs(_chunks("Help us understand why demand changed."))
    high_payload = compute_behavioral_outputs(
        _chunks("Why should we believe that and isn't that inconsistent with prior guidance?")
    )

    assert low_payload["skepticism_df"].iloc[0]["skepticism_label"] == "low"
    assert medium_payload["skepticism_df"].iloc[0]["skepticism_label"] == "low" or medium_payload["skepticism_df"].iloc[0]["skepticism_label"] == "medium"
    assert high_payload["skepticism_df"].iloc[0]["skepticism_label"] == "high"


def test_behavior_regression_non_matches_stay_non_matches() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "Operator, our next question comes from John.",
            "Thanks for taking the question.",
            "We delivered revenue growth and margin expansion in the quarter.",
        )
    )
    assert payload["uncertainty_df"].empty
    assert payload["reassurance_df"].empty
    assert payload["skepticism_df"].empty
