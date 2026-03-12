from __future__ import annotations

import pandas as pd

from earnings_call_sentiment import cli
from earnings_call_sentiment.review_scorecard import build_review_scorecard


def _guidance_df(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": 0.0,
                "end": 6.0,
                "text": text,
                "guidance_strength": 0.8,
            }
        ]
    )


def _guidance_revision(*, label: str, snippet: str, matched_count: int = 1) -> dict[str, object]:
    counts = {
        "raised_count": 0,
        "lowered_count": 0,
        "reaffirmed_count": 0,
        "unclear_count": 0,
        "mixed_count": 0,
        "withdrawn_count": 0,
    }
    key = {
        "raised": "raised_count",
        "lowered": "lowered_count",
        "reaffirmed": "reaffirmed_count",
        "withdrawn": "withdrawn_count",
    }.get(label)
    if key is not None:
        counts[key] = 1
    elif label == "unclear":
        counts["unclear_count"] = 1
    elif label == "mixed":
        counts["mixed_count"] = 1
    return {
        "prior_guidance_path": "/tmp/prior_guidance.csv",
        "matched_count": matched_count,
        **counts,
        "top_revisions": [{"label": label, "snippet": snippet}],
    }


def _behavioral_summary(
    *,
    uncertainty_level: str,
    reassurance_level: str,
    skepticism_level: str,
    reassurance_rows: list[dict[str, object]] | None = None,
    uncertainty_rows: list[dict[str, object]] | None = None,
    skepticism_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "uncertainty_score_overall": {"score": 0, "level": uncertainty_level},
        "reassurance_score_management": {"score": 0, "level": reassurance_level},
        "analyst_skepticism_score": {"score": 0, "level": skepticism_level},
        "strongest_evidence": {
            "reassurance": reassurance_rows or [],
            "uncertainty": uncertainty_rows or [],
            "analyst_skepticism": skepticism_rows or [],
        },
    }


def _qa_shift_summary(
    *,
    prepared_label: str = "mixed",
    early_late_label: str = "mixed",
    skeptic_level: str = "low",
    answer_uncertainty_label: str = "mixed",
    analyst_questions: int = 3,
    management_answers: int = 3,
    qa_pairs: list[dict[str, object]] | None = None,
    skepticism_rows: list[dict[str, object]] | None = None,
    answer_uncertainty_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "prepared_remarks_vs_q_and_a": {"label": prepared_label, "delta": 0.0},
        "analyst_skepticism": {"level": skeptic_level, "score": 0},
        "management_answers_vs_prepared_uncertainty": {"label": answer_uncertainty_label, "delta": 0.0},
        "early_vs_late_q_and_a": {"label": early_late_label, "delta": 0.0},
        "counts": {
            "analyst_questions": analyst_questions,
            "management_answers": management_answers,
        },
        "strongest_evidence": {
            "qa_pairs": qa_pairs or [],
            "skepticism": skepticism_rows or [],
            "answer_uncertainty": answer_uncertainty_rows or [],
        },
    }


def test_build_review_scorecard_maps_raised_guidance() -> None:
    scorecard = build_review_scorecard(
        guidance_df=_guidance_df("We are raising our full-year revenue guidance."),
        guidance_revision=_guidance_revision(
            label="raised",
            snippet="We are raising our full-year revenue guidance.",
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="low",
            reassurance_level="medium",
            skepticism_level="low",
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="mixed",
            early_late_label="mixed",
            skeptic_level="low",
            qa_pairs=[
                {
                    "answer_minus_question": 0.08,
                    "question_text": "What changed in the outlook?",
                    "answer_text": "Demand improved and we raised guidance.",
                },
                {
                    "answer_minus_question": 0.05,
                    "question_text": "Anything else behind the raise?",
                    "answer_text": "Execution remained on plan.",
                },
            ],
        ),
    )

    assert scorecard["guidance_strength_score"] == 10
    assert scorecard["guidance_strength_band"] == "green"


def test_build_review_scorecard_maps_lowered_guidance() -> None:
    scorecard = build_review_scorecard(
        guidance_df=_guidance_df("We are lowering our full-year revenue outlook."),
        guidance_revision=_guidance_revision(
            label="lowered",
            snippet="We are lowering our full-year revenue outlook.",
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="medium",
            reassurance_level="low",
            skepticism_level="medium",
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="weaker",
            early_late_label="weaker",
            skeptic_level="medium",
            answer_uncertainty_label="more uncertain",
            qa_pairs=[
                {
                    "answer_minus_question": -0.09,
                    "question_text": "Why reduce the outlook now?",
                    "answer_text": "Visibility is lower and demand is harder to estimate.",
                },
                {
                    "answer_minus_question": -0.06,
                    "question_text": "What changed since last quarter?",
                    "answer_text": "Several assumptions became less certain.",
                },
            ],
        ),
    )

    assert scorecard["guidance_strength_score"] == 3
    assert scorecard["guidance_strength_band"] == "red"


def test_build_review_scorecard_reverse_colors_uncertainty() -> None:
    scorecard = build_review_scorecard(
        guidance_df=_guidance_df("We reaffirmed guidance."),
        guidance_revision=_guidance_revision(
            label="reaffirmed",
            snippet="We reaffirmed guidance.",
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="low",
            reassurance_level="medium",
            skepticism_level="low",
            uncertainty_rows=[
                {
                    "matched_phrase": "seasonality",
                    "text": "Seasonality remains normal and visibility is solid.",
                }
            ],
        ),
        qa_shift_summary=_qa_shift_summary(
            qa_pairs=[
                {
                    "answer_minus_question": 0.02,
                    "question_text": "Any change to assumptions?",
                    "answer_text": "No material change to assumptions.",
                },
                {
                    "answer_minus_question": 0.01,
                    "question_text": "Any hidden risks?",
                    "answer_text": "Nothing notable beyond normal seasonality.",
                },
            ],
        ),
    )

    assert scorecard["uncertainty_score"] == 2
    assert scorecard["uncertainty_band"] == "green"


def test_build_review_scorecard_confidence_stays_clamped() -> None:
    low_confidence = build_review_scorecard(
        guidance_df=pd.DataFrame(),
        guidance_revision={
            "prior_guidance_path": "/tmp/prior_guidance.csv",
            "matched_count": 0,
            "raised_count": 0,
            "lowered_count": 0,
            "reaffirmed_count": 0,
            "unclear_count": 1,
            "mixed_count": 0,
            "top_revisions": [{"label": "unclear", "snippet": "The outlook remains subject to change."}],
        },
        behavioral_summary=_behavioral_summary(
            uncertainty_level="high",
            reassurance_level="low",
            skepticism_level="high",
        ),
        qa_shift_summary=_qa_shift_summary(
            analyst_questions=0,
            management_answers=0,
            qa_pairs=[],
        ),
    )
    high_confidence = build_review_scorecard(
        guidance_df=_guidance_df("We are raising our full-year revenue guidance."),
        guidance_revision=_guidance_revision(
            label="raised",
            snippet="We are raising our full-year revenue guidance.",
            matched_count=2,
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="low",
            reassurance_level="high",
            skepticism_level="low",
            reassurance_rows=[
                {"matched_phrase": "confident", "text": "We are confident in the second half."},
                {"matched_phrase": "on track", "text": "Execution remains on track."},
            ],
            uncertainty_rows=[
                {"matched_phrase": "stable", "text": "Demand trends remain stable."}
            ],
            skepticism_rows=[
                {"matched_phrase": "clarify", "text": "Analysts asked for routine clarification only."}
            ],
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="stronger",
            early_late_label="stronger",
            skeptic_level="low",
            answer_uncertainty_label="less uncertain",
            qa_pairs=[
                {
                    "answer_minus_question": 0.11,
                    "question_text": "Why raise guidance now?",
                    "answer_text": "Bookings and conversion both improved materially.",
                },
                {
                    "answer_minus_question": 0.09,
                    "question_text": "Is the raise broad-based?",
                    "answer_text": "Yes, strength was broad across segments.",
                },
            ],
            skepticism_rows=[
                {"matched_phrase": "clarify", "text": "Can you clarify the cadence?"},
            ],
            answer_uncertainty_rows=[
                {"matched_phrase": "committed", "text": "We remain committed to the raised target."},
            ],
        ),
    )

    assert low_confidence["review_confidence_pct"] == 35
    assert 35 <= high_confidence["review_confidence_pct"] <= 95


def test_build_review_scorecard_sets_green_overall_signal() -> None:
    scorecard = build_review_scorecard(
        guidance_df=_guidance_df("We are raising our full-year revenue guidance."),
        guidance_revision=_guidance_revision(
            label="raised",
            snippet="We are raising our full-year revenue guidance.",
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="low",
            reassurance_level="high",
            skepticism_level="low",
            reassurance_rows=[
                {"matched_phrase": "confident", "text": "We are confident in the second half."}
            ],
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="stronger",
            early_late_label="stronger",
            skeptic_level="low",
            answer_uncertainty_label="less uncertain",
            qa_pairs=[
                {
                    "answer_minus_question": 0.17,
                    "question_text": "What changed in the outlook?",
                    "answer_text": "Demand remained strong and we raised guidance.",
                },
                {
                    "answer_minus_question": 0.10,
                    "question_text": "Can you sustain this momentum?",
                    "answer_text": "Yes, our backlog and execution support it.",
                },
            ],
        ),
    )

    assert scorecard["overall_review_signal"] == "green"
    assert scorecard["management_confidence_score"] >= 9
    assert scorecard["qa_pressure_shift_score"] == 3
    assert scorecard["answer_directness_score"] >= 7


def test_build_review_scorecard_sets_red_overall_signal() -> None:
    scorecard = build_review_scorecard(
        guidance_df=_guidance_df("We are lowering our full-year revenue outlook."),
        guidance_revision=_guidance_revision(
            label="lowered",
            snippet="We are lowering our full-year revenue outlook.",
        ),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="high",
            reassurance_level="low",
            skepticism_level="high",
            uncertainty_rows=[
                {
                    "matched_phrase": "hard to estimate",
                    "text": "Demand remains hard to estimate and visibility remains limited.",
                }
            ],
            skepticism_rows=[
                {"matched_phrase": "why", "text": "Why was the prior guide missed?"},
                {"matched_phrase": "what changed", "text": "What changed so quickly in the quarter?"},
            ],
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="weaker",
            early_late_label="weaker",
            skeptic_level="high",
            answer_uncertainty_label="more uncertain",
            qa_pairs=[
                {
                    "answer_minus_question": -0.12,
                    "question_text": "Why did the forecast deteriorate?",
                    "answer_text": "The environment is difficult to predict and subject to change.",
                },
                {
                    "answer_minus_question": -0.10,
                    "question_text": "How much confidence should we have now?",
                    "answer_text": "Visibility remains limited and assumptions are less certain.",
                },
            ],
            skepticism_rows=[
                {"matched_phrase": "why", "text": "Why should investors trust this reset?"},
            ],
            answer_uncertainty_rows=[
                {"matched_phrase": "subject to", "text": "The outlook remains subject to significant change."},
            ],
        ),
    )

    assert scorecard["overall_review_signal"] == "red"
    assert scorecard["uncertainty_band"] == "red"
    assert scorecard["analyst_skepticism_band"] == "red"
    assert scorecard["qa_pressure_shift_band"] == "red"


def test_build_metrics_payload_includes_review_scorecard_fields() -> None:
    payload = cli._build_metrics_payload(
        chunks_scored=pd.DataFrame(
            [
                {"signed_score": 0.6},
                {"signed_score": 0.3},
            ]
        ),
        guidance_df=pd.DataFrame(
            [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "We reaffirmed full-year guidance.",
                    "guidance_strength": 0.4,
                }
            ]
        ),
        guidance_revision_df=pd.DataFrame(
            [
                {
                    "revision_label": "reaffirmed",
                    "is_matched": True,
                    "diff": 0.05,
                    "current_start": 0.0,
                    "current_end": 5.0,
                    "current_text_snippet": "We reaffirmed full-year guidance.",
                }
            ]
        ),
        tone_changes_df=pd.DataFrame([{"is_change": True}]),
        behavioral_summary=_behavioral_summary(
            uncertainty_level="low",
            reassurance_level="medium",
            skepticism_level="low",
            reassurance_rows=[
                {
                    "matched_phrase": "confident",
                    "text": "We remain confident in our plan.",
                }
            ],
        ),
        qa_shift_summary=_qa_shift_summary(
            prepared_label="mixed",
            early_late_label="mixed",
            skeptic_level="low",
            qa_pairs=[
                {
                    "answer_minus_question": 0.01,
                    "question_text": "Any change to outlook?",
                    "answer_text": "No, we reaffirmed guidance.",
                },
                {
                    "answer_minus_question": 0.02,
                    "question_text": "Anything else to call out?",
                    "answer_text": "Execution is tracking to plan.",
                },
            ],
        ),
        prior_guidance_path=None,
        sentiment_model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        sentiment_revision="714eb0f",
    )

    assert payload["schema_version"] == "1.0.0"
    assert payload["overall_review_signal"] in {"green", "amber", "red"}
    assert isinstance(payload["review_confidence_pct"], int)
    assert payload["guidance_strength_score"] == payload["review_scorecard"]["guidance_strength_score"]
    assert payload["answer_directness_score"] == payload["review_scorecard"]["answer_directness_score"]
    assert payload["review_scorecard"]["categories"]
