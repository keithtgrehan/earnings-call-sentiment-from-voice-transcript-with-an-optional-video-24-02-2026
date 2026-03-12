from __future__ import annotations

import pandas as pd

from earnings_call_sentiment import cli
from earnings_call_sentiment.review_scorecard import build_review_scorecard


def test_build_review_scorecard_maps_deterministic_signals() -> None:
    guidance_df = pd.DataFrame(
        [
            {
                "start": 0.0,
                "end": 6.0,
                "text": "We are raising our full-year revenue guidance.",
                "guidance_strength": 0.88,
            }
        ]
    )
    guidance_revision = {
        "prior_guidance_path": "/tmp/prior_guidance.csv",
        "matched_count": 1,
        "raised_count": 1,
        "lowered_count": 0,
        "reaffirmed_count": 0,
        "unclear_count": 0,
        "mixed_count": 0,
        "top_revisions": [
            {"label": "raised", "snippet": "We are raising our full-year revenue guidance."}
        ],
    }
    behavioral_summary = {
        "uncertainty_score_overall": {"score": 0, "level": "low"},
        "reassurance_score_management": {"score": 3, "level": "high"},
        "analyst_skepticism_score": {"score": 1, "level": "low"},
        "strongest_evidence": {
            "reassurance": [
                {"matched_phrase": "confident", "text": "We are confident in the second half.", "strength": 0.9}
            ],
            "uncertainty": [],
            "analyst_skepticism": [],
        },
    }
    qa_shift_summary = {
        "prepared_remarks_vs_q_and_a": {"label": "stronger", "delta": 0.16},
        "analyst_skepticism": {"level": "low", "score": 0},
        "management_answers_vs_prepared_uncertainty": {"label": "less uncertain", "delta": -0.12},
        "early_vs_late_q_and_a": {"label": "stronger", "delta": 0.08},
        "counts": {"analyst_questions": 3, "management_answers": 3},
        "strongest_evidence": {
            "qa_pairs": [
                {
                    "answer_minus_question": 0.17,
                    "question_text": "What changed in the outlook?",
                    "answer_text": "Demand remained strong and we raised guidance.",
                }
            ],
            "answer_uncertainty": [],
            "skepticism": [],
        },
    }

    scorecard = build_review_scorecard(
        guidance_df=guidance_df,
        guidance_revision=guidance_revision,
        behavioral_summary=behavioral_summary,
        qa_shift_summary=qa_shift_summary,
    )

    assert scorecard["overall_review_signal"] == "green"
    assert scorecard["review_confidence_pct"] >= 70
    assert scorecard["guidance_strength_score"] >= 8
    assert scorecard["uncertainty_score"] >= 8
    assert scorecard["answer_directness_score"] >= 7
    assert any(
        item["name"] == "Guidance Strength" and item["rank"] <= 3
        for item in scorecard["ranked_categories"]
    )
    assert scorecard["categories"][0]["strongest_evidence"]


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
        behavioral_summary={
            "uncertainty_score_overall": {"score": 1, "level": "low"},
            "reassurance_score_management": {"score": 2, "level": "medium"},
            "analyst_skepticism_score": {"score": 1, "level": "low"},
            "strongest_evidence": {
                "uncertainty": [],
                "reassurance": [
                    {
                        "matched_phrase": "confident",
                        "text": "We remain confident in our plan.",
                        "strength": 0.7,
                    }
                ],
                "analyst_skepticism": [],
            },
        },
        qa_shift_summary={
            "prepared_remarks_vs_q_and_a": {"label": "mixed", "delta": 0.0},
            "analyst_skepticism": {"level": "low", "score": 0},
            "management_answers_vs_prepared_uncertainty": {"label": "mixed", "delta": 0.0},
            "early_vs_late_q_and_a": {"label": "mixed", "delta": 0.0},
            "counts": {"analyst_questions": 2, "management_answers": 2},
            "strongest_evidence": {
                "qa_pairs": [
                    {
                        "answer_minus_question": 0.01,
                        "question_text": "Any change to outlook?",
                        "answer_text": "No, we reaffirmed guidance.",
                    }
                ]
            },
        },
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
