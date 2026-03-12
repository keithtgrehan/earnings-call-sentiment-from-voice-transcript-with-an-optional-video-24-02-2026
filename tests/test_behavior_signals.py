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


def test_uncertainty_does_not_fire_on_plain_guidance_expectation() -> None:
    payload = compute_behavioral_outputs(
        _chunks("We expect revenue to be between $10 billion and $11 billion next quarter.")
    )
    assert payload["uncertainty_df"].empty


def test_uncertainty_covers_explicit_prediction_difficulty_phrases() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "It's difficult to predict when supply and demand will balance.",
            "It's hard to estimate with precision what the demand will be.",
            "I wouldn't want\nto predict how the market reacts in the future.",
        )
    )
    phrases = set(payload["uncertainty_df"]["matched_phrase"])
    assert "difficult to predict" in phrases
    assert "hard to estimate with precision" in phrases
    assert "wouldn't want to predict" in phrases
    assert payload["summary"]["uncertainty_score_overall"]["level"] == "high"


def test_uncertainty_covers_strong_condition_and_uncertainty_nouns() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "This portion of the contract is subject to final resolution.",
            "Balanced against economic uncertainties and project timing movement, we are maintaining guidance.",
        )
    )
    phrases = set(payload["uncertainty_df"]["matched_phrase"])
    assert "subject to final resolution" in phrases
    assert "macro uncertainty" in phrases
    assert payload["summary"]["uncertainty_score_overall"]["level"] == "high"


def test_uncertainty_covers_persistent_supply_tightness_caution() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "While we expect tightness in the supply for our advanced architectures to persist, we remain confident in execution."
        )
    )
    row = payload["uncertainty_df"].iloc[0]
    assert row["matched_phrase"] == "tightness in the supply to persist"
    assert payload["summary"]["uncertainty_score_overall"]["level"] == "medium"


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


def test_skepticism_covers_light_probe_stems() -> None:
    payload = compute_behavioral_outputs(
        _chunks("What's driving the strength over here?", "Could you just talk a little bit about the partner motion?")
    )
    labels = list(payload["skepticism_df"]["skepticism_label"])
    assert labels == ["low", "low"]


def test_skepticism_covers_medium_probe_stems() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "Can you help us understand maybe what the revenue upside potential is with AI?",
            "Are you seeing any cannibalization of search as far as activity as people start using that app more?",
            "And are you confident you've reserved sufficient data center capacity to support the rollout?",
        )
    )
    assert list(payload["skepticism_df"]["skepticism_label"]) == ["medium", "medium", "medium"]


def test_skepticism_covers_high_confidence_and_sustainability_challenges() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "But what gives you confidence that both brands still have good momentum with consumers?",
            "How sustainable do you think those are?",
            "What could be things that you could do to reverse that in future quarters?",
        )
    )
    assert list(payload["skepticism_df"]["skepticism_label"]) == ["high", "high", "high"]


def test_skepticism_does_not_flag_plain_information_request() -> None:
    payload = compute_behavioral_outputs(
        _chunks("Could you walk us through the sequence of prepared remarks before we move to Q&A?")
    )
    assert payload["skepticism_df"].empty


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


def test_operator_line_does_not_count_as_management_uncertainty() -> None:
    payload = compute_behavioral_outputs(_chunks("You may begin."))
    assert payload["uncertainty_df"].empty


def test_question_sentence_does_not_count_as_management_uncertainty() -> None:
    payload = compute_behavioral_outputs(
        _chunks("I was just wondering if you could talk about demand trends?")
    )
    assert payload["uncertainty_df"].empty


def test_boilerplate_forward_looking_disclaimer_does_not_dominate_uncertainty() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "These statements are forward-looking statements and actual results may differ materially as described in our SEC filings."
        )
    )
    assert payload["uncertainty_df"].empty


def test_skepticism_uses_matched_sentence_not_full_segment_blob() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "What changed in the marketplace? Thank you for the question. We remain confident in the trajectory."
        )
    )
    row = payload["skepticism_df"].iloc[0]
    assert row["text"] == "What changed in the marketplace?"


def test_operator_replay_and_disconnect_lines_do_not_count_as_uncertainty() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "You may access the replay system at any time.",
            "You may now disconnect your lines.",
            "You may press star two if you would like to remove your question from the queue.",
        )
    )
    assert payload["uncertainty_df"].empty


def test_question_stems_do_not_leak_into_management_uncertainty() -> None:
    payload = compute_behavioral_outputs(
        _chunks(
            "Where do you think this could stabilize?",
            "I'm curious if this is something we should think about as weaker.",
            "Maybe you could just talk a little bit about that.",
            "I have two as well if I could.",
            "In that vein, I'd like to just dig down on something that might be a really obvious question.",
            "And what that might mean for adherence in the real world.",
            "And I think you could look at that performance.",
        )
    )
    assert payload["uncertainty_df"].empty


def test_could_not_be_more_excited_does_not_count_as_uncertainty() -> None:
    payload = compute_behavioral_outputs(_chunks("We could not be more excited about the launch."))
    assert payload["uncertainty_df"].empty


def test_newline_split_question_stem_does_not_count_as_uncertainty() -> None:
    payload = compute_behavioral_outputs(_chunks("And I think you\ncould look at that performance."))
    assert payload["uncertainty_df"].empty


def test_you_might_have_already_seen_does_not_count_as_uncertainty() -> None:
    payload = compute_behavioral_outputs(_chunks("You might have already seen that we launched the campaign yesterday."))
    assert payload["uncertainty_df"].empty
