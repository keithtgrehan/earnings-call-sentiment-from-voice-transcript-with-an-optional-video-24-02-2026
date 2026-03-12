"""Reviewer-friendly presentation mapping for deterministic artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

REVIEW_SCORECARD_SCHEMA_VERSION = "1.0.0"
_CATEGORY_ORDER = (
    "guidance_strength",
    "management_confidence",
    "uncertainty",
    "analyst_skepticism",
    "qa_pressure_shift",
    "answer_directness",
)
_CATEGORY_NAMES = {
    "guidance_strength": "Guidance Strength",
    "management_confidence": "Management Confidence",
    "uncertainty": "Uncertainty / Hedging",
    "analyst_skepticism": "Analyst Skepticism",
    "qa_pressure_shift": "Q&A Pressure Shift",
    "answer_directness": "Answer Directness",
}
_CATEGORY_WEIGHTS = {
    "guidance_strength": 0.24,
    "management_confidence": 0.18,
    "uncertainty": 0.16,
    "analyst_skepticism": 0.14,
    "qa_pressure_shift": 0.14,
    "answer_directness": 0.14,
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(text: str, limit: int = 140) -> str:
    compact = " ".join(str(text).replace("\n", " ").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _clamp_score(score: int) -> int:
    return max(1, min(10, int(score)))


def _band_from_score(score: int) -> str:
    if score >= 7:
        return "green"
    if score >= 4:
        return "amber"
    return "red"


def _level_to_positive_score(level: str) -> int:
    normalized = str(level).strip().lower()
    if normalized == "high":
        return 8
    if normalized == "medium":
        return 6
    return 4


def _level_to_reverse_score(level: str, *, zero_case: bool = False) -> int:
    normalized = str(level).strip().lower()
    if normalized == "high":
        return 2
    if normalized == "medium":
        return 5
    return 9 if zero_case else 8


def _behavior_evidence(summary: dict[str, Any], key: str) -> list[str]:
    strongest = _as_dict(summary.get("strongest_evidence"))
    rows = _as_list(strongest.get(key))
    snippets: list[str] = []
    for item in rows[:2]:
        payload = _as_dict(item)
        phrase = str(payload.get("matched_phrase", "")).strip()
        text = _clip(str(payload.get("text", "")), limit=150)
        if phrase and text:
            snippets.append(f"[{phrase}] {text}")
        elif text:
            snippets.append(text)
    return snippets


def _guidance_evidence(guidance_revision: dict[str, Any], guidance_df: pd.DataFrame) -> list[str]:
    revisions = _as_list(guidance_revision.get("top_revisions"))
    snippets: list[str] = []
    for item in revisions[:2]:
        payload = _as_dict(item)
        label = str(payload.get("label", "unclear")).strip()
        snippet = _clip(str(payload.get("snippet", "")), limit=150)
        if snippet:
            snippets.append(f"{label}: {snippet}")
    if snippets:
        return snippets

    if guidance_df.empty:
        return []

    for _, row in guidance_df.head(2).iterrows():
        snippet = _clip(str(row.get("text", "")), limit=150)
        if snippet:
            snippets.append(snippet)
    return snippets


def _qa_pair_evidence(qa_shift_summary: dict[str, Any]) -> list[str]:
    strongest = _as_dict(qa_shift_summary.get("strongest_evidence"))
    pairs = _as_list(strongest.get("qa_pairs"))
    snippets: list[str] = []
    for item in pairs[:2]:
        payload = _as_dict(item)
        delta = _float_value(payload.get("answer_minus_question"))
        question = _clip(str(payload.get("question_text", "")), limit=72)
        answer = _clip(str(payload.get("answer_text", "")), limit=72)
        if question or answer:
            snippets.append(f"delta={delta:+.2f} | Q: {question} | A: {answer}")
    return snippets


def _qa_answer_uncertainty_evidence(qa_shift_summary: dict[str, Any]) -> list[str]:
    strongest = _as_dict(qa_shift_summary.get("strongest_evidence"))
    rows = _as_list(strongest.get("answer_uncertainty"))
    snippets: list[str] = []
    for item in rows[:2]:
        payload = _as_dict(item)
        phrase = str(payload.get("matched_phrase", "")).strip()
        text = _clip(str(payload.get("text", "")), limit=150)
        if phrase and text:
            snippets.append(f"[{phrase}] {text}")
        elif text:
            snippets.append(text)
    return snippets


def _build_guidance_strength(
    *,
    guidance_df: pd.DataFrame,
    guidance_revision: dict[str, Any],
) -> dict[str, Any]:
    matched = _int_value(guidance_revision.get("matched_count"))
    raised = _int_value(guidance_revision.get("raised_count"))
    lowered = _int_value(guidance_revision.get("lowered_count"))
    reaffirmed = _int_value(guidance_revision.get("reaffirmed_count"))
    unclear = _int_value(guidance_revision.get("unclear_count"))
    mixed = _int_value(guidance_revision.get("mixed_count"))
    has_prior = bool(guidance_revision.get("prior_guidance_path"))

    if raised > max(lowered, reaffirmed, mixed, unclear):
        score = 9 if lowered == 0 and unclear == 0 and mixed == 0 else 8
        explanation = (
            f"Guidance reads stronger versus prior guidance, with {raised} raised item(s) "
            "and no stronger negative offset."
        )
    elif lowered > max(raised, reaffirmed, mixed, unclear):
        score = 2 if raised == 0 else 3
        explanation = (
            f"Guidance reads weaker versus prior guidance, with {lowered} lowered item(s) "
            "driving the score."
        )
    elif reaffirmed > 0 and raised == 0 and lowered == 0:
        score = 6
        explanation = "Guidance looks maintained versus prior guidance, which keeps this category neutral-positive."
    elif mixed > 0:
        score = 5
        explanation = "Guidance revision evidence is mixed across matched items, so this category stays neutral."
    elif has_prior and matched == 0:
        score = 5
        explanation = "Guidance language is present, but prior guidance could not be matched cleanly enough to rank it strongly."
    elif len(guidance_df) > 0:
        score = 6 if not has_prior else 5
        explanation = (
            "Guidance language is present in the call, but the direction versus prior guidance remains conservative or unavailable."
        )
    else:
        score = 5
        explanation = "Guidance evidence is limited, so the score stays neutral."

    return {
        "score": _clamp_score(score),
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": _guidance_evidence(guidance_revision, guidance_df),
    }


def _build_uncertainty(summary: dict[str, Any]) -> dict[str, Any]:
    uncertainty = _as_dict(summary.get("uncertainty_score_overall"))
    raw_score = _int_value(uncertainty.get("score"))
    level = str(uncertainty.get("level", "low")).lower()
    score = _level_to_reverse_score(level, zero_case=raw_score == 0)
    explanation = {
        "low": "Few hedging cues were detected in management remarks, so uncertainty stays reviewer-friendly.",
        "medium": "Management language contains a moderate amount of hedging or conditional phrasing.",
        "high": "Management language shows repeated hedging or visibility caveats, which is a clear caution signal.",
    }.get(level, "Uncertainty evidence is limited, so this category stays neutral.")
    return {
        "score": score,
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": _behavior_evidence(summary, "uncertainty"),
    }


def _build_analyst_skepticism(summary: dict[str, Any]) -> dict[str, Any]:
    skepticism = _as_dict(summary.get("analyst_skepticism_score"))
    level = str(skepticism.get("level", "low")).lower()
    raw_score = _int_value(skepticism.get("score"))
    score = _level_to_reverse_score(level, zero_case=raw_score == 0)
    explanation = {
        "low": "Analyst questions read relatively routine, with limited skeptical pressure cues.",
        "medium": "Analyst questions show some pushback or probing follow-up, but not sustained pressure.",
        "high": "Analyst questions show strong skeptical pressure, which is a meaningful review flag.",
    }.get(level, "Analyst skepticism evidence is limited, so this category stays neutral.")
    return {
        "score": score,
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": _behavior_evidence(summary, "analyst_skepticism"),
    }


def _build_answer_directness(qa_shift_summary: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts.get("analyst_questions"))
    management_answers = _int_value(counts.get("management_answers"))
    has_qa = analyst_questions > 0 and management_answers > 0
    prepared_vs_qa = _as_dict(qa_shift_summary.get("prepared_remarks_vs_q_and_a"))
    answer_uncertainty = _as_dict(qa_shift_summary.get("management_answers_vs_prepared_uncertainty"))
    analyst_skepticism = _as_dict(qa_shift_summary.get("analyst_skepticism"))
    pair_rows = _as_list(_as_dict(qa_shift_summary.get("strongest_evidence")).get("qa_pairs"))

    if not has_qa:
        return {
            "score": 5,
            "color_band": "amber",
            "explanation": "Q&A evidence is sparse, so answer directness stays neutral by design.",
            "strongest_evidence": [],
        }

    score = 6
    skeptic_level = str(analyst_skepticism.get("level", "low")).lower()
    if skeptic_level == "low":
        score += 1
    elif skeptic_level == "high":
        score -= 2
    else:
        score -= 1

    uncertainty_label = str(answer_uncertainty.get("label", "mixed")).lower()
    if uncertainty_label == "less uncertain":
        score += 1
    elif uncertainty_label == "more uncertain":
        score -= 2

    prepared_label = str(prepared_vs_qa.get("label", "mixed")).lower()
    if prepared_label == "stronger":
        score += 1
    elif prepared_label == "weaker":
        score -= 1

    if pair_rows:
        deltas = [_float_value(_as_dict(item).get("answer_minus_question")) for item in pair_rows]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta >= 0.08:
            score += 1
        elif avg_delta <= -0.08:
            score -= 1

    if min(analyst_questions, management_answers) < 2:
        score = round((score + 5) / 2)

    score = _clamp_score(score)
    if score >= 7:
        explanation = "Answers hold up reasonably well under questioning, so directness looks solid."
    elif score <= 3:
        explanation = "Answers look less direct under pressure, with more uncertainty or weaker answer tone."
    else:
        explanation = "Answer directness is mixed, so reviewers should verify the Q&A excerpts directly."

    evidence = _qa_pair_evidence(qa_shift_summary)
    evidence.extend(_qa_answer_uncertainty_evidence(qa_shift_summary))
    return {
        "score": score,
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": evidence[:2],
    }


def _build_management_confidence(
    *,
    behavioral_summary: dict[str, Any],
    answer_directness_score: int,
) -> dict[str, Any]:
    reassurance = _as_dict(behavioral_summary.get("reassurance_score_management"))
    uncertainty = _as_dict(behavioral_summary.get("uncertainty_score_overall"))
    reassurance_level = str(reassurance.get("level", "low")).lower()
    uncertainty_level = str(uncertainty.get("level", "low")).lower()

    score = _level_to_positive_score(reassurance_level)
    if answer_directness_score >= 7:
        score += 1
    elif answer_directness_score <= 3:
        score -= 1

    if uncertainty_level == "high":
        score -= 2
    elif uncertainty_level == "medium":
        score -= 1

    score = _clamp_score(score)
    explanation = (
        f"Management confidence blends reassurance cues ({reassurance_level}) with answer directness evidence "
        "rather than relying on sentiment alone."
    )
    evidence = _behavior_evidence(behavioral_summary, "reassurance")
    return {
        "score": score,
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": evidence[:2],
    }


def _build_qa_pressure_shift(qa_shift_summary: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts.get("analyst_questions"))
    management_answers = _int_value(counts.get("management_answers"))
    has_qa = analyst_questions > 0 and management_answers > 0

    if not has_qa:
        return {
            "score": 5,
            "color_band": "amber",
            "explanation": "Q&A evidence is limited, so pressure shift stays neutral.",
            "strongest_evidence": [],
        }

    prepared = _as_dict(qa_shift_summary.get("prepared_remarks_vs_q_and_a"))
    early_late = _as_dict(qa_shift_summary.get("early_vs_late_q_and_a"))
    skepticism = _as_dict(qa_shift_summary.get("analyst_skepticism"))
    answer_uncertainty = _as_dict(qa_shift_summary.get("management_answers_vs_prepared_uncertainty"))

    score = 6
    prepared_label = str(prepared.get("label", "mixed")).lower()
    if prepared_label == "stronger":
        score += 1
    elif prepared_label == "weaker":
        score -= 2

    early_late_label = str(early_late.get("label", "mixed")).lower()
    if early_late_label == "stronger":
        score += 1
    elif early_late_label == "weaker":
        score -= 1

    skeptic_level = str(skepticism.get("level", "low")).lower()
    if skeptic_level == "high":
        score -= 2
    elif skeptic_level == "medium":
        score -= 1

    uncertainty_label = str(answer_uncertainty.get("label", "mixed")).lower()
    if uncertainty_label == "less uncertain":
        score += 1
    elif uncertainty_label == "more uncertain":
        score -= 1

    if min(analyst_questions, management_answers) < 2:
        score = round((score + 5) / 2)

    score = _clamp_score(score)
    explanation = (
        "Q&A pressure shift combines answer-tone drift, analyst skepticism, and whether answers become more uncertain."
    )
    evidence = _qa_pair_evidence(qa_shift_summary)
    skepticism_evidence = _as_list(_as_dict(qa_shift_summary.get("strongest_evidence")).get("skepticism"))
    for item in skepticism_evidence[:1]:
        payload = _as_dict(item)
        phrase = str(payload.get("matched_phrase", "")).strip()
        text = _clip(str(payload.get("text", "")), limit=150)
        if phrase and text:
            evidence.append(f"[{phrase}] {text}")
    return {
        "score": score,
        "color_band": _band_from_score(score),
        "explanation": explanation,
        "strongest_evidence": evidence[:2],
    }


def _confidence_pct(
    *,
    guidance_revision: dict[str, Any],
    guidance_df: pd.DataFrame,
    behavioral_summary: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    categories: list[dict[str, Any]],
) -> int:
    confidence = 52
    matched = _int_value(guidance_revision.get("matched_count"))
    explicit_guidance = (
        _int_value(guidance_revision.get("raised_count"))
        + _int_value(guidance_revision.get("lowered_count"))
        + _int_value(guidance_revision.get("reaffirmed_count"))
    )
    unclear = _int_value(guidance_revision.get("unclear_count"))
    mixed = _int_value(guidance_revision.get("mixed_count"))

    if matched > 0:
        confidence += 14
    elif len(guidance_df) > 0:
        confidence += 6
    else:
        confidence -= 6

    if explicit_guidance > 0:
        confidence += 8
    elif unclear > 0 or mixed > 0:
        confidence -= 6

    strongest = _as_dict(behavioral_summary.get("strongest_evidence"))
    behavior_evidence_count = sum(len(_as_list(strongest.get(key))) for key in strongest)
    if behavior_evidence_count >= 4:
        confidence += 8
    elif behavior_evidence_count >= 2:
        confidence += 4
    else:
        confidence -= 6

    counts = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts.get("analyst_questions"))
    management_answers = _int_value(counts.get("management_answers"))
    if analyst_questions > 0 and management_answers > 0:
        confidence += 8
    else:
        confidence -= 10

    qa_pairs = len(_as_list(_as_dict(qa_shift_summary.get("strongest_evidence")).get("qa_pairs")))
    if qa_pairs > 0:
        confidence += 4
    else:
        confidence -= 4

    uncertainty_level = str(_as_dict(behavioral_summary.get("uncertainty_score_overall")).get("level", "low")).lower()
    reassurance_level = str(_as_dict(behavioral_summary.get("reassurance_score_management")).get("level", "low")).lower()
    if uncertainty_level == "high" and reassurance_level == "high":
        confidence -= 8

    green_count = sum(1 for item in categories if str(item.get("color_band")) == "green")
    red_count = sum(1 for item in categories if str(item.get("color_band")) == "red")
    if green_count >= 4 or red_count >= 3:
        confidence += 4

    return max(35, min(95, int(round(confidence))))


def build_review_scorecard(
    *,
    guidance_df: pd.DataFrame,
    guidance_revision: dict[str, Any],
    behavioral_summary: dict[str, Any],
    qa_shift_summary: dict[str, Any],
) -> dict[str, Any]:
    guidance_category = _build_guidance_strength(
        guidance_df=guidance_df,
        guidance_revision=guidance_revision,
    )
    uncertainty_category = _build_uncertainty(behavioral_summary)
    skepticism_category = _build_analyst_skepticism(behavioral_summary)
    directness_category = _build_answer_directness(qa_shift_summary)
    management_category = _build_management_confidence(
        behavioral_summary=behavioral_summary,
        answer_directness_score=int(directness_category["score"]),
    )
    qa_pressure_category = _build_qa_pressure_shift(qa_shift_summary)

    categories: list[dict[str, Any]] = []
    raw_categories = {
        "guidance_strength": guidance_category,
        "management_confidence": management_category,
        "uncertainty": uncertainty_category,
        "analyst_skepticism": skepticism_category,
        "qa_pressure_shift": qa_pressure_category,
        "answer_directness": directness_category,
    }
    for key in _CATEGORY_ORDER:
        payload = dict(raw_categories[key])
        payload["id"] = key
        payload["name"] = _CATEGORY_NAMES[key]
        categories.append(payload)

    overall_score = round(
        sum(float(item["score"]) * _CATEGORY_WEIGHTS[item["id"]] for item in categories),
        1,
    )
    overall_review_signal = _band_from_score(int(round(overall_score)))
    if overall_score < 5.5 and sum(1 for item in categories if int(item["score"]) <= 3) >= 2:
        overall_review_signal = "red"

    review_confidence_pct = _confidence_pct(
        guidance_revision=guidance_revision,
        guidance_df=guidance_df,
        behavioral_summary=behavioral_summary,
        qa_shift_summary=qa_shift_summary,
        categories=categories,
    )

    ranked_categories = [
        {
            **item,
            "rank": index + 1,
        }
        for index, item in enumerate(
            sorted(
                categories,
                key=lambda row: (-int(row["score"]), _CATEGORY_ORDER.index(str(row["id"]))),
            )
        )
    ]

    flat_fields = {
        "guidance_strength_score": int(guidance_category["score"]),
        "guidance_strength_band": str(guidance_category["color_band"]),
        "management_confidence_score": int(management_category["score"]),
        "management_confidence_band": str(management_category["color_band"]),
        "uncertainty_score": int(uncertainty_category["score"]),
        "uncertainty_band": str(uncertainty_category["color_band"]),
        "analyst_skepticism_score": int(skepticism_category["score"]),
        "analyst_skepticism_band": str(skepticism_category["color_band"]),
        "qa_pressure_shift_score": int(qa_pressure_category["score"]),
        "qa_pressure_shift_band": str(qa_pressure_category["color_band"]),
        "answer_directness_score": int(directness_category["score"]),
        "answer_directness_band": str(directness_category["color_band"]),
    }

    return {
        "schema_version": REVIEW_SCORECARD_SCHEMA_VERSION,
        "overall_review_signal": overall_review_signal,
        "overall_score": overall_score,
        "review_confidence_pct": review_confidence_pct,
        "confidence_note": (
            "Confidence reflects how clear and well-supported the deterministic interpretation is, "
            "not investment conviction."
        ),
        **flat_fields,
        "categories": categories,
        "ranked_categories": ranked_categories,
    }
