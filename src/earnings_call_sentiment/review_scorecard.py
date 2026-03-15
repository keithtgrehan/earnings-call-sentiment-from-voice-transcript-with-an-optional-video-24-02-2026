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
_REVERSE_POLARITY_CATEGORIES = {
    "uncertainty",
    "analyst_skepticism",
    "qa_pressure_shift",
}
_UNCERTAINTY_CAUTION_PHRASES = (
    "difficult to predict",
    "hard to estimate",
    "subject to",
    "uncertainty",
    "visibility remains limited",
)


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


def _category_band(category_id: str, score: int) -> str:
    if category_id in {"uncertainty", "analyst_skepticism"}:
        if score <= 3:
            return "green"
        if score <= 6:
            return "amber"
        return "red"
    if category_id == "qa_pressure_shift":
        if score <= 3:
            return "green"
        if score <= 7:
            return "amber"
        return "red"
    if score >= 7:
        return "green"
    if score >= 4:
        return "amber"
    return "red"


def _desirability_score(category_id: str, score: int) -> int:
    if category_id in _REVERSE_POLARITY_CATEGORIES:
        return 11 - score
    return score


def _category_payload(
    *,
    category_id: str,
    score: int,
    explanation: str,
    strongest_evidence: list[str],
) -> dict[str, Any]:
    final_score = _clamp_score(score)
    return {
        "score": final_score,
        "color_band": _category_band(category_id, final_score),
        "explanation": explanation,
        "strongest_evidence": strongest_evidence[:2],
    }


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


def _qa_skepticism_evidence(qa_shift_summary: dict[str, Any]) -> list[str]:
    strongest = _as_dict(qa_shift_summary.get("strongest_evidence"))
    rows = _as_list(strongest.get("skepticism"))
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


def _infer_guidance_counts(guidance_revision: dict[str, Any]) -> dict[str, int]:
    labels = [
        str(_as_dict(item).get("label", "")).strip().lower()
        for item in _as_list(guidance_revision.get("top_revisions"))
    ]
    reaffirmed = _int_value(guidance_revision.get("reaffirmed_count"))
    maintained = max(
        _int_value(guidance_revision.get("maintained_count")),
        reaffirmed,
        sum(1 for label in labels if label == "maintained"),
    )
    return {
        "raised": _int_value(guidance_revision.get("raised_count")),
        "lowered": _int_value(guidance_revision.get("lowered_count")),
        "reaffirmed": reaffirmed,
        "maintained": maintained,
        "withdrawn": max(
            _int_value(guidance_revision.get("withdrawn_count")),
            sum(1 for label in labels if label == "withdrawn"),
        ),
        "unclear": _int_value(guidance_revision.get("unclear_count")),
        "mixed": _int_value(guidance_revision.get("mixed_count")),
    }


def _build_guidance_strength(
    *,
    guidance_df: pd.DataFrame,
    guidance_revision: dict[str, Any],
) -> dict[str, Any]:
    counts = _infer_guidance_counts(guidance_revision)
    matched = _int_value(guidance_revision.get("matched_count"))
    evidence = _guidance_evidence(guidance_revision, guidance_df)
    explicit_labels = counts["raised"] + counts["lowered"] + counts["maintained"] + counts["withdrawn"]

    if counts["withdrawn"] > 0 and counts["withdrawn"] >= max(
        counts["raised"],
        counts["lowered"],
        counts["maintained"],
        counts["unclear"],
        counts["mixed"],
    ):
        score = 1
        explanation = "Guidance looks withdrawn versus prior guidance, which is a clear weak-review outcome."
    elif counts["lowered"] > 0 and counts["lowered"] >= max(
        counts["raised"],
        counts["maintained"],
        counts["unclear"],
        counts["mixed"],
        counts["withdrawn"],
    ):
        score = 2
        explanation = "Guidance reads lower versus prior guidance, which is a clear negative revision signal."
    elif counts["raised"] > 0 and counts["raised"] > max(
        counts["lowered"],
        counts["maintained"],
        counts["unclear"],
        counts["mixed"],
        counts["withdrawn"],
    ):
        score = 9
        explanation = "Guidance reads raised versus prior guidance, which is a strong positive review signal."
    elif counts["maintained"] > 0 and counts["raised"] == 0 and counts["lowered"] == 0 and counts["withdrawn"] == 0:
        score = 6
        explanation = "Guidance reads maintained or reaffirmed, which keeps this category modestly constructive."
    elif counts["unclear"] > 0 or counts["mixed"] > 0:
        score = 5
        explanation = "Guidance revision evidence is unclear or mixed, so this category stays conservative."
    elif not guidance_df.empty:
        score = 5
        explanation = "Guidance language is present, but its direction is not explicit enough to score strongly."
    else:
        score = 5
        explanation = "Guidance evidence is limited, so this category stays neutral."

    if explicit_labels > 0 and matched > 0 and evidence and counts["unclear"] == 0 and counts["mixed"] == 0:
        score += 1
    elif not evidence or (explicit_labels == 0 and matched == 0):
        score -= 1

    return _category_payload(
        category_id="guidance_strength",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _build_uncertainty(summary: dict[str, Any]) -> dict[str, Any]:
    uncertainty = _as_dict(summary.get("uncertainty_score_overall"))
    level = str(uncertainty.get("level", "low")).strip().lower()
    evidence = _behavior_evidence(summary, "uncertainty")
    score = {
        "low": 2,
        "medium": 5,
        "high": 8,
    }.get(level, 5)
    if any(phrase in " ".join(evidence).lower() for phrase in _UNCERTAINTY_CAUTION_PHRASES):
        score += 1

    explanation = {
        "low": "Management language shows limited hedging, so uncertainty stays low.",
        "medium": "Management language contains a meaningful but not overwhelming amount of hedging.",
        "high": "Management language includes repeated hedging or visibility caveats, which raises uncertainty.",
    }.get(level, "Uncertainty evidence is limited, so this category stays conservative.")

    return _category_payload(
        category_id="uncertainty",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _build_analyst_skepticism(summary: dict[str, Any]) -> dict[str, Any]:
    skepticism = _as_dict(summary.get("analyst_skepticism_score"))
    level = str(skepticism.get("level", "low")).strip().lower()
    evidence = _behavior_evidence(summary, "analyst_skepticism")
    score = {
        "low": 3,
        "medium": 6,
        "high": 8,
    }.get(level, 6)
    if level in {"medium", "high"} and len(evidence) >= 2:
        score += 1

    explanation = {
        "low": "Analyst questions look relatively routine, with limited skeptical pressure.",
        "medium": "Analyst questions show some pushback or follow-up pressure, but not a full escalation.",
        "high": "Analyst questions show sustained skeptical pressure, which is a meaningful caution signal.",
    }.get(level, "Analyst skepticism evidence is limited, so this category stays conservative.")

    return _category_payload(
        category_id="analyst_skepticism",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _build_qa_pressure_shift(qa_shift_summary: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts.get("analyst_questions"))
    management_answers = _int_value(counts.get("management_answers"))
    qa_evidence = _qa_pair_evidence(qa_shift_summary)
    skepticism_evidence = _qa_skepticism_evidence(qa_shift_summary)
    evidence = (qa_evidence + skepticism_evidence)[:2]

    if analyst_questions == 0 or management_answers == 0 or min(analyst_questions, management_answers) < 2 or not qa_evidence:
        return _category_payload(
            category_id="qa_pressure_shift",
            score=5,
            explanation="Insufficient Q&A contrast evidence was available, so pressure shift stays neutral.",
            strongest_evidence=evidence,
        )

    prepared = _as_dict(qa_shift_summary.get("prepared_remarks_vs_q_and_a"))
    early_late = _as_dict(qa_shift_summary.get("early_vs_late_q_and_a"))
    skepticism = _as_dict(qa_shift_summary.get("analyst_skepticism"))
    answer_uncertainty = _as_dict(qa_shift_summary.get("management_answers_vs_prepared_uncertainty"))

    pressure_points = 0

    prepared_label = str(prepared.get("label", "mixed")).strip().lower()
    if prepared_label == "weaker":
        pressure_points += 2
    elif prepared_label == "stronger":
        pressure_points -= 1

    early_late_label = str(early_late.get("label", "mixed")).strip().lower()
    if early_late_label == "weaker":
        pressure_points += 1
    elif early_late_label == "stronger":
        pressure_points -= 1

    skeptic_level = str(skepticism.get("level", "low")).strip().lower()
    if skeptic_level == "medium":
        pressure_points += 1
    elif skeptic_level == "high":
        pressure_points += 2

    answer_uncertainty_label = str(answer_uncertainty.get("label", "mixed")).strip().lower()
    if answer_uncertainty_label == "more uncertain":
        pressure_points += 1
    elif answer_uncertainty_label == "less uncertain":
        pressure_points -= 1

    if pressure_points <= 0:
        score = 3
        explanation = "Q&A answers do not show meaningful deterioration versus prepared remarks."
    elif pressure_points == 1:
        score = 5
        explanation = "Q&A pressure rises somewhat, but the shift remains fairly mild."
    elif pressure_points <= 3:
        score = 7
        explanation = "Q&A answers weaken under questioning, indicating a moderate pressure shift."
    else:
        score = 9
        explanation = "Q&A answers deteriorate clearly under questioning, which is a strong review flag."

    return _category_payload(
        category_id="qa_pressure_shift",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _build_answer_directness(
    *,
    behavioral_summary: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    qa_pressure_score: int,
) -> dict[str, Any]:
    counts = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts.get("analyst_questions"))
    management_answers = _int_value(counts.get("management_answers"))
    evidence = (_qa_pair_evidence(qa_shift_summary) + _qa_answer_uncertainty_evidence(qa_shift_summary))[:2]

    if analyst_questions == 0 or management_answers == 0:
        return _category_payload(
            category_id="answer_directness",
            score=5,
            explanation="Q&A evidence is unavailable, so answer directness stays neutral by design.",
            strongest_evidence=evidence,
        )

    reassurance_level = str(
        _as_dict(behavioral_summary.get("reassurance_score_management")).get("level", "low")
    ).strip().lower()
    uncertainty_level = str(
        _as_dict(behavioral_summary.get("uncertainty_score_overall")).get("level", "low")
    ).strip().lower()
    skeptic_level = str(_as_dict(qa_shift_summary.get("analyst_skepticism")).get("level", "low")).strip().lower()
    answer_uncertainty_label = str(
        _as_dict(qa_shift_summary.get("management_answers_vs_prepared_uncertainty")).get("label", "mixed")
    ).strip().lower()

    score = 5
    if skeptic_level == "high" and qa_pressure_score >= 7:
        score -= 2
    elif skeptic_level == "medium" and qa_pressure_score >= 7:
        score -= 1

    if reassurance_level == "high" and qa_pressure_score <= 3:
        score += 2
    elif reassurance_level == "medium" and qa_pressure_score <= 5:
        score += 1

    if uncertainty_level == "high":
        score -= 1

    if answer_uncertainty_label == "more uncertain":
        score -= 1

    pair_rows = _as_list(_as_dict(qa_shift_summary.get("strongest_evidence")).get("qa_pairs"))
    if pair_rows:
        deltas = [_float_value(_as_dict(item).get("answer_minus_question")) for item in pair_rows]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta >= 0.08 and qa_pressure_score <= 5:
            score += 1
        elif avg_delta <= -0.08:
            score -= 1

    sparse_qa = min(analyst_questions, management_answers) < 2 or len(pair_rows) < 2
    if sparse_qa:
        score = min(score, 5)

    score = _clamp_score(score)
    if sparse_qa:
        explanation = "Q&A evidence is thin, so answer directness stays conservative rather than overstated."
    elif score >= 7:
        explanation = "Answers remain reasonably direct under questioning, with limited signs of evasion."
    elif score <= 3:
        explanation = "Answers look less direct under pressure, with more uncertainty or defensive drift."
    else:
        explanation = "Answer directness is mixed, so the Q&A excerpts still merit reviewer attention."

    return _category_payload(
        category_id="answer_directness",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _build_management_confidence(
    *,
    behavioral_summary: dict[str, Any],
    qa_pressure_score: int,
    answer_directness_score: int,
) -> dict[str, Any]:
    reassurance = _as_dict(behavioral_summary.get("reassurance_score_management"))
    uncertainty = _as_dict(behavioral_summary.get("uncertainty_score_overall"))
    reassurance_level = str(reassurance.get("level", "low")).strip().lower()
    uncertainty_level = str(uncertainty.get("level", "low")).strip().lower()

    score = 5
    score += {
        "low": 0,
        "medium": 2,
        "high": 4,
    }.get(reassurance_level, 0)

    if uncertainty_level == "medium":
        score -= 1
    elif uncertainty_level == "high":
        score -= 3

    if qa_pressure_score >= 9:
        score -= 2
    elif qa_pressure_score >= 7:
        score -= 1

    if answer_directness_score >= 7:
        score += 1
    elif answer_directness_score <= 3:
        score -= 1

    explanation = (
        f"Management confidence blends reassurance cues ({reassurance_level}) with uncertainty and Q&A follow-through."
    )
    evidence = _behavior_evidence(behavioral_summary, "reassurance")
    return _category_payload(
        category_id="management_confidence",
        score=score,
        explanation=explanation,
        strongest_evidence=evidence,
    )


def _signals_are_coherent(
    *,
    guidance_score: int,
    management_score: int,
    uncertainty_score: int,
    skepticism_score: int,
    qa_pressure_score: int,
) -> tuple[bool, bool]:
    positive_coherence = (
        guidance_score >= 7
        and management_score >= 6
        and uncertainty_score <= 4
        and skepticism_score <= 5
        and qa_pressure_score <= 5
    )
    negative_coherence = guidance_score <= 3 and (
        uncertainty_score >= 7 or skepticism_score >= 7 or qa_pressure_score >= 7 or management_score <= 4
    )
    material_conflict = (
        guidance_score >= 7 and (uncertainty_score >= 7 or skepticism_score >= 7 or qa_pressure_score >= 7)
    ) or (
        guidance_score <= 3 and management_score >= 7 and uncertainty_score <= 4 and skepticism_score <= 5
    )
    return positive_coherence or negative_coherence, material_conflict


def _confidence_pct(
    *,
    guidance_revision: dict[str, Any],
    guidance_df: pd.DataFrame,
    behavioral_summary: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    categories: list[dict[str, Any]],
) -> int:
    confidence = 70
    counts = _infer_guidance_counts(guidance_revision)
    explicit_guidance = counts["raised"] + counts["lowered"] + counts["maintained"] + counts["withdrawn"]
    guidance_unclear = counts["unclear"] > 0 or counts["mixed"] > 0 or (guidance_df.empty and explicit_guidance == 0)

    if explicit_guidance > 0:
        confidence += 10
    if guidance_unclear:
        confidence -= 15

    evidence_categories = sum(1 for item in categories if _as_list(item.get("strongest_evidence")))
    if evidence_categories >= 4:
        confidence += 5

    counts_payload = _as_dict(qa_shift_summary.get("counts"))
    analyst_questions = _int_value(counts_payload.get("analyst_questions"))
    management_answers = _int_value(counts_payload.get("management_answers"))
    if analyst_questions == 0 or management_answers == 0 or min(analyst_questions, management_answers) < 2:
        confidence -= 10

    guidance_score = int(next(item["score"] for item in categories if item["id"] == "guidance_strength"))
    management_score = int(next(item["score"] for item in categories if item["id"] == "management_confidence"))
    uncertainty_score = int(next(item["score"] for item in categories if item["id"] == "uncertainty"))
    skepticism_score = int(next(item["score"] for item in categories if item["id"] == "analyst_skepticism"))
    qa_pressure_score = int(next(item["score"] for item in categories if item["id"] == "qa_pressure_shift"))

    coherent, conflicting = _signals_are_coherent(
        guidance_score=guidance_score,
        management_score=management_score,
        uncertainty_score=uncertainty_score,
        skepticism_score=skepticism_score,
        qa_pressure_score=qa_pressure_score,
    )
    if coherent:
        confidence += 5
    if conflicting:
        confidence -= 10

    strongest = _as_dict(behavioral_summary.get("strongest_evidence"))
    behavior_evidence_count = sum(len(_as_list(strongest.get(key))) for key in strongest)
    qa_evidence_count = len(_as_list(_as_dict(qa_shift_summary.get("strongest_evidence")).get("qa_pairs")))
    excerpt_heavy_only = behavior_evidence_count == 0 and qa_evidence_count == 0 and bool(guidance_revision.get("top_revisions"))
    if excerpt_heavy_only:
        confidence -= 10

    all_snippets: list[str] = []
    for item in categories:
        all_snippets.extend(str(snippet) for snippet in _as_list(item.get("strongest_evidence")))
    unique_snippets = {snippet.strip() for snippet in all_snippets if snippet.strip()}
    if evidence_categories <= 2 or len(unique_snippets) <= 2 or len(unique_snippets) < len(all_snippets):
        confidence -= 5

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
    qa_pressure_category = _build_qa_pressure_shift(qa_shift_summary)
    directness_category = _build_answer_directness(
        behavioral_summary=behavioral_summary,
        qa_shift_summary=qa_shift_summary,
        qa_pressure_score=int(qa_pressure_category["score"]),
    )
    management_category = _build_management_confidence(
        behavioral_summary=behavioral_summary,
        qa_pressure_score=int(qa_pressure_category["score"]),
        answer_directness_score=int(directness_category["score"]),
    )

    raw_categories = {
        "guidance_strength": guidance_category,
        "management_confidence": management_category,
        "uncertainty": uncertainty_category,
        "analyst_skepticism": skepticism_category,
        "qa_pressure_shift": qa_pressure_category,
        "answer_directness": directness_category,
    }
    categories: list[dict[str, Any]] = []
    for key in _CATEGORY_ORDER:
        payload = dict(raw_categories[key])
        payload["id"] = key
        payload["name"] = _CATEGORY_NAMES[key]
        categories.append(payload)

    overall_score = round(
        sum(
            float(_desirability_score(str(item["id"]), int(item["score"]))) * _CATEGORY_WEIGHTS[str(item["id"])]
            for item in categories
        ),
        1,
    )

    guidance_score = int(guidance_category["score"])
    management_score = int(management_category["score"])
    uncertainty_score = int(uncertainty_category["score"])
    skepticism_score = int(skepticism_category["score"])
    qa_pressure_score = int(qa_pressure_category["score"])
    red_count = sum(1 for item in categories if str(item.get("color_band")) == "red")

    if (
        guidance_score >= 7
        and management_score >= 6
        and uncertainty_score <= 4
        and skepticism_score <= 5
        and qa_pressure_score <= 5
    ):
        overall_review_signal = "green"
    elif (
        guidance_score <= 3
        or red_count >= 3
        or (uncertainty_score >= 7 and skepticism_score >= 7 and qa_pressure_score >= 7)
    ):
        overall_review_signal = "red"
    else:
        overall_review_signal = "amber"

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
                key=lambda row: (
                    -abs(_desirability_score(str(row["id"]), int(row["score"])) - 5.5),
                    _CATEGORY_ORDER.index(str(row["id"])),
                ),
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
