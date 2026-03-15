from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.media_quality import build_media_quality_summary
from earnings_call_sentiment.media_support_eval import repo_root
from earnings_call_sentiment.multimodal_support import build_multimodal_support_summary


def aggregate_support_target(
    labels: pd.DataFrame,
    *,
    direction_column: str = "multimodal_support_direction",
    direction_threshold: float = 0.2,
) -> dict[str, Any]:
    counts = {
        str(key): int(value)
        for key, value in labels[direction_column].astype(str).value_counts().to_dict().items()
    }
    supportive = counts.get("supportive", 0)
    cautionary = counts.get("cautionary", 0)
    neutral = counts.get("neutral", 0)
    denominator = supportive + cautionary + neutral
    signed_mean = 0.0 if denominator == 0 else float(cautionary - supportive) / float(denominator)
    if signed_mean >= direction_threshold:
        direction = "cautionary"
    elif signed_mean <= -direction_threshold:
        direction = "supportive"
    else:
        direction = "neutral"
    return {
        "counts": counts,
        "target_support_signed_mean": round(signed_mean, 4),
        "target_support_direction": direction,
    }


def classify_support_score(score: float, *, neutral_band: float = 0.03) -> str:
    if score >= neutral_band:
        return "cautionary"
    if score <= -neutral_band:
        return "supportive"
    return "neutral"


def _load_json(path_value: str | float | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = repo_root() / path
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_optional_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def recompute_multimodal_case(case_row: dict[str, Any]) -> dict[str, Any]:
    metrics_payload = _load_json(case_row.get("metrics_path")) or {}
    qa_shift_summary = _load_json(case_row.get("qa_shift_path")) or {}
    audio_summary = _load_json(case_row.get("audio_summary_path"))
    visual_summary = _load_json(case_row.get("visual_summary_path"))
    media_quality = build_media_quality_summary(
        audio_summary=audio_summary,
        visual_summary=visual_summary,
    )
    summary = build_multimodal_support_summary(
        metrics_payload=metrics_payload,
        qa_shift_summary=qa_shift_summary,
        audio_summary=audio_summary,
        visual_summary=visual_summary,
        media_quality=media_quality,
    )
    return {
        "summary": summary,
        "media_quality": media_quality,
    }


def evaluate_downstream_decision_cases(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        case_dict = case.to_dict()
        recomputed = recompute_multimodal_case(case_dict)
        current_summary = recomputed["summary"]
        current_score = float(current_summary.get("calibrated_support_score", 0.0))
        current_direction = classify_support_score(current_score)

        baseline_score = 0.0
        baseline_direction = "neutral"

        saved_score = None
        saved_direction = None
        saved_path = str(case_dict.get("saved_multimodal_summary_path", "")).strip()
        if saved_path:
            saved_summary = _load_json(saved_path) or {}
            saved_score = float(saved_summary.get("calibrated_support_score", 0.0))
            saved_direction = classify_support_score(saved_score)

        target_score = _coerce_optional_float(case_dict.get("target_support_signed_mean"))
        target_direction = str(case_dict.get("target_support_direction", "")).strip()
        has_support_target = bool(target_direction) and target_score is not None

        rows.append(
            {
                **case_dict,
                "has_support_target": has_support_target,
                "baseline_direction": baseline_direction,
                "baseline_score": baseline_score,
                "baseline_match": "" if not has_support_target else baseline_direction == target_direction,
                "baseline_abs_error": "" if not has_support_target else round(abs(target_score - baseline_score), 4),
                "current_direction": current_direction,
                "current_score": round(current_score, 4),
                "current_match": "" if not has_support_target else current_direction == target_direction,
                "current_abs_error": "" if not has_support_target else round(abs(target_score - current_score), 4),
                "saved_direction": saved_direction or "",
                "saved_score": "" if saved_score is None else round(saved_score, 4),
                "saved_match": ""
                if (saved_direction is None or not has_support_target)
                else str(saved_direction == target_direction).lower(),
                "saved_abs_error": ""
                if (saved_score is None or not has_support_target)
                else round(abs(target_score - saved_score), 4),
                "audio_support_direction": current_summary.get("audio_support_direction", "unavailable"),
                "video_support_direction": current_summary.get("video_support_direction", "unavailable"),
                "fusion_mode": current_summary.get("fusion_mode", "heuristic_fallback"),
                "multimodal_alignment": current_summary.get("multimodal_alignment", "low"),
                "multimodal_confidence_adjustment": int(current_summary.get("multimodal_confidence_adjustment", 0)),
                "audio_quality_ok": bool(recomputed["media_quality"].get("audio_quality_ok", False)),
                "video_quality_ok": bool(recomputed["media_quality"].get("video_quality_ok", False)),
            }
        )

    result_frame = pd.DataFrame(rows)
    comparable = result_frame[result_frame["has_support_target"]].copy()

    summary: dict[str, Any] = {
        "case_count": int(len(result_frame)),
        "case_count_with_support_targets": int(len(comparable)),
        "case_count_without_support_targets": int(len(result_frame) - len(comparable)),
        "transcript_only_baseline": {
            "label_accuracy": round(float(comparable["baseline_match"].mean()), 4) if not comparable.empty else 0.0,
            "mean_abs_error_vs_target": round(float(comparable["baseline_abs_error"].mean()), 4)
            if not comparable.empty
            else 0.0,
        },
        "current_conservative_multimodal": {
            "label_accuracy": round(float(comparable["current_match"].mean()), 4) if not comparable.empty else 0.0,
            "mean_abs_error_vs_target": round(float(comparable["current_abs_error"].mean()), 4)
            if not comparable.empty
            else 0.0,
        },
    }

    if "gold_guidance_label" in result_frame.columns:
        summary["gold_guidance_label_distribution"] = {
            str(key): int(value)
            for key, value in result_frame["gold_guidance_label"].astype(str).value_counts().to_dict().items()
        }

    comparable_saved = comparable[comparable["saved_direction"].astype(str).str.strip() != ""].copy()
    if not comparable_saved.empty:
        comparable_saved["saved_match_bool"] = comparable_saved["saved_match"].astype(str).str.lower() == "true"
        comparable_saved["saved_abs_error_float"] = comparable_saved["saved_abs_error"].astype(float)
        summary["legacy_saved_multimodal"] = {
            "case_count": int(len(comparable_saved)),
            "label_accuracy": round(float(comparable_saved["saved_match_bool"].mean()), 4),
            "mean_abs_error_vs_target": round(float(comparable_saved["saved_abs_error_float"].mean()), 4),
        }

    return result_frame, summary


def summarize_task_impact_results(cases: pd.DataFrame, results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {
            "submission_count": 0,
            "conditions": {},
            "notes": ["No participant submissions were recorded yet."],
        }

    merged = results.merge(
        cases[["case_id", "gold_guidance_label"]],
        on="case_id",
        how="left",
    )
    merged["label_match"] = merged["predicted_guidance_label"].astype(str) == merged["gold_guidance_label"].astype(str)

    conditions: dict[str, Any] = {}
    for condition, group in merged.groupby("condition"):
        conditions[str(condition)] = {
            "submission_count": int(len(group)),
            "mean_completion_seconds": round(float(pd.to_numeric(group["completion_seconds"], errors="coerce").mean()), 2),
            "label_accuracy": round(float(group["label_match"].mean()), 4),
            "mean_clarity_rating": round(float(pd.to_numeric(group["clarity_rating"], errors="coerce").mean()), 2),
            "mean_evidence_quality_score": round(
                float(pd.to_numeric(group["evidence_quality_score"], errors="coerce").mean()), 2
            ),
        }

    return {
        "submission_count": int(len(merged)),
        "conditions": conditions,
        "notes": [
            "This summary is descriptive only until enough counterbalanced participant observations exist for paired testing."
        ],
    }
