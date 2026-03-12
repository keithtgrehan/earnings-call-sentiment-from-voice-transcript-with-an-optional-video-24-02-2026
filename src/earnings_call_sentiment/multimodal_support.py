from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _transcript_confidence(metrics_payload: dict[str, Any]) -> float:
    try:
        value = float(metrics_payload.get("review_confidence_pct", 70.0)) / 100.0
    except Exception:
        value = 0.7
    return max(0.35, min(0.95, value))


def _audio_direction(
    audio_summary: dict[str, Any] | None,
    media_quality: dict[str, Any],
) -> tuple[str, list[str], float, float, str]:
    notes: list[str] = []
    if not isinstance(audio_summary, dict) or not bool(media_quality.get("audio_quality_ok")):
        return "unavailable", ["Audio support was suppressed because quality gates were not met."], 0.0, 0.0, "unavailable"

    model_support = audio_summary.get("model_support", {})
    if isinstance(model_support, dict) and bool(model_support.get("available")):
        direction = str(model_support.get("support_direction", "neutral"))
        score = float(model_support.get("calibrated_support_score", 0.0))
        reliability = float(model_support.get("reliability_weight", 0.0))
        notes.append(
            f"Audio support used model-backed scoring ({direction}, score {score:+.2f}, reliability {reliability:.2f})."
        )
        return direction, notes, score, reliability, "model_backed"

    hesitation = str(audio_summary.get("hesitation_level", {}).get("level", "low"))
    pause_delta = str(audio_summary.get("pause_pressure_delta", {}).get("label", "mixed"))
    latency = str(audio_summary.get("answer_latency_pressure", {}).get("level", "low"))

    if hesitation == "high" or latency == "high" or pause_delta == "more_paused_under_questions":
        notes.append("Audio delivery became more hesitant under questioning.")
        return "cautionary", notes, 0.08, 0.12, "heuristic_fallback"
    if hesitation == "low" and latency == "low" and pause_delta != "more_paused_under_questions":
        notes.append("Audio delivery stayed comparatively steady across answers.")
        return "supportive", notes, -0.08, 0.12, "heuristic_fallback"
    notes.append("Audio cues were mixed and stayed secondary to the transcript.")
    return "neutral", notes, 0.0, 0.12, "heuristic_fallback"


def _video_direction(
    visual_summary: dict[str, Any] | None,
    media_quality: dict[str, Any],
) -> tuple[str, list[str], float, float, str]:
    notes: list[str] = []
    if not isinstance(visual_summary, dict) or not bool(media_quality.get("video_quality_ok")):
        return "unavailable", ["Video support was suppressed because quality gates were not met."], 0.0, 0.0, "unavailable"

    model_support = visual_summary.get("model_support", {})
    if isinstance(model_support, dict) and bool(model_support.get("available")):
        direction = str(model_support.get("support_direction", "neutral"))
        score = float(model_support.get("calibrated_support_score", 0.0))
        reliability = float(model_support.get("reliability_weight", 0.0))
        notes.append(
            f"Visual support used model-backed scoring ({direction}, score {score:+.2f}, reliability {reliability:.2f})."
        )
        return direction, notes, score, reliability, "model_backed"

    facial_tension = str(visual_summary.get("facial_tension_level", {}).get("level", "low"))
    head_motion = str(visual_summary.get("head_motion_pressure", {}).get("level", "low"))
    qa_shift = str(visual_summary.get("qa_visual_shift_score", {}).get("level", "low"))

    if facial_tension == "high" or head_motion == "high" or qa_shift == "high":
        notes.append("Visual pressure cues rose during Q&A.")
        return "cautionary", notes, 0.06, 0.08, "heuristic_fallback"
    if facial_tension == "low" and head_motion == "low" and qa_shift != "high":
        notes.append("Visible delivery stayed comparatively stable when the face remained usable.")
        return "supportive", notes, -0.06, 0.08, "heuristic_fallback"
    notes.append("Visual cues were mixed and stayed secondary to the transcript.")
    return "neutral", notes, 0.0, 0.08, "heuristic_fallback"


def _alignment(
    transcript_signal: str,
    *,
    transcript_confidence: float,
    audio_direction: str,
    video_direction: str,
    audio_support_score: float,
    video_support_score: float,
) -> tuple[str, int]:
    combined_score = float(audio_support_score + video_support_score)
    if audio_direction == "unavailable" and video_direction == "unavailable":
        return "low", 0
    if abs(combined_score) < 0.03:
        return ("low" if transcript_confidence < 0.6 else "medium"), 0

    strength = max(1, min(4, round(abs(combined_score) * 12 * transcript_confidence)))
    cautionary = combined_score > 0
    supportive = combined_score < 0

    if transcript_signal == "green":
        if supportive:
            return "high", strength
        if cautionary:
            return "low", -strength
    if transcript_signal == "red":
        if cautionary:
            return "high", strength
        if supportive:
            return "low", -strength
    if transcript_signal == "amber":
        if cautionary:
            return "medium", min(2, strength)
        if supportive:
            return "medium", -min(2, strength)
    return "medium", 0


def build_multimodal_support_summary(
    *,
    metrics_payload: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    audio_summary: dict[str, Any] | None,
    visual_summary: dict[str, Any] | None,
    media_quality: dict[str, Any],
) -> dict[str, Any]:
    transcript_signal = str(metrics_payload.get("overall_review_signal", "amber"))
    transcript_confidence = _transcript_confidence(metrics_payload)
    audio_direction, audio_notes, audio_score, audio_weight, audio_mode = _audio_direction(audio_summary, media_quality)
    video_direction, video_notes, video_score, video_weight, video_mode = _video_direction(visual_summary, media_quality)
    audio_support_score = round(audio_score * audio_weight, 4)
    video_support_score = round(video_score * video_weight, 4)
    alignment, adjustment = _alignment(
        transcript_signal,
        transcript_confidence=transcript_confidence,
        audio_direction=audio_direction,
        video_direction=video_direction,
        audio_support_score=audio_support_score,
        video_support_score=video_support_score,
    )
    fusion_mode = "heuristic_fallback"
    if audio_mode == "model_backed" and video_mode == "model_backed":
        fusion_mode = "model_backed"
    elif audio_mode == "model_backed" or video_mode == "model_backed":
        fusion_mode = "hybrid"

    notes = [
        f"Transcript-first signal remains {transcript_signal} with interpretation confidence {transcript_confidence:.2f}.",
        *audio_notes,
        *video_notes,
    ]
    qa_level = str(qa_shift_summary.get("prepared_remarks_vs_q_and_a", {}).get("label", "mixed"))
    notes.append(f"Q&A transcript shift stayed {qa_level}.")

    return {
        "transcript_primary_assessment": transcript_signal,
        "audio_support_direction": audio_direction,
        "video_support_direction": video_direction,
        "fusion_mode": fusion_mode,
        "calibrated_support_score": round(audio_support_score + video_support_score, 4),
        "multimodal_alignment": alignment,
        "multimodal_confidence_adjustment": int(adjustment),
        "modality_weights": {
            "audio": round(audio_weight, 4),
            "video": round(video_weight, 4),
        },
        "notes": notes,
    }


def write_multimodal_support_summary(
    *,
    metrics_payload: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    audio_summary: dict[str, Any] | None,
    visual_summary: dict[str, Any] | None,
    media_quality: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    summary = build_multimodal_support_summary(
        metrics_payload=metrics_payload,
        qa_shift_summary=qa_shift_summary,
        audio_summary=audio_summary,
        visual_summary=visual_summary,
        media_quality=media_quality,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "multimodal_support_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "output_path": output_path,
    }
