from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _audio_direction(audio_summary: dict[str, Any] | None, media_quality: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not isinstance(audio_summary, dict) or not bool(media_quality.get("audio_quality_ok")):
        return "unavailable", ["Audio support was suppressed because quality gates were not met."]

    hesitation = str(audio_summary.get("hesitation_level", {}).get("level", "low"))
    pause_delta = str(audio_summary.get("pause_pressure_delta", {}).get("label", "mixed"))
    latency = str(audio_summary.get("answer_latency_pressure", {}).get("level", "low"))

    if hesitation == "high" or latency == "high" or pause_delta == "more_paused_under_questions":
        notes.append("Audio delivery became more hesitant under questioning.")
        return "cautionary", notes
    if hesitation == "low" and latency == "low" and pause_delta != "more_paused_under_questions":
        notes.append("Audio delivery stayed comparatively steady across answers.")
        return "supportive", notes
    notes.append("Audio cues were mixed and stayed secondary to the transcript.")
    return "neutral", notes


def _video_direction(visual_summary: dict[str, Any] | None, media_quality: dict[str, Any]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not isinstance(visual_summary, dict) or not bool(media_quality.get("video_quality_ok")):
        return "unavailable", ["Video support was suppressed because quality gates were not met."]

    facial_tension = str(visual_summary.get("facial_tension_level", {}).get("level", "low"))
    head_motion = str(visual_summary.get("head_motion_pressure", {}).get("level", "low"))
    qa_shift = str(visual_summary.get("qa_visual_shift_score", {}).get("level", "low"))

    if facial_tension == "high" or head_motion == "high" or qa_shift == "high":
        notes.append("Visual pressure cues rose during Q&A.")
        return "cautionary", notes
    if facial_tension == "low" and head_motion == "low" and qa_shift != "high":
        notes.append("Visible delivery stayed comparatively stable when the face remained usable.")
        return "supportive", notes
    notes.append("Visual cues were mixed and stayed secondary to the transcript.")
    return "neutral", notes


def _alignment(
    transcript_signal: str,
    audio_direction: str,
    video_direction: str,
) -> tuple[str, int]:
    cautionary_count = sum(direction == "cautionary" for direction in (audio_direction, video_direction))
    supportive_count = sum(direction == "supportive" for direction in (audio_direction, video_direction))

    if transcript_signal == "red" and cautionary_count >= 1:
        return "high", min(6, 2 + (2 * cautionary_count))
    if transcript_signal == "green" and cautionary_count >= 1:
        return "low", max(-6, -2 * cautionary_count)
    if transcript_signal == "green" and supportive_count >= 1:
        return "high", min(4, 1 + supportive_count)
    if transcript_signal == "amber" and (cautionary_count + supportive_count) >= 1:
        return "medium", 0
    if audio_direction == "unavailable" and video_direction == "unavailable":
        return "low", 0
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
    audio_direction, audio_notes = _audio_direction(audio_summary, media_quality)
    video_direction, video_notes = _video_direction(visual_summary, media_quality)
    alignment, adjustment = _alignment(transcript_signal, audio_direction, video_direction)

    notes = [
        f"Transcript-first signal remains {transcript_signal}.",
        *audio_notes,
        *video_notes,
    ]
    qa_level = str(qa_shift_summary.get("prepared_remarks_vs_q_and_a", {}).get("label", "mixed"))
    notes.append(f"Q&A transcript shift stayed {qa_level}.")

    return {
        "transcript_primary_assessment": transcript_signal,
        "audio_support_direction": audio_direction,
        "video_support_direction": video_direction,
        "multimodal_alignment": alignment,
        "multimodal_confidence_adjustment": int(adjustment),
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
