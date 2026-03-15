from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_media_quality_summary(
    *,
    audio_summary: dict[str, Any] | None,
    visual_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    quality_notes: list[str] = []
    suppression_flags: dict[str, bool] = {}

    audio_quality_ok = False
    if isinstance(audio_summary, dict):
        audio_quality_ok = bool(audio_summary.get("audio_quality_ok"))
        for item in audio_summary.get("limitations", [])[:3]:
            quality_notes.append(f"audio: {item}")
        gate = audio_summary.get("quality_gate", {})
        suppression_flags["audio_support_suppressed"] = bool(gate.get("suppression_recommended", False))
    else:
        suppression_flags["audio_support_suppressed"] = True

    video_quality_ok = False
    if isinstance(visual_summary, dict):
        video_quality_ok = bool(visual_summary.get("video_quality_ok"))
        for item in visual_summary.get("limitations", [])[:3]:
            quality_notes.append(f"video: {item}")
        gate = visual_summary.get("quality_gate", {})
        suppression_flags["video_support_suppressed"] = bool(gate.get("suppression_recommended", False))
    else:
        suppression_flags["video_support_suppressed"] = True

    if not quality_notes:
        quality_notes.append("Audio and video quality gates did not raise additional caution notes.")

    return {
        "audio_quality_ok": audio_quality_ok,
        "video_quality_ok": video_quality_ok,
        "quality_notes": quality_notes,
        "suppression_flags": suppression_flags,
    }


def write_media_quality_summary(
    *,
    audio_summary: dict[str, Any] | None,
    visual_summary: dict[str, Any] | None,
    out_dir: Path,
) -> dict[str, Any]:
    summary = build_media_quality_summary(
        audio_summary=audio_summary,
        visual_summary=visual_summary,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "media_quality.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "output_path": output_path,
    }
