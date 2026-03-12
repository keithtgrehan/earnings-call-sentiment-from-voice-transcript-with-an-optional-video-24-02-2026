from __future__ import annotations

from typing import Any

import pandas as pd


def _visual_change_score(row: pd.Series) -> float:
    return float(
        (float(row.get("avg_motion_score", 0.0)) * 0.4)
        + (float(row.get("avg_head_shift", 0.0)) * 0.25)
        + (float(row.get("avg_gaze_shift", 0.0)) * 0.15)
        + (float(row.get("avg_shoulder_shift", 0.0)) * 0.2)
    )


def _stability_label(score: float) -> str:
    if score >= 0.18:
        return "elevated_change"
    if score >= 0.08:
        return "somewhat_changed"
    return "stable"


def _confidence_note(frame_count: int, face_visible_pct: float, landmark_confidence: float) -> str:
    if frame_count <= 0:
        return "no sampled frames overlapped this segment"
    if face_visible_pct < 0.35:
        return "low face visibility reduces confidence"
    if landmark_confidence < 0.45:
        return "low landmark confidence reduces confidence"
    return "usable visual segment"


def aggregate_visual_segments(
    frames_df: pd.DataFrame,
    qa_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "segment_id",
        "section",
        "speaker_role",
        "start_time_s",
        "end_time_s",
        "face_visible_pct",
        "avg_motion_score",
        "max_motion_score",
        "avg_head_shift",
        "avg_gaze_shift",
        "blink_rate_proxy",
        "avg_shoulder_shift",
        "visual_stability_label",
        "confidence_note",
        "avg_landmark_confidence",
        "visual_change_score",
        "frame_count",
        "text",
    ]
    if qa_segments_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    frames = frames_df.sort_values("timestamp_s").reset_index(drop=True)
    segments = qa_segments_df.sort_values("start").reset_index(drop=True)

    for idx, segment in segments.iterrows():
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end < start:
            end = start
        window = frames[(frames["timestamp_s"] >= start) & (frames["timestamp_s"] <= end)]
        face_visible_pct = float(window["face_visible"].mean()) if not window.empty else 0.0
        avg_motion = float(window["motion_score"].mean()) if not window.empty else 0.0
        max_motion = float(window["motion_score"].max()) if not window.empty else 0.0
        avg_head_shift = float(window["head_shift_score"].mean()) if not window.empty else 0.0
        avg_gaze_shift = float(window["gaze_shift_proxy"].mean()) if not window.empty else 0.0
        blink_rate = float(window["blink_proxy"].mean()) if not window.empty else 0.0
        avg_shoulder_shift = float(window["shoulder_shift_score"].mean()) if not window.empty else 0.0
        avg_landmark_conf = float(window["landmark_confidence"].fillna(0.0).mean()) if not window.empty else 0.0
        frame_count = int(len(window))
        row = {
            "segment_id": int(segment.get("segment_id", idx)),
            "section": str(segment.get("phase", "prepared_remarks")),
            "speaker_role": str(segment.get("speaker_role", "management")),
            "start_time_s": round(start, 4),
            "end_time_s": round(end, 4),
            "face_visible_pct": round(face_visible_pct, 4),
            "avg_motion_score": round(avg_motion, 4),
            "max_motion_score": round(max_motion, 4),
            "avg_head_shift": round(avg_head_shift, 4),
            "avg_gaze_shift": round(avg_gaze_shift, 4),
            "blink_rate_proxy": round(blink_rate, 4),
            "avg_shoulder_shift": round(avg_shoulder_shift, 4),
            "avg_landmark_confidence": round(avg_landmark_conf, 4),
            "frame_count": frame_count,
            "text": str(segment.get("text", "")),
        }
        score = _visual_change_score(pd.Series(row))
        row["visual_change_score"] = round(score, 4)
        row["visual_stability_label"] = _stability_label(score)
        row["confidence_note"] = _confidence_note(frame_count, face_visible_pct, avg_landmark_conf)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
