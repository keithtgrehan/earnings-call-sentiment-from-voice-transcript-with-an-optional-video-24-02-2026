from __future__ import annotations

from typing import Any

import pandas as pd


def _visual_change_score(row: pd.Series) -> float:
    return float(
        (float(row.get("avg_motion_score", 0.0)) * 0.3)
        + (float(row.get("max_motion_score", 0.0)) * 0.15)
        + (float(row.get("head_motion_energy", 0.0)) * 0.2)
        + (float(row.get("avg_gaze_shift", 0.0)) * 0.1)
        + (float(row.get("shoulder_motion_energy", 0.0)) * 0.1)
        + (float(row.get("mouth_open_variance", 0.0)) * 0.1)
        + (float(row.get("head_pose_drift_mean", 0.0)) * 0.05)
    )


def _stability_label(score: float) -> str:
    if score >= 0.18:
        return "elevated_change"
    if score >= 0.08:
        return "somewhat_changed"
    return "stable"


def _confidence_note(
    frame_count: int,
    face_visible_pct: float,
    stable_face_pct: float,
    landmark_confidence: float,
    face_size_ratio: float,
    pose_visible_pct: float,
) -> str:
    if frame_count <= 0:
        return "no sampled frames overlapped this segment"
    if face_visible_pct < 0.35:
        return "low face visibility reduces confidence"
    if stable_face_pct < 0.35:
        return "unstable face tracking reduces confidence"
    if landmark_confidence < 0.45:
        return "low landmark confidence reduces confidence"
    if face_size_ratio < 0.06:
        return "small on-screen face reduces confidence"
    if pose_visible_pct < 0.2:
        return "pose coverage is limited for shoulder and hand features"
    return "usable visual segment"


def _segment_support_direction(
    *,
    stable_face_pct: float,
    face_size_ratio_mean: float,
    head_motion_energy: float,
    head_pose_drift_mean: float,
    avg_lower_face_tension: float,
) -> tuple[str, str]:
    if stable_face_pct < 0.35 or face_size_ratio_mean < 0.06:
        return "unavailable", "visual quality gate is too weak for segment-level support"
    risk_score = (
        (head_motion_energy * 0.5)
        + (head_pose_drift_mean * 0.2)
        + (avg_lower_face_tension * 0.3)
    )
    if risk_score >= 0.2:
        return "cautionary", "visible motion and lower-face tension rose in this window"
    if risk_score <= 0.08:
        return "supportive", "visible delivery stayed comparatively steady in this window"
    return "neutral", "visible delivery cues were mixed in this window"


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
        "stable_face_pct",
        "face_size_ratio_mean",
        "avg_motion_score",
        "max_motion_score",
        "avg_head_shift",
        "avg_head_yaw_abs",
        "avg_head_pitch_abs",
        "avg_head_roll_abs",
        "head_motion_energy",
        "head_pose_drift_mean",
        "avg_gaze_shift",
        "gaze_stability_mean",
        "blink_rate_proxy",
        "blink_rate_per_10s",
        "blink_burstiness_proxy",
        "eye_aspect_ratio_mean",
        "mouth_open_variance",
        "mouth_open_delta_mean",
        "avg_lower_face_tension",
        "pose_visible_pct",
        "pose_confidence_mean",
        "shoulder_asymmetry_mean",
        "avg_shoulder_shift",
        "shoulder_motion_energy",
        "hand_visibility_pct",
        "hand_motion_presence_pct",
        "visual_stability_label",
        "confidence_note",
        "support_direction",
        "support_note",
        "avg_landmark_confidence",
        "visual_change_score",
        "frame_count",
        "text",
    ]
    if qa_segments_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    frames = frames_df.sort_values("timestamp_s").reset_index(drop=True)
    for column, default_value in (
        ("face_size_ratio", 0.18),
        ("head_yaw", 0.0),
        ("head_pitch", 0.0),
        ("head_roll", 0.0),
        ("head_pose_drift", 0.0),
        ("mouth_open_ratio", 0.0),
        ("mouth_open_delta", 0.0),
        ("lower_face_tension_proxy", 0.0),
        ("gaze_stability_proxy", 0.0),
        ("blink_onset_proxy", 0.0),
        ("eye_aspect_ratio", 0.0),
        ("pose_visible", False),
        ("hand_visible", False),
        ("pose_confidence", 0.0),
        ("shoulder_asymmetry", 0.0),
        ("shoulder_motion_energy", 0.0),
        ("hand_motion_proxy", 0.0),
    ):
        if column not in frames.columns:
            frames[column] = default_value
    segments = qa_segments_df.sort_values("start").reset_index(drop=True)

    for idx, segment in segments.iterrows():
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end < start:
            end = start
        duration_s = max(end - start, 0.01)
        window = frames[(frames["timestamp_s"] >= start) & (frames["timestamp_s"] <= end)]
        face_visible_pct = float(window["face_visible"].mean()) if not window.empty else 0.0
        stable_face_pct = (
            float(
                (
                    window["face_visible"]
                    & (window["landmark_confidence"].fillna(0.0) >= 0.5)
                    & (window["face_size_ratio"].fillna(0.0) >= 0.06)
                ).mean()
            )
            if not window.empty
            else 0.0
        )
        face_size_ratio_mean = float(window["face_size_ratio"].mean()) if not window.empty else 0.0
        avg_motion = float(window["motion_score"].mean()) if not window.empty else 0.0
        max_motion = float(window["motion_score"].max()) if not window.empty else 0.0
        avg_head_shift = float(window["head_shift_score"].mean()) if not window.empty else 0.0
        avg_head_yaw_abs = float(window["head_yaw"].abs().mean()) if not window.empty else 0.0
        avg_head_pitch_abs = float(window["head_pitch"].abs().mean()) if not window.empty else 0.0
        avg_head_roll_abs = float(window["head_roll"].abs().mean()) if not window.empty else 0.0
        head_pose_drift_mean = float(window["head_pose_drift"].mean()) if not window.empty else 0.0
        head_motion_energy = (
            (avg_head_shift * 0.45)
            + (avg_head_yaw_abs * 0.2)
            + (avg_head_pitch_abs * 0.2)
            + (avg_head_roll_abs * 0.15)
        )
        avg_gaze_shift = float(window["gaze_shift_proxy"].mean()) if not window.empty else 0.0
        gaze_stability_mean = float(window["gaze_stability_proxy"].mean()) if not window.empty else 0.0
        blink_rate = float(window["blink_proxy"].mean()) if not window.empty else 0.0
        blink_onsets = float(window["blink_onset_proxy"].sum()) if not window.empty else 0.0
        blink_rate_per_10s = float(blink_onsets / duration_s * 10.0) if duration_s > 0 else 0.0
        blink_runs = 0
        if not window.empty:
            previous_on = False
            for value in window["blink_proxy"].fillna(0.0):
                current_on = float(value) > 0.5
                if current_on and not previous_on:
                    blink_runs += 1
                previous_on = current_on
        blink_burstiness = float(blink_runs / max(len(window), 1)) if not window.empty else 0.0
        eye_aspect_ratio_mean = float(window["eye_aspect_ratio"].mean()) if not window.empty else 0.0
        mouth_open_variance = float(window["mouth_open_ratio"].var(ddof=0)) if not window.empty else 0.0
        mouth_open_delta_mean = float(window["mouth_open_delta"].mean()) if not window.empty else 0.0
        avg_lower_face_tension = (
            float(window["lower_face_tension_proxy"].mean()) if not window.empty else 0.0
        )
        pose_visible_pct = float(window["pose_visible"].mean()) if not window.empty else 0.0
        pose_confidence_mean = float(window["pose_confidence"].mean()) if not window.empty else 0.0
        shoulder_asymmetry_mean = float(window["shoulder_asymmetry"].mean()) if not window.empty else 0.0
        avg_shoulder_shift = float(window["shoulder_shift_score"].mean()) if not window.empty else 0.0
        shoulder_motion_energy = float(window["shoulder_motion_energy"].mean()) if not window.empty else 0.0
        hand_visibility_pct = float(window["hand_visible"].mean()) if not window.empty else 0.0
        hand_motion_presence_pct = (
            float((window["hand_motion_proxy"] > 0.04).mean()) if not window.empty else 0.0
        )
        avg_landmark_conf = float(window["landmark_confidence"].fillna(0.0).mean()) if not window.empty else 0.0
        frame_count = int(len(window))
        support_direction, support_note = _segment_support_direction(
            stable_face_pct=stable_face_pct,
            face_size_ratio_mean=face_size_ratio_mean,
            head_motion_energy=head_motion_energy,
            head_pose_drift_mean=head_pose_drift_mean,
            avg_lower_face_tension=avg_lower_face_tension,
        )
        row = {
            "segment_id": int(segment.get("segment_id", idx)),
            "section": str(segment.get("phase", "prepared_remarks")),
            "speaker_role": str(segment.get("speaker_role", "management")),
            "start_time_s": round(start, 4),
            "end_time_s": round(end, 4),
            "face_visible_pct": round(face_visible_pct, 4),
            "stable_face_pct": round(stable_face_pct, 4),
            "face_size_ratio_mean": round(face_size_ratio_mean, 4),
            "avg_motion_score": round(avg_motion, 4),
            "max_motion_score": round(max_motion, 4),
            "avg_head_shift": round(avg_head_shift, 4),
            "avg_head_yaw_abs": round(avg_head_yaw_abs, 4),
            "avg_head_pitch_abs": round(avg_head_pitch_abs, 4),
            "avg_head_roll_abs": round(avg_head_roll_abs, 4),
            "head_motion_energy": round(head_motion_energy, 4),
            "head_pose_drift_mean": round(head_pose_drift_mean, 4),
            "avg_gaze_shift": round(avg_gaze_shift, 4),
            "gaze_stability_mean": round(gaze_stability_mean, 4),
            "blink_rate_proxy": round(blink_rate, 4),
            "blink_rate_per_10s": round(blink_rate_per_10s, 4),
            "blink_burstiness_proxy": round(blink_burstiness, 4),
            "eye_aspect_ratio_mean": round(eye_aspect_ratio_mean, 4),
            "mouth_open_variance": round(mouth_open_variance, 4),
            "mouth_open_delta_mean": round(mouth_open_delta_mean, 4),
            "avg_lower_face_tension": round(avg_lower_face_tension, 4),
            "pose_visible_pct": round(pose_visible_pct, 4),
            "pose_confidence_mean": round(pose_confidence_mean, 4),
            "shoulder_asymmetry_mean": round(shoulder_asymmetry_mean, 4),
            "avg_shoulder_shift": round(avg_shoulder_shift, 4),
            "shoulder_motion_energy": round(shoulder_motion_energy, 4),
            "hand_visibility_pct": round(hand_visibility_pct, 4),
            "hand_motion_presence_pct": round(hand_motion_presence_pct, 4),
            "avg_landmark_confidence": round(avg_landmark_conf, 4),
            "frame_count": frame_count,
            "text": str(segment.get("text", "")),
        }
        score = _visual_change_score(pd.Series(row))
        row["visual_change_score"] = round(score, 4)
        row["visual_stability_label"] = _stability_label(score)
        row["confidence_note"] = _confidence_note(
            frame_count,
            face_visible_pct,
            stable_face_pct,
            avg_landmark_conf,
            face_size_ratio_mean,
            pose_visible_pct,
        )
        row["support_direction"] = support_direction
        row["support_note"] = support_note
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
