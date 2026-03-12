from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.media_support_models import score_visual_support

from .face_features import FaceFeatureExtractor
from .frame_extract import VideoMetadata, iter_sampled_frames, probe_video_metadata
from .pose_features import PoseFeatureExtractor
from .runtime import load_cv2
from .segment_aggregate import aggregate_visual_segments

VISUAL_SCHEMA_VERSION = "1.2.0"
FRAME_COLUMNS = [
    "timestamp_s",
    "frame_index",
    "face_detected",
    "face_count",
    "face_visible",
    "landmark_confidence",
    "face_size_ratio",
    "motion_score",
    "head_shift_score",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_pose_drift",
    "gaze_shift_proxy",
    "gaze_stability_proxy",
    "blink_proxy",
    "blink_onset_proxy",
    "eye_aspect_ratio",
    "mouth_open_ratio",
    "mouth_open_delta",
    "lower_face_tension_proxy",
    "pose_visible",
    "pose_confidence",
    "hand_visible",
    "shoulder_shift_score",
    "shoulder_asymmetry",
    "shoulder_motion_energy",
    "hand_motion_proxy",
    "feature_note",
]
SEGMENT_COLUMNS = [
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


def _visibility_level(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _stability_level(change_score: float) -> str:
    if change_score <= 0.06:
        return "high"
    if change_score <= 0.16:
        return "medium"
    return "low"


def _shift_level(delta: float) -> str:
    if abs(delta) >= 0.15:
        return "high"
    if abs(delta) >= 0.06:
        return "medium"
    return "low"


def _pressure_level(score: float) -> str:
    if score >= 0.18:
        return "high"
    if score >= 0.08:
        return "medium"
    return "low"


def _truncate(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _empty_frames_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAME_COLUMNS)


def _empty_segments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEGMENT_COLUMNS)


def _summary_unavailable(reason: str, *, metadata: VideoMetadata | None = None) -> dict[str, Any]:
    limitations = [reason]
    if metadata is not None:
        limitations.append(
            f"video metadata: duration={metadata.duration_s:.2f}s, fps={metadata.fps:.2f}, size={metadata.width}x{metadata.height}"
        )
    quality_gate = {
        "quality_ok": False,
        "suppression_recommended": True,
        "stable_face_frame_pct": 0.0,
        "stable_face_frame_pct_ok": False,
        "stable_face_tracking_pct": 0.0,
        "stable_face_tracking_ok": False,
        "mean_landmark_confidence": 0.0,
        "landmark_confidence_ok": False,
        "face_size_ratio_mean": 0.0,
        "face_size_ratio_ok": False,
        "pose_frame_pct": 0.0,
        "pose_frame_pct_ok": False,
        "frame_resolution_ok": False,
        "sampled_frame_count": 0,
        "min_face_frame_count_ok": False,
    }
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "video_available": metadata is not None,
        "visual_features_available": False,
        "video_quality_ok": False,
        "quality_gate": quality_gate,
        "face_visibility_overall": {"pct": 0.0, "level": "low"},
        "prepared_baseline_visual_stability": {"score": 0.0, "level": "low"},
        "qa_visual_shift_score": {"delta": 0.0, "level": "low"},
        "facial_tension_level": {"score": 0.0, "level": "low"},
        "head_motion_pressure": {"score": 0.0, "level": "low"},
        "visual_stability": {"score": 0.0, "level": "low"},
        "visual_support_direction": "unavailable",
        "visual_confidence_support": {
            "level": "low",
            "suppressed": True,
            "reason": reason,
        },
        "support_mode": "heuristic_fallback",
        "model_support": {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "unavailable",
            "calibrated_support_score": 0.0,
            "reason": reason,
        },
        "strongest_visual_evidence": [],
        "most_visually_changed_segments": [],
        "most_confident_visual_segments": [],
        "notable_low_confidence_segments": [],
        "limitations": limitations,
        "notes": [
            "Visual behavior signals are observational proxies only; they are not emotion or deception inference.",
            "Low face visibility or framing quality can make visual interpretation unreliable.",
        ],
    }


def _segment_items(frame: pd.DataFrame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, row in frame.head(3).iterrows():
        items.append(
            {
                "segment_id": int(row["segment_id"]),
                "section": str(row["section"]),
                "speaker_role": str(row["speaker_role"]),
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "visual_change_score": round(float(row["visual_change_score"]), 4),
                "head_motion_energy": round(float(row.get("head_motion_energy", 0.0)), 4),
                "head_pose_drift_mean": round(float(row.get("head_pose_drift_mean", 0.0)), 4),
                "blink_rate_per_10s": round(float(row.get("blink_rate_per_10s", 0.0)), 4),
                "face_visible_pct": round(float(row["face_visible_pct"]), 4),
                "confidence_note": str(row["confidence_note"]),
                "support_direction": str(row.get("support_direction", "unavailable")),
                "text": _truncate(str(row["text"])),
            }
        )
    return items


def _quality_gate(frames_df: pd.DataFrame, segments_df: pd.DataFrame, metadata: VideoMetadata) -> dict[str, Any]:
    stable_face_pct = float(frames_df["face_visible"].mean()) if not frames_df.empty else 0.0
    stable_tracking_pct = (
        float(
            (
                frames_df["face_visible"]
                & (frames_df["landmark_confidence"].fillna(0.0) >= 0.5)
                & (frames_df["face_size_ratio"].fillna(0.0) >= 0.06)
            ).mean()
        )
        if not frames_df.empty
        else 0.0
    )
    landmark_mean = float(frames_df["landmark_confidence"].fillna(0.0).mean()) if not frames_df.empty else 0.0
    face_size_mean = float(frames_df["face_size_ratio"].fillna(0.0).mean()) if "face_size_ratio" in frames_df.columns else 0.0
    pose_frame_pct = float(frames_df["pose_visible"].mean()) if "pose_visible" in frames_df.columns and not frames_df.empty else 0.0
    frame_resolution_ok = bool(metadata.width >= 320 and metadata.height >= 240)
    sampled_frame_count = int(len(frames_df))
    min_face_frame_count = int((frames_df["face_visible"] == True).sum()) if not frames_df.empty else 0
    quality_ok = (
        stable_face_pct >= 0.45
        and stable_tracking_pct >= 0.35
        and landmark_mean >= 0.5
        and face_size_mean >= 0.06
        and frame_resolution_ok
        and min_face_frame_count >= 6
    )
    return {
        "quality_ok": bool(quality_ok),
        "suppression_recommended": not bool(quality_ok),
        "stable_face_frame_pct": round(stable_face_pct, 4),
        "stable_face_frame_pct_ok": bool(stable_face_pct >= 0.45),
        "stable_face_tracking_pct": round(stable_tracking_pct, 4),
        "stable_face_tracking_ok": bool(stable_tracking_pct >= 0.35),
        "mean_landmark_confidence": round(landmark_mean, 4),
        "landmark_confidence_ok": bool(landmark_mean >= 0.5),
        "face_size_ratio_mean": round(face_size_mean, 4),
        "face_size_ratio_ok": bool(face_size_mean >= 0.06),
        "pose_frame_pct": round(pose_frame_pct, 4),
        "pose_frame_pct_ok": bool(pose_frame_pct >= 0.2),
        "frame_resolution_ok": frame_resolution_ok,
        "sampled_frame_count": sampled_frame_count,
        "min_face_frame_count_ok": bool(min_face_frame_count >= 6),
    }


def _confidence_support_level(quality_gate: dict[str, Any], qa_segments: pd.DataFrame) -> str:
    score = 0
    if bool(quality_gate.get("quality_ok")):
        score += 2
    if bool(quality_gate.get("stable_face_frame_pct_ok")):
        score += 1
    if bool(quality_gate.get("stable_face_tracking_ok")):
        score += 1
    if bool(quality_gate.get("face_size_ratio_ok")):
        score += 1
    if bool(quality_gate.get("pose_frame_pct_ok")):
        score += 1
    if len(qa_segments) >= 2:
        score += 1
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _strongest_visual_evidence(segments_df: pd.DataFrame) -> list[dict[str, Any]]:
    ranked = segments_df.sort_values(
        ["visual_change_score", "head_motion_energy", "face_visible_pct"],
        ascending=[False, False, False],
    )
    return _segment_items(ranked)


def _build_summary(frames_df: pd.DataFrame, segments_df: pd.DataFrame, metadata: VideoMetadata, sample_fps: float) -> dict[str, Any]:
    if frames_df.empty or segments_df.empty:
        return _summary_unavailable(
            "No usable visual segments were produced from the sampled frames.",
            metadata=metadata,
        )

    face_visible_pct = float(frames_df["face_visible"].mean()) if not frames_df.empty else 0.0
    prepared_segments = segments_df[segments_df["section"] == "prepared_remarks"].copy()
    qa_segments = segments_df[segments_df["section"] == "q_and_a"].copy()
    prepared_change = float(prepared_segments["visual_change_score"].mean()) if not prepared_segments.empty else 0.0
    qa_change = float(qa_segments["visual_change_score"].mean()) if not qa_segments.empty else 0.0
    qa_delta = qa_change - prepared_change
    facial_tension_score = (
        float(qa_segments["avg_lower_face_tension"].mean()) if not qa_segments.empty else float(segments_df["avg_lower_face_tension"].mean())
    )
    head_motion_score = (
        float(qa_segments["head_motion_energy"].mean()) if not qa_segments.empty else float(segments_df["head_motion_energy"].mean())
    )
    visual_stability_score = max(0.0, 1.0 - min(float(segments_df["visual_change_score"].mean()), 1.0))
    if not qa_segments.empty and "support_direction" in qa_segments.columns:
        direction_counts = qa_segments["support_direction"].value_counts()
    else:
        direction_counts = pd.Series(dtype="int64")
    if int(direction_counts.get("cautionary", 0)) >= 1:
        support_direction = "cautionary"
    elif int(direction_counts.get("supportive", 0)) >= 1 and int(direction_counts.get("unavailable", 0)) == 0:
        support_direction = "supportive"
    elif int(direction_counts.get("unavailable", 0)) == len(qa_segments) and len(qa_segments) > 0:
        support_direction = "unavailable"
    else:
        support_direction = "neutral"

    changed_segments = segments_df.sort_values(["visual_change_score", "face_visible_pct"], ascending=[False, False])
    confident_segments = segments_df.sort_values(["face_visible_pct", "avg_landmark_confidence"], ascending=[False, False])
    low_conf_segments = segments_df.sort_values(["face_visible_pct", "avg_landmark_confidence"], ascending=[True, True])
    quality_gate = _quality_gate(frames_df, segments_df, metadata)

    limitations = []
    if not bool(quality_gate["stable_face_frame_pct_ok"]):
        limitations.append("Low stable face visibility reduces confidence in visual interpretation.")
    if not bool(quality_gate["face_size_ratio_ok"]):
        limitations.append("On-screen face size is small, so visual pressure cues are less reliable.")
    if not bool(quality_gate["pose_frame_pct_ok"]):
        limitations.append("Upper-body coverage is limited, so shoulder and hand signals remain secondary.")
    if any(str(note).startswith("low face visibility") or str(note).startswith("small on-screen face") for note in segments_df["confidence_note"]):
        limitations.append("Some segments have weak face visibility, small face crops, or sparse usable frames.")
    if not qa_segments.empty and len(qa_segments) < 3:
        limitations.append("Q&A visual shift is based on few Q&A segments and should be treated cautiously.")

    model_support = score_visual_support(segments_df)

    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "video_available": True,
        "visual_features_available": True,
        "video_quality_ok": bool(quality_gate["quality_ok"]),
        "quality_gate": quality_gate,
        "video_metadata": {
            "path": str(metadata.path),
            "duration_s": round(metadata.duration_s, 3),
            "fps": round(metadata.fps, 3),
            "frame_count": int(metadata.frame_count),
            "width": int(metadata.width),
            "height": int(metadata.height),
            "sample_fps": float(sample_fps),
        },
        "face_visibility_overall": {
            "pct": round(face_visible_pct, 4),
            "level": _visibility_level(face_visible_pct),
        },
        "prepared_baseline_visual_stability": {
            "score": round(prepared_change, 4),
            "level": _stability_level(prepared_change),
        },
        "qa_visual_shift_score": {
            "delta": round(qa_delta, 4),
            "prepared_mean": round(prepared_change, 4),
            "q_and_a_mean": round(qa_change, 4),
            "level": _shift_level(qa_delta),
        },
        "facial_tension_level": {
            "score": round(facial_tension_score, 4),
            "level": _pressure_level(facial_tension_score),
        },
        "head_motion_pressure": {
            "score": round(head_motion_score, 4),
            "level": _pressure_level(head_motion_score),
        },
        "visual_stability": {
            "score": round(visual_stability_score, 4),
            "level": _visibility_level(visual_stability_score),
        },
        "visual_support_direction": support_direction if bool(quality_gate["quality_ok"]) else "unavailable",
        "visual_confidence_support": {
            "level": _confidence_support_level(quality_gate, qa_segments),
            "suppressed": not bool(quality_gate["quality_ok"]),
            "reason": (
                "quality gate suppressed visual confidence uplift"
                if not bool(quality_gate["quality_ok"])
                else "usable face visibility and landmark support"
            ),
        },
        "support_mode": str(model_support.get("mode", "heuristic_fallback")),
        "model_support": model_support,
        "strongest_visual_evidence": _strongest_visual_evidence(changed_segments),
        "most_visually_changed_segments": _segment_items(changed_segments),
        "most_confident_visual_segments": _segment_items(confident_segments),
        "notable_low_confidence_segments": _segment_items(
            low_conf_segments[low_conf_segments["face_visible_pct"] < 0.5]
        ),
        "limitations": limitations,
        "notes": [
            "Visual behavior signals are observational proxies only; they are not emotion or deception inference.",
            "Frame features are sampled at a low rate and aggregated into transcript-aligned segments for reviewer decision support.",
        ],
    }


def compute_visual_behavior_outputs(
    video_path: Path | None,
    qa_segments_df: pd.DataFrame,
    *,
    sample_fps: float = 1.0,
) -> dict[str, Any]:
    if video_path is None:
        return {
            "frames_df": _empty_frames_df(),
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable("No video source was available for this run."),
        }

    try:
        metadata = probe_video_metadata(video_path)
    except Exception as exc:
        return {
            "frames_df": _empty_frames_df(),
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable(f"Video analysis unavailable: {exc}"),
        }

    frames: list[dict[str, Any]] = []
    try:
        cv2 = load_cv2()
    except Exception as exc:  # pragma: no cover - runtime dependency
        return {
            "frames_df": _empty_frames_df(),
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable(f"Video analysis unavailable: {exc}", metadata=metadata),
        }

    previous_gray = None
    try:
        with FaceFeatureExtractor() as face_extractor, PoseFeatureExtractor() as pose_extractor:
            for item in iter_sampled_frames(metadata.path, sample_fps=sample_fps):
                frame_bgr = item["frame_bgr"]
                timestamp_s = float(item["timestamp_s"])
                frame_index = int(item["frame_index"])

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                motion_score = 0.0
                if previous_gray is not None:
                    diff = cv2.absdiff(gray, previous_gray)
                    motion_score = float(diff.mean() / 255.0)
                previous_gray = gray

                face_features = face_extractor.process(frame_bgr)
                pose_features = pose_extractor.process(frame_bgr)
                feature_note = str(face_features.get("feature_note", ""))
                if not pose_features.get("pose_visible", False):
                    feature_note = feature_note or "pose_not_visible"
                frames.append(
                    {
                        "timestamp_s": round(timestamp_s, 4),
                        "frame_index": frame_index,
                        "face_detected": bool(face_features["face_detected"]),
                        "face_count": int(face_features["face_count"]),
                        "face_visible": bool(face_features["face_visible"]),
                        "landmark_confidence": float(face_features["landmark_confidence"]),
                        "face_size_ratio": float(face_features.get("face_size_ratio", 0.0)),
                        "motion_score": round(motion_score, 4),
                        "head_shift_score": float(face_features["head_shift_score"]),
                        "head_yaw": float(face_features["head_yaw"]),
                        "head_pitch": float(face_features["head_pitch"]),
                        "head_roll": float(face_features.get("head_roll", 0.0)),
                        "head_pose_drift": float(face_features.get("head_pose_drift", 0.0)),
                        "gaze_shift_proxy": float(face_features["gaze_shift_proxy"]),
                        "gaze_stability_proxy": float(face_features.get("gaze_stability_proxy", 0.0)),
                        "blink_proxy": float(face_features["blink_proxy"]),
                        "blink_onset_proxy": float(face_features.get("blink_onset_proxy", 0.0)),
                        "eye_aspect_ratio": float(face_features.get("eye_aspect_ratio", 0.0)),
                        "mouth_open_ratio": float(face_features.get("mouth_open_ratio", 0.0)),
                        "mouth_open_delta": float(face_features.get("mouth_open_delta", 0.0)),
                        "lower_face_tension_proxy": float(face_features.get("lower_face_tension_proxy", 0.0)),
                        "pose_visible": bool(pose_features["pose_visible"]),
                        "pose_confidence": float(pose_features.get("pose_confidence", 0.0)),
                        "hand_visible": bool(pose_features["hand_visible"]),
                        "shoulder_shift_score": float(pose_features["shoulder_shift_score"]),
                        "shoulder_asymmetry": float(pose_features.get("shoulder_asymmetry", 0.0)),
                        "shoulder_motion_energy": float(pose_features.get("shoulder_motion_energy", 0.0)),
                        "hand_motion_proxy": float(pose_features.get("hand_motion_proxy", 0.0)),
                        "feature_note": feature_note or "ok",
                    }
                )
    except Exception as exc:
        return {
            "frames_df": _empty_frames_df(),
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable(f"Video analysis unavailable: {exc}", metadata=metadata),
        }

    frames_df = pd.DataFrame(frames, columns=FRAME_COLUMNS)
    if frames_df.empty:
        return {
            "frames_df": frames_df,
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable("Frame sampling produced no usable frames.", metadata=metadata),
        }

    return summarize_visual_behavior_frames(
        frames_df,
        qa_segments_df,
        metadata,
        sample_fps=sample_fps,
    )


def summarize_visual_behavior_frames(
    frames_df: pd.DataFrame,
    qa_segments_df: pd.DataFrame,
    metadata: VideoMetadata,
    *,
    sample_fps: float = 1.0,
) -> dict[str, Any]:
    segments_df = aggregate_visual_segments(frames_df, qa_segments_df)
    summary = _build_summary(frames_df, segments_df, metadata, sample_fps)
    return {
        "frames_df": frames_df,
        "segments_df": segments_df,
        "summary": summary,
    }


def write_visual_behavior_outputs(
    video_path: Path | None,
    qa_segments_df: pd.DataFrame,
    out_dir: Path,
    *,
    sample_fps: float = 1.0,
) -> dict[str, Any]:
    payload = compute_visual_behavior_outputs(video_path, qa_segments_df, sample_fps=sample_fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "visual_behavior_frames.csv"
    segments_path = out_dir / "visual_behavior_segments.csv"
    summary_path = out_dir / "visual_behavior_summary.json"
    payload["frames_df"].to_csv(frames_path, index=False)
    payload["segments_df"].to_csv(segments_path, index=False)
    summary_path.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    return {
        **payload,
        "frames_path": frames_path,
        "segments_path": segments_path,
        "summary_path": summary_path,
    }
