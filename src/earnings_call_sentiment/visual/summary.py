from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .face_features import FaceFeatureExtractor
from .frame_extract import VideoMetadata, iter_sampled_frames, probe_video_metadata
from .pose_features import PoseFeatureExtractor
from .segment_aggregate import aggregate_visual_segments

VISUAL_SCHEMA_VERSION = "1.0.0"
FRAME_COLUMNS = [
    "timestamp_s",
    "frame_index",
    "face_detected",
    "face_count",
    "face_visible",
    "landmark_confidence",
    "motion_score",
    "head_shift_score",
    "head_yaw",
    "head_pitch",
    "gaze_shift_proxy",
    "blink_proxy",
    "pose_visible",
    "hand_visible",
    "shoulder_shift_score",
    "feature_note",
]
SEGMENT_COLUMNS = [
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


def _visibility_level(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def _stability_level(change_score: float) -> str:
    if change_score <= 0.06:
        return "high"
    if change_score <= 0.14:
        return "medium"
    return "low"


def _shift_level(delta: float) -> str:
    if abs(delta) >= 0.12:
        return "high"
    if abs(delta) >= 0.05:
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
    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "video_available": metadata is not None,
        "visual_features_available": False,
        "face_visibility_overall": {"pct": 0.0, "level": "low"},
        "prepared_baseline_visual_stability": {"score": 0.0, "level": "low"},
        "qa_visual_shift_score": {"delta": 0.0, "level": "low"},
        "most_visually_changed_segments": [],
        "most_confident_visual_segments": [],
        "notable_low_confidence_segments": [],
        "limitations": limitations,
        "notes": [
            "Visual behavior signals are observational proxies only; they are not emotion or deception inference.",
            "Low face visibility or framing quality can make visual interpretation unreliable.",
        ],
    }


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

    changed_segments = segments_df.sort_values(["visual_change_score", "face_visible_pct"], ascending=[False, False])
    confident_segments = segments_df.sort_values(["face_visible_pct", "avg_landmark_confidence"], ascending=[False, False])
    low_conf_segments = segments_df.sort_values(["face_visible_pct", "avg_landmark_confidence"], ascending=[True, True])

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
                    "face_visible_pct": round(float(row["face_visible_pct"]), 4),
                    "confidence_note": str(row["confidence_note"]),
                    "text": _truncate(str(row["text"])),
                }
            )
        return items

    limitations = []
    if face_visible_pct < 0.4:
        limitations.append("Low face visibility reduces confidence in visual interpretation.")
    if any(str(note).startswith("low face visibility") for note in segments_df["confidence_note"]):
        limitations.append("Some segments have low face visibility or sparse usable frames.")
    if not qa_segments.empty and len(qa_segments) < 3:
        limitations.append("Q&A visual shift is based on few Q&A segments and should be treated cautiously.")

    return {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "video_available": True,
        "visual_features_available": True,
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
        "most_visually_changed_segments": _segment_items(changed_segments),
        "most_confident_visual_segments": _segment_items(confident_segments),
        "notable_low_confidence_segments": _segment_items(low_conf_segments[low_conf_segments["face_visible_pct"] < 0.5]),
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

    metadata = probe_video_metadata(video_path)
    frames: list[dict[str, Any]] = []
    import cv2  # type: ignore

    previous_gray = None
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
                    "motion_score": round(motion_score, 4),
                    "head_shift_score": float(face_features["head_shift_score"]),
                    "head_yaw": float(face_features["head_yaw"]),
                    "head_pitch": float(face_features["head_pitch"]),
                    "gaze_shift_proxy": float(face_features["gaze_shift_proxy"]),
                    "blink_proxy": float(face_features["blink_proxy"]),
                    "pose_visible": bool(pose_features["pose_visible"]),
                    "hand_visible": bool(pose_features["hand_visible"]),
                    "shoulder_shift_score": float(pose_features["shoulder_shift_score"]),
                    "feature_note": feature_note or "ok",
                }
            )

    frames_df = pd.DataFrame(frames, columns=FRAME_COLUMNS)
    if frames_df.empty:
        return {
            "frames_df": frames_df,
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable("Frame sampling produced no usable frames.", metadata=metadata),
        }

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
