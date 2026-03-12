from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "media_support"

AUDIO_FEATURE_COLUMNS = [
    "pause_before_answer_ms",
    "answer_onset_delay_ms",
    "silence_ratio",
    "speech_duration_s",
    "pause_count",
    "pause_density_per_10s",
    "pause_burst_count",
    "mean_pause_ms",
    "max_pause_ms",
    "filler_density",
    "speech_rate_wpm",
    "articulation_rate_wpm",
    "mean_rms_db",
    "rms_std_db",
    "rms_range_db",
    "hesitation_score",
]

VISUAL_FEATURE_COLUMNS = [
    "face_visible_pct",
    "stable_face_pct",
    "face_size_ratio_mean",
    "visual_change_score",
    "head_motion_energy",
    "head_pose_drift_mean",
    "gaze_stability_mean",
    "blink_rate_per_10s",
    "blink_burstiness_proxy",
    "mouth_open_variance",
    "mouth_open_delta_mean",
    "avg_lower_face_tension",
    "pose_visible_pct",
    "shoulder_motion_energy",
    "hand_visibility_pct",
]

TASK_CONFIG = {
    "hesitation_pressure": {
        "feature_columns": AUDIO_FEATURE_COLUMNS,
        "label_column": "hesitation_pressure_label",
        "modality": "audio",
    },
    "delivery_confidence": {
        "feature_columns": AUDIO_FEATURE_COLUMNS,
        "label_column": "delivery_confidence_label",
        "modality": "audio",
    },
    "visual_tension": {
        "feature_columns": VISUAL_FEATURE_COLUMNS,
        "label_column": "visual_tension_label",
        "modality": "video",
    },
}


def _bundle_paths(task_name: str) -> tuple[Path, Path]:
    return (
        MODEL_ROOT / f"{task_name}.joblib",
        MODEL_ROOT / f"{task_name}_metadata.json",
    )


def load_task_bundle(task_name: str) -> dict[str, Any] | None:
    model_path, metadata_path = _bundle_paths(task_name)
    if not model_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    return {
        "model": model,
        "metadata": metadata,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def _prepare_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    for column in feature_columns:
        if column not in work.columns:
            work[column] = 0.0
    return work[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def score_task_frame(task_name: str, frame: pd.DataFrame) -> dict[str, Any] | None:
    bundle = load_task_bundle(task_name)
    if bundle is None or frame.empty:
        return None

    metadata = bundle["metadata"]
    feature_columns = list(metadata.get("feature_columns", []))
    if not feature_columns:
        return None

    features = _prepare_feature_frame(frame, feature_columns)
    model = bundle["model"]
    predicted_labels = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = [str(value) for value in model.classes_]

    scored = frame.copy()
    scored[f"{task_name}_predicted_label"] = [str(value) for value in predicted_labels]
    for index, class_label in enumerate(classes):
        scored[f"{task_name}_prob_{class_label}"] = probabilities[:, index]

    probability_means = {
        class_label: round(float(scored[f"{task_name}_prob_{class_label}"].mean()), 4)
        for class_label in classes
    }
    predicted_label = max(probability_means, key=probability_means.get)

    return {
        "available": True,
        "task_name": task_name,
        "rows": len(scored),
        "predicted_label": predicted_label,
        "probability_means": probability_means,
        "reliability_weight": float(metadata.get("reliability_weight", 0.0)),
        "calibration_mode": str(metadata.get("calibration_mode", "not_available")),
        "scored_frame": scored,
    }


def score_audio_support(segments_df: pd.DataFrame) -> dict[str, Any]:
    answer_segments = segments_df[
        (segments_df["speaker_role"] == "management")
        & (segments_df["section"] == "q_and_a")
        & (segments_df["confidence_note"] == "usable audio segment")
    ].copy()
    if answer_segments.empty:
        return {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "unavailable",
            "calibrated_support_score": 0.0,
            "reason": "No usable management answer segments were available for model-backed audio scoring.",
        }

    hesitation = score_task_frame("hesitation_pressure", answer_segments)
    delivery = score_task_frame("delivery_confidence", answer_segments)
    if hesitation is None or delivery is None:
        return {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "neutral",
            "calibrated_support_score": 0.0,
            "reason": "Audio support models are unavailable; heuristic fallback remains active.",
        }

    cautionary_score = hesitation["probability_means"].get("high", 0.0) + delivery["probability_means"].get("low", 0.0)
    supportive_score = hesitation["probability_means"].get("low", 0.0) + delivery["probability_means"].get("high", 0.0)
    reliability = float((hesitation["reliability_weight"] + delivery["reliability_weight"]) / 2.0)
    signed_score = round((cautionary_score - supportive_score) * reliability, 4)
    if signed_score >= 0.1:
        support_direction = "cautionary"
    elif signed_score <= -0.1:
        support_direction = "supportive"
    else:
        support_direction = "neutral"

    return {
        "available": True,
        "mode": "model_backed",
        "support_direction": support_direction,
        "calibrated_support_score": signed_score,
        "reliability_weight": round(reliability, 4),
        "hesitation_pressure": {
            "predicted_label": hesitation["predicted_label"],
            "probability_means": hesitation["probability_means"],
        },
        "delivery_confidence": {
            "predicted_label": delivery["predicted_label"],
            "probability_means": delivery["probability_means"],
        },
    }


def score_visual_support(segments_df: pd.DataFrame) -> dict[str, Any]:
    visual_segments = segments_df[
        (segments_df["section"] == "q_and_a")
        & (segments_df["confidence_note"] == "usable visual segment")
    ].copy()
    if visual_segments.empty:
        return {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "unavailable",
            "calibrated_support_score": 0.0,
            "reason": "No usable visual segments were available for model-backed scoring.",
        }

    visual = score_task_frame("visual_tension", visual_segments)
    if visual is None:
        return {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "neutral",
            "calibrated_support_score": 0.0,
            "reason": "Visual support model is unavailable; heuristic fallback remains active.",
        }

    cautionary_score = visual["probability_means"].get("high", 0.0)
    supportive_score = visual["probability_means"].get("low", 0.0)
    reliability = float(visual["reliability_weight"])
    signed_score = round((cautionary_score - supportive_score) * reliability, 4)
    if signed_score >= 0.08:
        support_direction = "cautionary"
    elif signed_score <= -0.08:
        support_direction = "supportive"
    else:
        support_direction = "neutral"
    return {
        "available": True,
        "mode": "model_backed",
        "support_direction": support_direction,
        "calibrated_support_score": signed_score,
        "reliability_weight": round(reliability, 4),
        "visual_tension": {
            "predicted_label": visual["predicted_label"],
            "probability_means": visual["probability_means"],
        },
    }
