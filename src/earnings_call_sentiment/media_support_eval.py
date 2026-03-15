from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

MEDIA_SUPPORT_EVAL_DIR = Path("data/media_support_eval")
MANIFEST_FILE = MEDIA_SUPPORT_EVAL_DIR / "media_manifest.csv"
LABELS_FILE = MEDIA_SUPPORT_EVAL_DIR / "segment_labels.csv"
YOUTUBE_MANIFEST_FILE = MEDIA_SUPPORT_EVAL_DIR / "youtube_earnings_manifest.csv"
RUNTIME_SMOKE_FILE = MEDIA_SUPPORT_EVAL_DIR / "runtime_smoke_manifest.csv"

MANIFEST_COLUMNS = [
    "source_id",
    "source_call_id",
    "ticker",
    "company",
    "audio_path",
    "video_path",
    "transcript_path",
    "audio_segments_path",
    "visual_segments_path",
    "notes",
]

LABEL_COLUMNS = [
    "item_id",
    "source_id",
    "source_call_id",
    "ticker",
    "feature_modality",
    "feature_artifact_path",
    "segment_id",
    "start_time_s",
    "end_time_s",
    "transcript_text",
    "segment_type",
    "media_quality_label",
    "hesitation_pressure_label",
    "visual_tension_label",
    "delivery_confidence_label",
    "multimodal_support_direction",
    "notes",
]

YOUTUBE_MANIFEST_COLUMNS = [
    "item_id",
    "ticker",
    "company",
    "exchange",
    "year",
    "quarter_or_period",
    "youtube_url",
    "channel_name",
    "source_quality",
    "speaker_visibility",
    "likely_use",
    "notes",
]

RUNTIME_SMOKE_COLUMNS = [
    "source_call_id",
    "source_path_or_url",
    "runtime_success",
    "audio_quality_ok",
    "video_quality_ok",
    "suppression_flags",
    "face_visibility_outcome",
    "notes",
]

ALLOWED_LABELS = {
    "segment_type": {"prepared", "q_and_a"},
    "media_quality_label": {"poor", "usable", "strong"},
    "hesitation_pressure_label": {"", "low", "medium", "high"},
    "visual_tension_label": {"", "low", "medium", "high"},
    "delivery_confidence_label": {"", "low", "medium", "high"},
    "multimodal_support_direction": {"supportive", "cautionary", "neutral", "unavailable"},
}

ALLOWED_YOUTUBE_VALUES = {
    "exchange": {"NYSE", "NASDAQ", "OTHER"},
    "source_quality": {"official", "likely_official", "third_party"},
    "speaker_visibility": {"strong", "mixed", "weak", "unknown"},
    "likely_use": {"smoke_test", "label_candidate", "visual_train_candidate"},
}

ALLOWED_RUNTIME_SMOKE_VALUES = {
    "runtime_success": {"true", "false"},
    "audio_quality_ok": {"true", "false"},
    "video_quality_ok": {"true", "false"},
    "face_visibility_outcome": {"strong", "mixed", "weak", "suppressed", "unavailable"},
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | float | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    return path if path.is_absolute() else (repo_root() / path).resolve()


def load_media_manifest() -> pd.DataFrame:
    path = repo_root() / MANIFEST_FILE
    return pd.read_csv(path, dtype=str).fillna("")


def load_segment_labels() -> pd.DataFrame:
    path = repo_root() / LABELS_FILE
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("segment_id",):
        frame[column] = frame[column].astype(int)
    for column in ("start_time_s", "end_time_s"):
        frame[column] = frame[column].astype(float)
    return frame


def load_youtube_earnings_manifest() -> pd.DataFrame:
    path = repo_root() / YOUTUBE_MANIFEST_FILE
    if not path.exists():
        return pd.DataFrame(columns=YOUTUBE_MANIFEST_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_runtime_smoke_manifest() -> pd.DataFrame:
    path = repo_root() / RUNTIME_SMOKE_FILE
    if not path.exists():
        return pd.DataFrame(columns=RUNTIME_SMOKE_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _validate_columns(frame: pd.DataFrame, expected_columns: list[str], name: str) -> list[str]:
    missing = [column for column in expected_columns if column not in frame.columns]
    return [f"{name} missing column: {column}" for column in missing]


def validate_media_support_eval(*, timing_tolerance_s: float = 1.0) -> dict[str, Any]:
    manifest = load_media_manifest()
    labels = load_segment_labels()
    youtube_manifest = load_youtube_earnings_manifest()
    runtime_smoke = load_runtime_smoke_manifest()
    errors: list[str] = []
    errors.extend(_validate_columns(manifest, MANIFEST_COLUMNS, "media_manifest"))
    errors.extend(_validate_columns(labels, LABEL_COLUMNS, "segment_labels"))
    if not youtube_manifest.empty:
        errors.extend(_validate_columns(youtube_manifest, YOUTUBE_MANIFEST_COLUMNS, "youtube_earnings_manifest"))
    if not runtime_smoke.empty:
        errors.extend(_validate_columns(runtime_smoke, RUNTIME_SMOKE_COLUMNS, "runtime_smoke_manifest"))

    source_ids = set(manifest["source_id"].tolist()) if "source_id" in manifest.columns else set()
    missing_sources = sorted(set(labels["source_id"].tolist()) - source_ids)
    errors.extend([f"segment_labels references unknown source_id: {value}" for value in missing_sources])

    for column, allowed in ALLOWED_LABELS.items():
        if column not in labels.columns:
            continue
        observed = sorted({str(value).strip() for value in labels[column].tolist() if str(value).strip() not in allowed})
        errors.extend([f"segment_labels has invalid {column}: {value}" for value in observed])

    for column, allowed in ALLOWED_YOUTUBE_VALUES.items():
        if column not in youtube_manifest.columns:
            continue
        observed = sorted(
            {
                str(value).strip()
                for value in youtube_manifest[column].tolist()
                if str(value).strip() not in allowed
            }
        )
        errors.extend([f"youtube_earnings_manifest has invalid {column}: {value}" for value in observed])

    for column, allowed in ALLOWED_RUNTIME_SMOKE_VALUES.items():
        if column not in runtime_smoke.columns:
            continue
        observed = sorted(
            {
                str(value).strip().lower()
                for value in runtime_smoke[column].tolist()
                if str(value).strip().lower() not in allowed
            }
        )
        errors.extend([f"runtime_smoke_manifest has invalid {column}: {value}" for value in observed])

    for _, row in manifest.iterrows():
        for path_column in (
            "audio_path",
            "video_path",
            "transcript_path",
            "audio_segments_path",
            "visual_segments_path",
        ):
            path = _resolve_path(row.get(path_column))
            if path is None:
                continue
            if not path.exists():
                errors.append(f"manifest path missing for {row['source_id']} {path_column}: {path}")

    label_counts_by_modality = labels["feature_modality"].value_counts().to_dict() if "feature_modality" in labels.columns else {}
    label_distributions: dict[str, dict[str, int]] = {}
    for column in (
        "media_quality_label",
        "hesitation_pressure_label",
        "visual_tension_label",
        "delivery_confidence_label",
        "multimodal_support_direction",
    ):
        if column in labels.columns:
            series = labels[column].replace("", "<blank>")
            label_distributions[column] = {str(key): int(value) for key, value in series.value_counts().to_dict().items()}

    for _, label in labels.iterrows():
        artifact_path = _resolve_path(label.get("feature_artifact_path"))
        if artifact_path is None or not artifact_path.exists():
            errors.append(f"feature artifact missing for {label['item_id']}: {artifact_path}")
            continue
        try:
            frame = pd.read_csv(artifact_path)
        except Exception as exc:
            errors.append(f"unable to read feature artifact for {label['item_id']}: {exc}")
            continue
        if "segment_id" not in frame.columns:
            errors.append(f"feature artifact missing segment_id column for {label['item_id']}: {artifact_path}")
            continue
        matched = frame[frame["segment_id"] == int(label["segment_id"])]
        if matched.empty:
            errors.append(f"segment_id {label['segment_id']} not found for {label['item_id']} in {artifact_path}")
            continue
        row = matched.iloc[0]
        if "start_time_s" in row and abs(float(row["start_time_s"]) - float(label["start_time_s"])) > timing_tolerance_s:
            errors.append(f"start_time mismatch for {label['item_id']}: label={label['start_time_s']} artifact={row['start_time_s']}")
        if "end_time_s" in row and abs(float(row["end_time_s"]) - float(label["end_time_s"])) > timing_tolerance_s:
            errors.append(f"end_time mismatch for {label['item_id']}: label={label['end_time_s']} artifact={row['end_time_s']}")

    rows_by_source_call_id = (
        {str(key): int(value) for key, value in labels["source_call_id"].value_counts().to_dict().items()}
        if "source_call_id" in labels.columns
        else {}
    )
    rows_by_segment_type = (
        {str(key): int(value) for key, value in labels["segment_type"].value_counts().to_dict().items()}
        if "segment_type" in labels.columns
        else {}
    )
    audio_training_groups = int(
        labels[
            (labels["feature_modality"] == "audio")
            & (
                (labels["hesitation_pressure_label"].astype(str).str.strip() != "")
                | (labels["delivery_confidence_label"].astype(str).str.strip() != "")
            )
        ]["source_call_id"].nunique()
    )
    video_training_groups = int(
        labels[
            (labels["feature_modality"] == "video")
            & (labels["visual_tension_label"].astype(str).str.strip() != "")
        ]["source_call_id"].nunique()
    )

    return {
        "status": "ok" if not errors else "error",
        "manifest_rows": int(len(manifest)),
        "label_rows": int(len(labels)),
        "youtube_candidate_rows": int(len(youtube_manifest)),
        "runtime_smoke_rows": int(len(runtime_smoke)),
        "label_counts_by_modality": {str(key): int(value) for key, value in label_counts_by_modality.items()},
        "rows_by_source_call_id": rows_by_source_call_id,
        "rows_by_segment_type": rows_by_segment_type,
        "label_distributions": label_distributions,
        "source_call_count": int(labels["source_call_id"].nunique()) if "source_call_id" in labels.columns else 0,
        "training_group_counts": {
            "audio_support": audio_training_groups,
            "video_support": video_training_groups,
        },
        "errors": errors,
    }


def load_labeled_feature_frame(
    *,
    feature_modality: str,
    label_column: str,
) -> pd.DataFrame:
    labels = load_segment_labels()
    filtered = labels[
        (labels["feature_modality"] == feature_modality)
        & (labels[label_column].astype(str).str.strip() != "")
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    cache: dict[Path, pd.DataFrame] = {}
    for _, label in filtered.iterrows():
        artifact_path = _resolve_path(label["feature_artifact_path"])
        if artifact_path is None:
            continue
        if artifact_path not in cache:
            cache[artifact_path] = pd.read_csv(artifact_path)
        artifact = cache[artifact_path]
        matched = artifact[artifact["segment_id"] == int(label["segment_id"])]
        if matched.empty:
            continue
        feature_row = matched.iloc[0].to_dict()
        feature_row.update({column: label[column] for column in LABEL_COLUMNS})
        rows.append(feature_row)

    return pd.DataFrame(rows)


def build_visual_trainability_report() -> dict[str, Any]:
    labels = load_segment_labels()
    video_rows = labels[labels["feature_modality"] == "video"].copy()
    labeled_rows = video_rows[video_rows["visual_tension_label"].astype(str).str.strip() != ""].copy()
    class_counts = {
        str(key): int(value)
        for key, value in labeled_rows["visual_tension_label"].value_counts().to_dict().items()
    }
    rows_by_group = {
        str(key): int(value)
        for key, value in labeled_rows["source_call_id"].value_counts().to_dict().items()
    }
    group_count = int(labeled_rows["source_call_id"].nunique()) if not labeled_rows.empty else 0
    class_count = int(len(class_counts))
    min_class_examples = min(class_counts.values()) if class_counts else 0

    basic_grouped_eval_ready = group_count >= 2 and class_count >= 2
    defensible_grouped_eval_ready = group_count >= 3 and class_count >= 2
    calibration_ready = group_count >= 3 and class_count >= 2 and min_class_examples >= 3

    blockers: list[str] = []
    if group_count < 2:
        blockers.append(
            f"Only {group_count} source group has nonblank visual_tension labels; grouped train/validation needs at least 2."
        )
    if group_count < 3:
        blockers.append(
            f"Only {group_count} source group has nonblank visual_tension labels; a more defensible grouped evaluation target is at least 3."
        )
    if class_count < 2:
        blockers.append("Visual labels do not span enough classes to train even a binary support scorer.")
    if min_class_examples < 3:
        blockers.append(
            "At least one visual_tension class has fewer than 3 labeled examples, so calibrated probability estimates are not supportable."
        )

    return {
        "video_label_rows_total": int(len(video_rows)),
        "video_label_rows_with_visual_tension": int(len(labeled_rows)),
        "source_groups_with_any_video_rows": int(video_rows["source_call_id"].nunique()) if not video_rows.empty else 0,
        "source_groups_with_visual_tension_labels": group_count,
        "class_counts": class_counts,
        "rows_by_group": rows_by_group,
        "basic_grouped_eval_ready": basic_grouped_eval_ready,
        "defensible_grouped_eval_ready": defensible_grouped_eval_ready,
        "calibration_ready": calibration_ready,
        "blockers": blockers,
        "minimum_next_data": {
            "additional_groups_for_basic_grouped_eval": max(0, 2 - group_count),
            "additional_groups_for_defensible_grouped_eval": max(0, 3 - group_count),
            "additional_examples_per_class_for_calibration": {
                label: max(0, 3 - int(count))
                for label, count in class_counts.items()
            },
        },
    }


def write_template_files() -> None:
    root = repo_root()
    manifest_path = root / MANIFEST_FILE
    labels_path = root / LABELS_FILE
    youtube_manifest_path = root / YOUTUBE_MANIFEST_FILE
    runtime_smoke_path = root / RUNTIME_SMOKE_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()

    if not labels_path.exists():
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()

    if not youtube_manifest_path.exists():
        with youtube_manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=YOUTUBE_MANIFEST_COLUMNS)
            writer.writeheader()

    if not runtime_smoke_path.exists():
        with runtime_smoke_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUNTIME_SMOKE_COLUMNS)
            writer.writeheader()
