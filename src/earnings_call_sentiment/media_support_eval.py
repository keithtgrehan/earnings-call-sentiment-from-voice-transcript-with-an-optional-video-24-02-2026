from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

MEDIA_SUPPORT_EVAL_DIR = Path("data/media_support_eval")
MANIFEST_FILE = MEDIA_SUPPORT_EVAL_DIR / "media_manifest.csv"
LABELS_FILE = MEDIA_SUPPORT_EVAL_DIR / "segment_labels.csv"

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

ALLOWED_LABELS = {
    "segment_type": {"prepared", "q_and_a"},
    "media_quality_label": {"poor", "usable", "strong"},
    "hesitation_pressure_label": {"", "low", "medium", "high"},
    "visual_tension_label": {"", "low", "medium", "high"},
    "delivery_confidence_label": {"", "low", "medium", "high"},
    "multimodal_support_direction": {"supportive", "cautionary", "neutral", "unavailable"},
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


def _validate_columns(frame: pd.DataFrame, expected_columns: list[str], name: str) -> list[str]:
    missing = [column for column in expected_columns if column not in frame.columns]
    return [f"{name} missing column: {column}" for column in missing]


def validate_media_support_eval(*, timing_tolerance_s: float = 1.0) -> dict[str, Any]:
    manifest = load_media_manifest()
    labels = load_segment_labels()
    errors: list[str] = []
    errors.extend(_validate_columns(manifest, MANIFEST_COLUMNS, "media_manifest"))
    errors.extend(_validate_columns(labels, LABEL_COLUMNS, "segment_labels"))

    source_ids = set(manifest["source_id"].tolist()) if "source_id" in manifest.columns else set()
    missing_sources = sorted(set(labels["source_id"].tolist()) - source_ids)
    errors.extend([f"segment_labels references unknown source_id: {value}" for value in missing_sources])

    for column, allowed in ALLOWED_LABELS.items():
        if column not in labels.columns:
            continue
        observed = sorted({str(value).strip() for value in labels[column].tolist() if str(value).strip() not in allowed})
        errors.extend([f"segment_labels has invalid {column}: {value}" for value in observed])

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

    return {
        "status": "ok" if not errors else "error",
        "manifest_rows": int(len(manifest)),
        "label_rows": int(len(labels)),
        "label_counts_by_modality": {str(key): int(value) for key, value in label_counts_by_modality.items()},
        "label_distributions": label_distributions,
        "source_call_count": int(labels["source_call_id"].nunique()) if "source_call_id" in labels.columns else 0,
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


def write_template_files() -> None:
    root = repo_root()
    manifest_path = root / MANIFEST_FILE
    labels_path = root / LABELS_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()

    if not labels_path.exists():
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
