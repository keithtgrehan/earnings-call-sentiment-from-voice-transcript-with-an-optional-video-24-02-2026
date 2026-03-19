from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_MANIFEST_DIR = Path("data/source_manifests")
SOURCES_FILE = SOURCE_MANIFEST_DIR / "earnings_call_sources.csv"
SEGMENTS_FILE = SOURCE_MANIFEST_DIR / "earnings_call_segments.csv"

SOURCE_COLUMNS = [
    "source_id",
    "company",
    "ticker",
    "event_title",
    "fiscal_period",
    "event_date",
    "source_family",
    "layout_type",
    "video_url",
    "transcript_url",
    "transcript_source_type",
    "video_source_type",
    "has_prepared_remarks",
    "has_qa",
    "language",
    "face_visibility_expectation",
    "notes",
    "status",
    "license_or_usage_notes",
]

SEGMENT_COLUMNS = [
    "segment_id",
    "source_id",
    "start_time",
    "end_time",
    "segment_type",
    "speaker_name",
    "speaker_role",
    "transcript_ref",
    "face_expected",
    "visual_usability_label",
    "audio_usability_label",
    "labeling_status",
    "notes",
]

ALLOWED_SOURCE_VALUES = {
    "source_family": {
        "official_investor_relations",
        "official_youtube",
        "official_results_page",
        "third_party_repost",
        "transcript_vendor",
        "transcript_only",
    },
    "layout_type": {
        "single_speaker_camera",
        "single_speaker_with_slides",
        "multi_speaker_grid",
        "slides_only",
        "audio_only",
        "transcript_only",
        "unknown",
    },
    "transcript_source_type": {
        "official_transcript",
        "vendor_transcript",
        "auto_transcript",
        "prepared_remarks_only",
        "no_transcript_yet",
    },
    "video_source_type": {
        "official_webcast",
        "official_youtube",
        "third_party_video",
        "audio_only",
        "no_video_yet",
    },
    "has_prepared_remarks": {"true", "false"},
    "has_qa": {"true", "false"},
    "language": {"en", "unknown"},
    "face_visibility_expectation": {"high", "medium", "low", "unknown"},
    "status": {
        "template_example",
        "candidate",
        "planned_collection",
        "ready_to_collect",
        "collected",
        "blocked",
    },
}

ALLOWED_SEGMENT_VALUES = {
    "segment_type": {
        "prepared_remarks",
        "q_and_a_question",
        "q_and_a_answer",
        "intro_or_safe_harbor",
        "closing_remarks",
    },
    "speaker_role": {
        "management",
        "analyst",
        "operator",
        "unknown",
    },
    "face_expected": {"true", "false", "unknown"},
    "visual_usability_label": {
        "",
        "face_visible",
        "single_speaker_visible",
        "multi_speaker_visible",
        "slides_only_or_face_too_small",
        "visual_unusable",
    },
    "audio_usability_label": {
        "",
        "clear_speech",
        "mixed_or_overlapped",
        "audio_unusable",
    },
    "labeling_status": {
        "planned",
        "pending_collection",
        "pending_review",
        "labeled",
        "blocked",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_sources_manifest() -> pd.DataFrame:
    path = repo_root() / SOURCES_FILE
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_segments_manifest() -> pd.DataFrame:
    path = repo_root() / SEGMENTS_FILE
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _validate_columns(frame: pd.DataFrame, expected_columns: list[str], name: str) -> list[str]:
    missing = [column for column in expected_columns if column not in frame.columns]
    return [f"{name} missing column: {column}" for column in missing]


def _validate_allowed_values(
    frame: pd.DataFrame,
    allowed_by_column: dict[str, set[str]],
    name: str,
) -> list[str]:
    errors: list[str] = []
    for column, allowed in allowed_by_column.items():
        if column not in frame.columns:
            continue
        observed = sorted(
            {
                str(value).strip()
                for value in frame[column].tolist()
                if str(value).strip() not in allowed
            }
        )
        errors.extend([f"{name} has invalid {column}: {value}" for value in observed])
    return errors


def _validate_httpish_url(value: str, *, field_name: str, row_id: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if text.startswith(("http://", "https://")):
        return []
    return [f"{field_name} must be blank or http(s) for {row_id}: {text}"]


def _validate_event_date(value: str, *, row_id: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return [f"event_date missing for {row_id}"]
    try:
        pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
    except Exception:
        return [f"event_date must be YYYY-MM-DD for {row_id}: {text}"]
    return []


def _validate_time_value(value: str, *, field_name: str, row_id: str) -> tuple[float | None, list[str]]:
    text = str(value).strip()
    if not text:
        return None, []
    try:
        number = float(text)
    except ValueError:
        return None, [f"{field_name} must be numeric or blank for {row_id}: {text}"]
    if number < 0:
        return None, [f"{field_name} must be non-negative for {row_id}: {text}"]
    return number, []


def validate_source_manifests() -> dict[str, Any]:
    sources = load_sources_manifest()
    segments = load_segments_manifest()
    errors: list[str] = []

    errors.extend(_validate_columns(sources, SOURCE_COLUMNS, "earnings_call_sources"))
    errors.extend(_validate_columns(segments, SEGMENT_COLUMNS, "earnings_call_segments"))
    errors.extend(_validate_allowed_values(sources, ALLOWED_SOURCE_VALUES, "earnings_call_sources"))
    errors.extend(_validate_allowed_values(segments, ALLOWED_SEGMENT_VALUES, "earnings_call_segments"))

    if "source_id" in sources.columns:
        duplicate_source_ids = sorted(
            value
            for value, count in sources["source_id"].value_counts().items()
            if int(count) > 1
        )
        errors.extend([f"duplicate source_id in earnings_call_sources: {value}" for value in duplicate_source_ids])

    if "segment_id" in segments.columns:
        duplicate_segment_ids = sorted(
            value
            for value, count in segments["segment_id"].value_counts().items()
            if int(count) > 1
        )
        errors.extend([f"duplicate segment_id in earnings_call_segments: {value}" for value in duplicate_segment_ids])

    source_ids = set(sources["source_id"].tolist()) if "source_id" in sources.columns else set()
    if "source_id" in segments.columns:
        missing_sources = sorted(set(segments["source_id"].tolist()) - source_ids)
        errors.extend([f"earnings_call_segments references unknown source_id: {value}" for value in missing_sources])

    for _, row in sources.iterrows():
        row_id = str(row.get("source_id", "<missing>"))
        errors.extend(_validate_event_date(str(row.get("event_date", "")), row_id=row_id))
        errors.extend(
            _validate_httpish_url(
                str(row.get("video_url", "")),
                field_name="video_url",
                row_id=row_id,
            )
        )
        errors.extend(
            _validate_httpish_url(
                str(row.get("transcript_url", "")),
                field_name="transcript_url",
                row_id=row_id,
            )
        )

    for _, row in segments.iterrows():
        row_id = str(row.get("segment_id", "<missing>"))
        start_time, start_errors = _validate_time_value(
            str(row.get("start_time", "")),
            field_name="start_time",
            row_id=row_id,
        )
        end_time, end_errors = _validate_time_value(
            str(row.get("end_time", "")),
            field_name="end_time",
            row_id=row_id,
        )
        errors.extend(start_errors)
        errors.extend(end_errors)
        if (start_time is None) != (end_time is None):
            errors.append(f"start_time and end_time must both be blank or both be set for {row_id}")
        if start_time is not None and end_time is not None and end_time <= start_time:
            errors.append(f"end_time must be greater than start_time for {row_id}")

    return {
        "status": "ok" if not errors else "error",
        "source_rows": int(len(sources)),
        "segment_rows": int(len(segments)),
        "source_family_counts": {
            str(key): int(value)
            for key, value in sources["source_family"].value_counts().to_dict().items()
        }
        if "source_family" in sources.columns
        else {},
        "segment_type_counts": {
            str(key): int(value)
            for key, value in segments["segment_type"].value_counts().to_dict().items()
        }
        if "segment_type" in segments.columns
        else {},
        "labeling_status_counts": {
            str(key): int(value)
            for key, value in segments["labeling_status"].value_counts().to_dict().items()
        }
        if "labeling_status" in segments.columns
        else {},
        "errors": errors,
    }


def write_template_files() -> None:
    root = repo_root()
    sources_path = root / SOURCES_FILE
    segments_path = root / SEGMENTS_FILE
    sources_path.parent.mkdir(parents=True, exist_ok=True)

    if not sources_path.exists():
        with sources_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
            writer.writeheader()

    if not segments_path.exists():
        with segments_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SEGMENT_COLUMNS)
            writer.writeheader()
