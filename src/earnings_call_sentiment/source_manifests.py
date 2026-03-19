from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from urllib import error, request

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

REQUIRED_SOURCE_PAIR_FIELDS = [
    "source_id",
    "company",
    "event_title",
    "event_date",
    "source_family",
    "layout_type",
    "transcript_source_type",
    "video_source_type",
    "status",
]

OFFICIAL_TRANSCRIPT_TYPES = {"official_transcript", "prepared_remarks_only"}
THIRD_PARTY_TRANSCRIPT_TYPES = {"vendor_transcript", "auto_transcript"}
MISSING_TRANSCRIPT_TYPES = {"no_transcript_yet"}

OFFICIAL_VIDEO_TYPES = {"official_webcast", "official_youtube"}
THIRD_PARTY_VIDEO_TYPES = {"third_party_video"}
MISSING_VIDEO_TYPES = {"no_video_yet", "audio_only"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_sources_manifest(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else (repo_root() / SOURCES_FILE)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_segments_manifest(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else (repo_root() / SEGMENTS_FILE)
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


def validate_source_manifests(
    *,
    sources_path: str | Path | None = None,
    segments_path: str | Path | None = None,
) -> dict[str, Any]:
    sources = load_sources_manifest(sources_path)
    segments = load_segments_manifest(segments_path)
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


def _is_template_example(row: pd.Series) -> bool:
    return str(row.get("status", "")).strip().lower() == "template_example"


def _required_source_pair_errors(row: pd.Series) -> list[str]:
    row_id = str(row.get("source_id", "<missing>")).strip() or "<missing>"
    errors: list[str] = []
    for field_name in REQUIRED_SOURCE_PAIR_FIELDS:
        if str(row.get(field_name, "")).strip():
            continue
        errors.append(f"{row_id} missing required pair field: {field_name}")
    return errors


def _transcript_origin(transcript_source_type: str) -> str:
    normalized = str(transcript_source_type).strip().lower()
    if normalized in OFFICIAL_TRANSCRIPT_TYPES:
        return "official"
    if normalized in THIRD_PARTY_TRANSCRIPT_TYPES:
        return "third_party"
    if normalized in MISSING_TRANSCRIPT_TYPES:
        return "missing"
    return "unknown"


def _video_origin(video_source_type: str) -> str:
    normalized = str(video_source_type).strip().lower()
    if normalized in OFFICIAL_VIDEO_TYPES:
        return "official"
    if normalized in THIRD_PARTY_VIDEO_TYPES:
        return "third_party"
    if normalized in MISSING_VIDEO_TYPES:
        return "missing"
    return "unknown"


def _pair_state(
    *,
    transcript_origin: str,
    video_origin: str,
) -> str:
    if transcript_origin == "missing" and video_origin == "missing":
        return "missing_transcript_and_video"
    if transcript_origin == "missing":
        return "missing_transcript"
    if video_origin == "missing":
        return "missing_video"
    if transcript_origin == "third_party":
        return "complete_pair_with_third_party_transcript"
    return "complete_pair"


def _source_pair_warnings(row: pd.Series) -> list[str]:
    row_id = str(row.get("source_id", "<missing>")).strip() or "<missing>"
    warnings: list[str] = []
    transcript_source_type = str(row.get("transcript_source_type", "")).strip().lower()
    video_source_type = str(row.get("video_source_type", "")).strip().lower()
    transcript_url = str(row.get("transcript_url", "")).strip()
    video_url = str(row.get("video_url", "")).strip()
    source_family = str(row.get("source_family", "")).strip().lower()

    if transcript_source_type in MISSING_TRANSCRIPT_TYPES and transcript_url:
        warnings.append(
            f"{row_id} is marked {transcript_source_type} but still has transcript_url metadata."
        )
    if video_source_type in MISSING_VIDEO_TYPES and video_url:
        warnings.append(
            f"{row_id} is marked {video_source_type} but still has video_url metadata."
        )
    if transcript_source_type in THIRD_PARTY_TRANSCRIPT_TYPES:
        warnings.append(
            f"{row_id} uses a third-party transcript type ({transcript_source_type}); keep it clearly secondary to official IR transcript sources."
        )
    if transcript_source_type in OFFICIAL_TRANSCRIPT_TYPES and source_family in {"transcript_vendor", "third_party_repost"}:
        warnings.append(
            f"{row_id} claims an official transcript but source_family={source_family}; confirm the pair metadata manually."
        )
    if video_source_type == "official_youtube" and source_family == "third_party_repost":
        warnings.append(
            f"{row_id} claims official_youtube while source_family=third_party_repost; confirm the video provenance manually."
        )
    return warnings


def _source_pair_errors(row: pd.Series) -> list[str]:
    row_id = str(row.get("source_id", "<missing>")).strip() or "<missing>"
    errors = _required_source_pair_errors(row)
    transcript_source_type = str(row.get("transcript_source_type", "")).strip().lower()
    video_source_type = str(row.get("video_source_type", "")).strip().lower()
    transcript_url = str(row.get("transcript_url", "")).strip()
    video_url = str(row.get("video_url", "")).strip()

    if transcript_source_type in OFFICIAL_TRANSCRIPT_TYPES | THIRD_PARTY_TRANSCRIPT_TYPES and not transcript_url:
        errors.append(
            f"{row_id} transcript_source_type={transcript_source_type} requires transcript_url."
        )
    if video_source_type in OFFICIAL_VIDEO_TYPES | THIRD_PARTY_VIDEO_TYPES and not video_url:
        errors.append(f"{row_id} video_source_type={video_source_type} requires video_url.")
    if transcript_source_type not in OFFICIAL_TRANSCRIPT_TYPES | THIRD_PARTY_TRANSCRIPT_TYPES | MISSING_TRANSCRIPT_TYPES:
        errors.append(
            f"{row_id} transcript_source_type is not clearly classified as official, third-party, or missing."
        )
    if video_source_type not in OFFICIAL_VIDEO_TYPES | THIRD_PARTY_VIDEO_TYPES | MISSING_VIDEO_TYPES:
        errors.append(
            f"{row_id} video_source_type is not clearly classified as official, third-party, or missing."
        )
    return errors


def _pair_row_payload(row: pd.Series) -> dict[str, Any]:
    transcript_source_type = str(row.get("transcript_source_type", "")).strip().lower()
    video_source_type = str(row.get("video_source_type", "")).strip().lower()
    transcript_origin = _transcript_origin(transcript_source_type)
    video_origin = _video_origin(video_source_type)
    return {
        "source_id": str(row.get("source_id", "")).strip(),
        "company": str(row.get("company", "")).strip(),
        "ticker": str(row.get("ticker", "")).strip(),
        "event_title": str(row.get("event_title", "")).strip(),
        "event_date": str(row.get("event_date", "")).strip(),
        "status": str(row.get("status", "")).strip(),
        "source_family": str(row.get("source_family", "")).strip(),
        "layout_type": str(row.get("layout_type", "")).strip(),
        "transcript_source_type": transcript_source_type,
        "transcript_origin": transcript_origin,
        "transcript_url_present": bool(str(row.get("transcript_url", "")).strip()),
        "video_source_type": video_source_type,
        "video_origin": video_origin,
        "video_url_present": bool(str(row.get("video_url", "")).strip()),
        "pair_state": _pair_state(
            transcript_origin=transcript_origin,
            video_origin=video_origin,
        ),
        "likely_third_party_transcript_pair": transcript_origin == "third_party",
        "template_example": _is_template_example(row),
    }


def _safe_url_status(url: str, *, timeout_s: int) -> dict[str, Any]:
    if not url:
        return {"url": "", "status": "blank", "http_status": None}

    for method in ("HEAD", "GET"):
        req = request.Request(url, method=method)
        try:
            with request.urlopen(req, timeout=timeout_s) as response:
                return {
                    "url": url,
                    "status": "ok",
                    "http_status": int(getattr(response, "status", 200)),
                    "method": method,
                }
        except error.HTTPError as exc:
            if method == "HEAD" and exc.code in {400, 403, 405, 429, 500, 501}:
                continue
            return {
                "url": url,
                "status": "http_error",
                "http_status": int(exc.code),
                "method": method,
            }
        except Exception as exc:  # pragma: no cover - network-dependent behavior
            return {
                "url": url,
                "status": "request_error",
                "http_status": None,
                "method": method,
                "error": str(exc),
            }
    return {"url": url, "status": "unchecked", "http_status": None}


def validate_source_pairs(
    *,
    sources_path: str | Path | None = None,
    check_urls: bool = False,
    timeout_s: int = 10,
) -> dict[str, Any]:
    sources = load_sources_manifest(sources_path)
    base_report = validate_source_manifests(sources_path=sources_path)

    pair_errors: list[str] = []
    pair_warnings: list[str] = []
    pair_rows: list[dict[str, Any]] = []
    url_checks: list[dict[str, Any]] = []

    for _, row in sources.iterrows():
        row_errors = _source_pair_errors(row)
        row_warnings = _source_pair_warnings(row)
        if _is_template_example(row):
            pair_warnings.extend(row_warnings)
        else:
            pair_errors.extend(row_errors)
            pair_warnings.extend(row_warnings)

        pair_row = _pair_row_payload(row)
        pair_rows.append(pair_row)

        if check_urls and not pair_row["template_example"]:
            if pair_row["transcript_url_present"]:
                url_checks.append(
                    {
                        "source_id": pair_row["source_id"],
                        "url_kind": "transcript_url",
                        **_safe_url_status(str(row.get("transcript_url", "")).strip(), timeout_s=timeout_s),
                    }
                )
            if pair_row["video_url_present"]:
                url_checks.append(
                    {
                        "source_id": pair_row["source_id"],
                        "url_kind": "video_url",
                        **_safe_url_status(str(row.get("video_url", "")).strip(), timeout_s=timeout_s),
                    }
                )

    complete_pairs = [row["source_id"] for row in pair_rows if row["pair_state"] in {"complete_pair", "complete_pair_with_third_party_transcript"}]
    missing_transcript = [row["source_id"] for row in pair_rows if row["pair_state"] in {"missing_transcript", "missing_transcript_and_video"}]
    missing_video = [row["source_id"] for row in pair_rows if row["pair_state"] in {"missing_video", "missing_transcript_and_video"}]
    third_party_pairs = [row["source_id"] for row in pair_rows if row["likely_third_party_transcript_pair"]]

    status = "ok"
    if base_report["status"] != "ok" or pair_errors:
        status = "error"

    return {
        "status": status,
        "manifest_validation_status": base_report["status"],
        "source_rows": int(len(sources)),
        "summary": {
            "complete_pairs": int(len(complete_pairs)),
            "missing_transcript": int(len(missing_transcript)),
            "missing_video": int(len(missing_video)),
            "likely_third_party_transcript_pairs": int(len(third_party_pairs)),
        },
        "source_ids": {
            "complete_pairs": complete_pairs,
            "missing_transcript": missing_transcript,
            "missing_video": missing_video,
            "likely_third_party_transcript_pairs": third_party_pairs,
        },
        "pair_rows": pair_rows,
        "errors": base_report["errors"] + pair_errors,
        "warnings": pair_warnings,
        "url_checks": url_checks,
        "notes": [
            "Manifest rows are the source of truth for source pairing in this repo.",
            "Official IR transcript metadata is preferred when available.",
            "YouTube or replay video may be paired as supporting visual metadata only; no automatic ingestion is implied.",
        ],
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
