"""Optional OpenFace sidecar utilities for conservative visual feature summaries.

This module is intentionally low-level:
- no psychological or deception labels
- no scoring integration
- no default pipeline changes
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd

from earnings_call_sentiment.optional_runtime import load_multimodal_config

DEFAULT_VISUAL_DIR = Path("data/processed/multimodal/visual")

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

OUTPUT_COLUMNS = [
    "source_id",
    "segment_id",
    "segment_type",
    "speaker_name",
    "speaker_role",
    "start_time_s",
    "end_time_s",
    "layout_type",
    "face_expected",
    "attempted",
    "extraction_succeeded",
    "missing_tool_flag",
    "segment_visual_usability",
    "usability_reason",
    "frames_total",
    "frames_with_face",
    "face_detection_rate",
    "mean_face_confidence",
    "mean_head_pose_change",
    "gaze_variability_proxy",
    "blink_or_eye_closure_proxy",
    "au_intensity_mean",
    "au_intensity_supported",
    "extraction_errors",
    "notes",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_visual_dir(*, source_id: str, out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        base = Path(out_dir)
    else:
        base = repo_root() / DEFAULT_VISUAL_DIR
    return base.expanduser().resolve() / source_id


def _load_manifest(path: Path, expected_columns: list[str], name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {', '.join(missing)}")
    return frame


def load_source_row(
    *,
    source_id: str,
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    path = source_manifest_path or (repo_root() / "data/source_manifests/earnings_call_sources.csv")
    frame = _load_manifest(path, SOURCE_COLUMNS, "source manifest")
    matched = frame[frame["source_id"].astype(str) == str(source_id)].copy()
    if matched.empty:
        raise RuntimeError(f"source_id not found in source manifest: {source_id}")
    return matched.iloc[0].to_dict()


def load_segment_rows(
    *,
    source_id: str,
    segment_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    path = segment_manifest_path or (repo_root() / "data/source_manifests/earnings_call_segments.csv")
    frame = _load_manifest(path, SEGMENT_COLUMNS, "segment manifest")
    matched = frame[frame["source_id"].astype(str) == str(source_id)].copy()
    return matched.to_dict(orient="records")


def resolve_openface_binary(openface_bin: str | None = None) -> str:
    if openface_bin:
        candidate = str(Path(openface_bin).expanduser())
    else:
        config = load_multimodal_config()
        candidate = str(config.openface_bin or "").strip()

    if not candidate:
        raise RuntimeError(
            "OpenFace binary is not configured. Pass --openface-bin or set "
            "EARNINGS_CALL_OPENFACE_BIN / EARNINGS_CALL_OPENFACE_ROOT."
        )

    path_candidate = Path(candidate).expanduser()
    if path_candidate.exists():
        return str(path_candidate.resolve())

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    raise RuntimeError(
        "OpenFace binary was configured but not found. "
        f"Checked: {candidate}. "
        "Set EARNINGS_CALL_OPENFACE_BIN to the FeatureExtraction executable or "
        "ensure it is on PATH."
    )


def _run_openface(
    *,
    openface_bin: str,
    video_path: Path,
    raw_out_dir: Path,
) -> Path:
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        openface_bin,
        "-f",
        str(video_path),
        "-out_dir",
        str(raw_out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        message = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        raise RuntimeError(f"OpenFace FeatureExtraction failed:\n{message}")

    expected = raw_out_dir / f"{video_path.stem}.csv"
    if expected.exists():
        return expected

    csv_files = sorted(raw_out_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            "OpenFace completed without producing a CSV output in "
            f"{raw_out_dir}"
        )
    return csv_files[0]


def _normalize_openface_frame_df(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "timestamp" not in frame.columns:
        raise RuntimeError(f"OpenFace output missing timestamp column: {csv_path}")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _subset_segment_window(frame_df: pd.DataFrame, start_time_s: float, end_time_s: float) -> pd.DataFrame:
    return frame_df[
        (frame_df["timestamp"] >= float(start_time_s))
        & (frame_df["timestamp"] <= float(end_time_s))
    ].copy()


def _success_mask(window: pd.DataFrame) -> pd.Series:
    if "success" in window.columns:
        return pd.to_numeric(window["success"], errors="coerce").fillna(0.0) > 0.5
    if "confidence" in window.columns:
        return pd.to_numeric(window["confidence"], errors="coerce").fillna(0.0) > 0.0
    return pd.Series([True] * len(window), index=window.index)


def _available_columns(frame: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


def _mean_abs_diff(frame: pd.DataFrame, columns: list[str]) -> float | None:
    available = _available_columns(frame, columns)
    if not available or len(frame) < 2:
        return None
    diffs = frame[available].diff().abs().dropna()
    if diffs.empty:
        return None
    return round(float(diffs.mean().mean()), 6)


def _mean_std(frame: pd.DataFrame, columns: list[str]) -> float | None:
    available = _available_columns(frame, columns)
    if not available or frame.empty:
        return None
    return round(float(frame[available].std(ddof=0).mean()), 6)


def _mean_value(frame: pd.DataFrame, columns: list[str]) -> float | None:
    available = _available_columns(frame, columns)
    if not available or frame.empty:
        return None
    return round(float(frame[available].mean().mean()), 6)


def _au_intensity_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column).startswith("AU") and str(column).endswith("_r")
    ]


def _segment_row_from_window(
    *,
    source_id: str,
    layout_type: str,
    segment: dict[str, Any],
    window: pd.DataFrame | None,
    attempted: bool,
    extraction_succeeded: bool,
    usability: str,
    reason: str,
    extraction_errors: str = "",
) -> dict[str, Any]:
    frames_total = int(len(window)) if window is not None else 0
    success_window = None
    if window is not None and not window.empty:
        success_window = window[_success_mask(window)].copy()
    frames_with_face = int(len(success_window)) if success_window is not None else 0
    detection_rate = round(float(frames_with_face / frames_total), 6) if frames_total else 0.0
    confidence = None
    if success_window is not None and not success_window.empty and "confidence" in success_window.columns:
        confidence = round(float(pd.to_numeric(success_window["confidence"], errors="coerce").fillna(0.0).mean()), 6)

    head_pose_change = None
    gaze_variability = None
    blink_proxy = None
    au_mean = None
    au_supported = False

    if success_window is not None and not success_window.empty:
        numeric_window = success_window.apply(pd.to_numeric, errors="coerce")
        head_pose_change = _mean_abs_diff(numeric_window, ["pose_Rx", "pose_Ry", "pose_Rz"])
        gaze_variability = _mean_std(
            numeric_window,
            [
                "gaze_0_x",
                "gaze_0_y",
                "gaze_0_z",
                "gaze_1_x",
                "gaze_1_y",
                "gaze_1_z",
            ],
        )
        blink_proxy = _mean_value(numeric_window, ["AU45_r", "AU45_c"])
        au_cols = _au_intensity_columns(numeric_window)
        if au_cols:
            au_supported = True
            au_mean = _mean_value(numeric_window, au_cols)

    return {
        "source_id": source_id,
        "segment_id": str(segment.get("segment_id", "")),
        "segment_type": str(segment.get("segment_type", "")),
        "speaker_name": str(segment.get("speaker_name", "")),
        "speaker_role": str(segment.get("speaker_role", "")),
        "start_time_s": _float_or_none(segment.get("start_time")),
        "end_time_s": _float_or_none(segment.get("end_time")),
        "layout_type": layout_type,
        "face_expected": str(segment.get("face_expected", "")),
        "attempted": _bool_text(attempted),
        "extraction_succeeded": _bool_text(extraction_succeeded),
        "missing_tool_flag": "false",
        "segment_visual_usability": usability,
        "usability_reason": reason,
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "face_detection_rate": detection_rate,
        "mean_face_confidence": confidence,
        "mean_head_pose_change": head_pose_change,
        "gaze_variability_proxy": gaze_variability,
        "blink_or_eye_closure_proxy": blink_proxy,
        "au_intensity_mean": au_mean,
        "au_intensity_supported": _bool_text(au_supported),
        "extraction_errors": extraction_errors,
        "notes": str(segment.get("notes", "")),
    }


def summarize_source_segments(
    *,
    source_id: str,
    source_row: dict[str, Any],
    segment_rows: list[dict[str, Any]],
    frame_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    layout_type = str(source_row.get("layout_type", "unknown"))
    rows: list[dict[str, Any]] = []

    for segment in segment_rows:
        face_expected = str(segment.get("face_expected", "")).strip().lower()
        start_time = _float_or_none(segment.get("start_time"))
        end_time = _float_or_none(segment.get("end_time"))

        if face_expected != "true":
            rows.append(
                _segment_row_from_window(
                    source_id=source_id,
                    layout_type=layout_type,
                    segment=segment,
                    window=None,
                    attempted=False,
                    extraction_succeeded=False,
                    usability="skipped",
                    reason="no_face_expected",
                )
            )
            continue

        if layout_type in {"slides_only", "audio_only", "transcript_only"}:
            rows.append(
                _segment_row_from_window(
                    source_id=source_id,
                    layout_type=layout_type,
                    segment=segment,
                    window=None,
                    attempted=False,
                    extraction_succeeded=False,
                    usability="unusable",
                    reason="slides_only_or_no_face_layout",
                )
            )
            continue

        if start_time is None or end_time is None or end_time <= start_time:
            rows.append(
                _segment_row_from_window(
                    source_id=source_id,
                    layout_type=layout_type,
                    segment=segment,
                    window=None,
                    attempted=False,
                    extraction_succeeded=False,
                    usability="skipped",
                    reason="missing_segment_times",
                )
            )
            continue

        window = _subset_segment_window(frame_df, start_time, end_time)
        if window.empty:
            rows.append(
                _segment_row_from_window(
                    source_id=source_id,
                    layout_type=layout_type,
                    segment=segment,
                    window=window,
                    attempted=True,
                    extraction_succeeded=False,
                    usability="unusable",
                    reason="no_frames_in_segment_window",
                )
            )
            continue

        success_window = window[_success_mask(window)].copy()
        frames_total = len(window)
        frames_with_face = len(success_window)
        detection_rate = float(frames_with_face / frames_total) if frames_total else 0.0

        usability = "usable"
        reason = "ok"
        extraction_succeeded = True

        if detection_rate < 0.15:
            usability = "unusable"
            reason = "face_too_small_or_intermittent"
        elif layout_type == "multi_speaker_grid":
            usability = "limited"
            reason = "multi_speaker_layout_limited"
        elif detection_rate < 0.4:
            usability = "limited"
            reason = "face_too_small_or_intermittent"

        rows.append(
            _segment_row_from_window(
                source_id=source_id,
                layout_type=layout_type,
                segment=segment,
                window=window,
                attempted=True,
                extraction_succeeded=extraction_succeeded,
                usability=usability,
                reason=reason,
            )
        )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_visual_coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "segments_total": 0,
            "segments_attempted": 0,
            "segments_usable": 0,
            "segments_limited": 0,
            "segments_unusable": 0,
            "segments_skipped": 0,
            "extraction_success_rate": 0.0,
            "source_group_coverage": 0,
        }

    attempted_mask = frame["attempted"].astype(str).str.lower() == "true"
    succeeded_mask = frame["extraction_succeeded"].astype(str).str.lower() == "true"
    usable_mask = frame["segment_visual_usability"].astype(str) == "usable"
    limited_mask = frame["segment_visual_usability"].astype(str) == "limited"
    unusable_mask = frame["segment_visual_usability"].astype(str) == "unusable"
    skipped_mask = frame["segment_visual_usability"].astype(str) == "skipped"

    attempted_count = int(attempted_mask.sum())
    succeeded_count = int((attempted_mask & succeeded_mask).sum())
    return {
        "segments_total": int(len(frame)),
        "segments_attempted": attempted_count,
        "segments_usable": int(usable_mask.sum()),
        "segments_limited": int(limited_mask.sum()),
        "segments_unusable": int(unusable_mask.sum()),
        "segments_skipped": int(skipped_mask.sum()),
        "extraction_success_rate": round(
            float(succeeded_count / attempted_count),
            6,
        )
        if attempted_count
        else 0.0,
        "source_group_coverage": int(frame["source_id"].astype(str).nunique()),
        "reason_breakdown": {
            str(key): int(value)
            for key, value in frame["usability_reason"].astype(str).value_counts().to_dict().items()
        },
    }


def run_openface_feature_sidecar(
    *,
    source_id: str,
    video_path: Path,
    out_dir: Path,
    openface_bin: str | None = None,
    source_manifest_path: Path | None = None,
    segment_manifest_path: Path | None = None,
) -> dict[str, Any]:
    resolved_bin = resolve_openface_binary(openface_bin)
    resolved_video = video_path.expanduser().resolve()
    if not resolved_video.exists():
        raise RuntimeError(f"Video file not found: {resolved_video}")

    source_row = load_source_row(source_id=source_id, source_manifest_path=source_manifest_path)
    segment_rows = load_segment_rows(source_id=source_id, segment_manifest_path=segment_manifest_path)
    if not segment_rows:
        raise RuntimeError(f"No segment rows found for source_id: {source_id}")

    raw_out_dir = out_dir / "openface_raw"
    raw_csv_path = _run_openface(
        openface_bin=resolved_bin,
        video_path=resolved_video,
        raw_out_dir=raw_out_dir,
    )
    frame_df = _normalize_openface_frame_df(raw_csv_path)
    rows = summarize_source_segments(
        source_id=source_id,
        source_row=source_row,
        segment_rows=segment_rows,
        frame_df=frame_df,
    )
    coverage_summary = build_visual_coverage_summary(rows)

    _write_csv(out_dir / "segment_visual_features.csv", rows)
    _write_json(
        out_dir / "segment_visual_features.json",
        {
            "source_id": source_id,
            "video_path": str(resolved_video),
            "openface_csv_path": str(raw_csv_path),
            "rows": rows,
        },
    )
    summary_payload = {
        "source_id": source_id,
        "video_path": str(resolved_video),
        "openface_binary": resolved_bin,
        "openface_csv_path": str(raw_csv_path),
        **coverage_summary,
        "notes": [
            "These are conservative low-level visual summaries only.",
            "No psychological, deception, or truthfulness labels are produced.",
            "Outputs are sidecar artifacts and are not wired into deterministic scoring.",
        ],
    }
    _write_json(out_dir / "visual_coverage_summary.json", summary_payload)
    return summary_payload
