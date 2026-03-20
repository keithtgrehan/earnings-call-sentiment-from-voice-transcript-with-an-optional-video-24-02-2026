#!/usr/bin/env python3
"""Build short manual runtime-check clips for curated visual sidecars.

This helper keeps the canonical source and segment manifests unchanged.
It creates small local clips for a narrow curated slice, writes temporary
clip-only manifests, runs the existing OpenFace sidecar on those clips, and
aggregates the resulting per-clip rows back into source-level visual artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from earnings_call_sentiment.openface_sidecar import (
    OUTPUT_COLUMNS,
    SEGMENT_COLUMNS,
    SOURCE_COLUMNS,
    build_visual_coverage_summary,
    run_openface_feature_sidecar,
)

DEFAULT_INPUTS_ROOT = Path("cache/curated_multimodal_slice")
DEFAULT_OUTPUT_ROOT = Path("data/processed/multimodal/visual")
DEFAULT_SOURCE_MANIFEST = Path("data/source_manifests/earnings_call_sources.csv")
DEFAULT_CLIP_FPS = 0.5
DEFAULT_CLIP_WIDTH = 640
DEFAULT_SOURCE_IDS = [
    "goog_q1_2025_example",
    "bac_q4_2025_example",
    "dis_q1_fy26_example",
    "sbux_prepared_remarks_example",
]

CLIP_PLAN = {
    "goog_q1_2025_example": [
        {
            "clip_id": "prepared_manual_01",
            "segment_id_ref": "goog_q1_2025_seg01",
            "segment_type": "prepared_remarks",
            "speaker_name": "Sundar Pichai",
            "speaker_role": "management",
            "clip_start_s": 300.0,
            "clip_duration_s": 60.0,
            "video_provenance": "official_ir_linked_youtube",
            "notes": "Approximate manual runtime-check clip near early prepared remarks. Not transcript-aligned or timestamp-verified.",
        }
    ],
    "bac_q4_2025_example": [
        {
            "clip_id": "prepared_manual_01",
            "segment_id_ref": "bac_q4_2025_seg01",
            "segment_type": "prepared_remarks",
            "speaker_name": "Brian Moynihan",
            "speaker_role": "management",
            "clip_start_s": 360.0,
            "clip_duration_s": 60.0,
            "video_provenance": "approved_youtube_fallback",
            "notes": "Approximate manual runtime-check clip near early prepared remarks. Not transcript-aligned or timestamp-verified.",
        }
    ],
    "dis_q1_fy26_example": [
        {
            "clip_id": "prepared_manual_01",
            "segment_id_ref": "dis_q1_fy26_seg01",
            "segment_type": "prepared_remarks",
            "speaker_name": "Robert A. Iger",
            "speaker_role": "management",
            "clip_start_s": 900.0,
            "clip_duration_s": 60.0,
            "video_provenance": "approved_youtube_fallback",
            "notes": "Approximate manual runtime-check clip near early management remarks. Not transcript-aligned or timestamp-verified.",
        }
    ],
    "sbux_prepared_remarks_example": [
        {
            "clip_id": "prepared_manual_01",
            "segment_id_ref": "sbux_prepared_remarks_seg02",
            "segment_type": "prepared_remarks",
            "speaker_name": "Unknown management speaker",
            "speaker_role": "management",
            "clip_start_s": 300.0,
            "clip_duration_s": 60.0,
            "video_provenance": "approved_youtube_fallback",
            "notes": "Approximate manual runtime-check clip for prepared remarks. Not transcript-aligned or timestamp-verified.",
        }
    ],
}

CLIP_STATUS_COLUMNS = [
    "source_id",
    "clip_id",
    "segment_id_ref",
    "clip_path",
    "video_path",
    "clip_start_s",
    "clip_end_s",
    "clip_duration_s",
    "timing_basis",
    "video_provenance",
    "clip_built",
    "openface_completed",
    "openface_runtime_seconds",
    "openface_out_dir",
    "status",
    "reason",
    "notes",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build short manual runtime-check clips for curated local videos and "
            "run the existing OpenFace sidecar on those clips."
        )
    )
    parser.add_argument(
        "--inputs-root",
        default=str(DEFAULT_INPUTS_ROOT),
        help="Curated local media root. Expected layout is cache/curated_multimodal_slice/<source_id>/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Base visual output root. Defaults to data/processed/multimodal/visual.",
    )
    parser.add_argument(
        "--source-manifest",
        default=str(DEFAULT_SOURCE_MANIFEST),
        help="Canonical source manifest path.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=DEFAULT_SOURCE_IDS,
        help="Curated source_ids to process.",
    )
    parser.add_argument(
        "--openface-bin",
        default=None,
        help="Optional explicit FeatureExtraction path. Otherwise resolve from environment.",
    )
    return parser.parse_args()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_source_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _load_csv_rows(path)
    return {str(row["source_id"]).strip(): row for row in rows if str(row.get("source_id", "")).strip()}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _video_duration_seconds(path: Path) -> float:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{proc.stderr.strip()}")
    return float(proc.stdout.strip())


def _build_clip(video_path: Path, clip_path: Path, *, start_s: float, duration_s: float) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_s:.3f}",
            "-vf",
            f"fps={DEFAULT_CLIP_FPS},scale={DEFAULT_CLIP_WIDTH}:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg clip build failed for {clip_path}:\n{proc.stderr.strip()}")


def _temp_source_row(base_row: dict[str, str], *, source_id: str, note_suffix: str) -> dict[str, str]:
    row = {column: str(base_row.get(column, "")).strip() for column in SOURCE_COLUMNS}
    row["source_id"] = source_id
    row["layout_type"] = "single_speaker_camera"
    row["face_visibility_expectation"] = "medium"
    row["notes"] = (
        (str(base_row.get("notes", "")).strip() + " | ").strip(" |")
        + " Temporary clip runtime source row for local OpenFace sidecar. "
        + note_suffix
    ).strip()
    return row


def _temp_segment_row(source_id: str, plan_row: dict[str, Any], *, clip_duration_s: float) -> dict[str, str]:
    row = {column: "" for column in SEGMENT_COLUMNS}
    row["segment_id"] = f"{source_id}_{plan_row['clip_id']}"
    row["source_id"] = source_id
    row["start_time"] = "0.0"
    row["end_time"] = f"{clip_duration_s:.3f}"
    row["segment_type"] = str(plan_row["segment_type"])
    row["speaker_name"] = str(plan_row["speaker_name"])
    row["speaker_role"] = str(plan_row["speaker_role"])
    row["transcript_ref"] = (
        f"manual_runtime_check_clip: approximate local clip from "
        f"{float(plan_row['clip_start_s']):.1f}s for {clip_duration_s:.1f}s; not transcript-aligned"
    )
    row["face_expected"] = "true"
    row["visual_usability_label"] = ""
    row["audio_usability_label"] = ""
    row["labeling_status"] = "planned"
    row["notes"] = str(plan_row["notes"]).strip()
    return row


def _remove_source_summary_outputs(source_out_dir: Path) -> None:
    for rel in [
        "segment_visual_features.csv",
        "segment_visual_features.json",
        "visual_coverage_summary.json",
    ]:
        target = source_out_dir / rel
        if target.exists():
            target.unlink()


def _annotated_feature_rows(source_id: str, plan_row: dict[str, Any], clip_path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    suffix = (
        f" | clip_id={plan_row['clip_id']}"
        f" | clip_path={clip_path}"
        f" | clip_start_s={float(plan_row['clip_start_s']):.1f}"
        f" | clip_duration_s={float(plan_row['clip_duration_s']):.1f}"
        f" | timing_basis=approximate_manual_runtime_check"
        f" | video_provenance={plan_row['video_provenance']}"
    )
    annotated: list[dict[str, str]] = []
    for row in rows:
        copy = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
        copy["source_id"] = source_id
        copy["notes"] = (str(copy.get("notes", "")).strip() + suffix).strip()
        annotated.append(copy)
    return annotated


def main() -> int:
    args = parse_args()
    repo_dir = repo_root()
    inputs_root = (repo_dir / args.inputs_root).resolve()
    output_root = (repo_dir / args.output_root).resolve()
    source_manifest_path = (repo_dir / args.source_manifest).resolve()
    source_rows = _load_source_rows(source_manifest_path)
    source_ids = [str(source_id).strip() for source_id in args.source_ids if str(source_id).strip()]

    status_rows: list[dict[str, Any]] = []
    total_full_duration_s = 0.0
    total_clip_duration_s = 0.0
    completed_clip_runs = 0

    for source_id in source_ids:
        if source_id not in CLIP_PLAN:
            raise RuntimeError(f"No clip plan configured for source_id: {source_id}")
        if source_id not in source_rows:
            raise RuntimeError(f"source_id missing from source manifest: {source_id}")

        source_dir = inputs_root / source_id
        video_path = source_dir / "video.mp4"
        if not video_path.exists():
            status_rows.append(
                {
                    "source_id": source_id,
                    "clip_id": "",
                    "segment_id_ref": "",
                    "clip_path": "",
                    "video_path": str(video_path),
                    "clip_start_s": "",
                    "clip_end_s": "",
                    "clip_duration_s": "",
                    "timing_basis": "approximate_manual_runtime_check",
                    "video_provenance": "",
                    "clip_built": "false",
                    "openface_completed": "false",
                    "openface_runtime_seconds": "",
                    "openface_out_dir": "",
                    "status": "missing_video",
                    "reason": "Local video.mp4 is missing for this curated source.",
                    "notes": "",
                }
            )
            continue

        full_duration_s = _video_duration_seconds(video_path)
        total_full_duration_s += full_duration_s
        source_out_dir = output_root / source_id
        clips_root = source_dir / "clips"
        clips_root.mkdir(parents=True, exist_ok=True)
        source_clip_manifest_path = clips_root / "clip_manifest.csv"
        source_temp_manifest_path = clips_root / "visual_runtime_source_manifest.csv"

        temp_source_row = _temp_source_row(
            source_rows[source_id],
            source_id=source_id,
            note_suffix="Canonical manifest stays transcript-first; this temporary row only enables clip-based visual runtime checks.",
        )
        _write_csv(source_temp_manifest_path, SOURCE_COLUMNS, [temp_source_row])

        aggregate_rows: list[dict[str, str]] = []
        source_clip_rows: list[dict[str, Any]] = []
        _remove_source_summary_outputs(source_out_dir)

        for plan_row in CLIP_PLAN[source_id]:
            clip_start_s = float(plan_row["clip_start_s"])
            requested_duration_s = float(plan_row["clip_duration_s"])
            safe_duration_s = min(requested_duration_s, max(1.0, full_duration_s - clip_start_s - 1.0))
            clip_end_s = clip_start_s + safe_duration_s
            total_clip_duration_s += safe_duration_s

            clip_path = clips_root / f"{plan_row['clip_id']}.mp4"
            clip_segment_manifest_path = clips_root / f"{plan_row['clip_id']}_segment_manifest.csv"
            clip_out_dir = source_out_dir / "clips" / str(plan_row["clip_id"])

            status_row = {
                "source_id": source_id,
                "clip_id": str(plan_row["clip_id"]),
                "segment_id_ref": str(plan_row["segment_id_ref"]),
                "clip_path": str(clip_path),
                "video_path": str(video_path),
                "clip_start_s": f"{clip_start_s:.3f}",
                "clip_end_s": f"{clip_end_s:.3f}",
                "clip_duration_s": f"{safe_duration_s:.3f}",
                "timing_basis": "approximate_manual_runtime_check",
                "video_provenance": str(plan_row["video_provenance"]),
                "clip_built": "false",
                "openface_completed": "false",
                "openface_runtime_seconds": "",
                "openface_out_dir": str(clip_out_dir),
                "status": "",
                "reason": "",
                "notes": str(plan_row["notes"]).strip()
                + f" Low-FPS runtime-check clip encoded at {DEFAULT_CLIP_FPS} fps and {DEFAULT_CLIP_WIDTH}px width for tractable local OpenFace extraction.",
            }

            try:
                _build_clip(
                    video_path,
                    clip_path,
                    start_s=clip_start_s,
                    duration_s=safe_duration_s,
                )
                status_row["clip_built"] = "true"

                temp_segment_row = _temp_segment_row(source_id, plan_row, clip_duration_s=safe_duration_s)
                _write_csv(clip_segment_manifest_path, SEGMENT_COLUMNS, [temp_segment_row])

                shutil.rmtree(clip_out_dir, ignore_errors=True)
                started_at = time.monotonic()
                run_openface_feature_sidecar(
                    source_id=source_id,
                    video_path=clip_path,
                    out_dir=clip_out_dir,
                    openface_bin=args.openface_bin,
                    source_manifest_path=source_temp_manifest_path,
                    segment_manifest_path=clip_segment_manifest_path,
                )
                elapsed_s = time.monotonic() - started_at
                status_row["openface_completed"] = "true"
                status_row["openface_runtime_seconds"] = f"{elapsed_s:.3f}"
                status_row["status"] = "completed"
                completed_clip_runs += 1

                clip_feature_rows = _load_csv_rows(clip_out_dir / "segment_visual_features.csv")
                aggregate_rows.extend(_annotated_feature_rows(source_id, plan_row, clip_path, clip_feature_rows))
            except Exception as exc:  # pragma: no cover - runtime sidecar failures are reported directly
                status_row["status"] = "openface_failed"
                status_row["reason"] = str(exc)

            status_rows.append(status_row)
            source_clip_rows.append(status_row)

        _write_csv(source_clip_manifest_path, CLIP_STATUS_COLUMNS, source_clip_rows)

        if aggregate_rows:
            coverage_summary = build_visual_coverage_summary(aggregate_rows)
            coverage_summary["source_id"] = source_id
            coverage_summary["clip_mode"] = "approximate_manual_runtime_check"
            coverage_summary["completed_clip_runs"] = int(
                sum(row["openface_completed"] == "true" for row in source_clip_rows)
            )
            _write_csv(source_out_dir / "segment_visual_features.csv", OUTPUT_COLUMNS, aggregate_rows)
            _write_json(
                source_out_dir / "segment_visual_features.json",
                {
                    "source_id": source_id,
                    "clip_mode": "approximate_manual_runtime_check",
                    "rows": aggregate_rows,
                    "clip_runs": source_clip_rows,
                },
            )
            _write_json(source_out_dir / "visual_coverage_summary.json", coverage_summary)

    status_csv_path = output_root / "curated_clip_run_status.csv"
    status_json_path = output_root / "curated_clip_run_status.json"
    _write_csv(status_csv_path, CLIP_STATUS_COLUMNS, status_rows)
    _write_json(
        status_json_path,
        {
            "status": "ok",
            "source_ids": source_ids,
            "completed_clip_runs": completed_clip_runs,
            "full_video_duration_seconds": round(total_full_duration_s, 3),
            "clip_duration_seconds": round(total_clip_duration_s, 3),
            "saved_duration_seconds": round(total_full_duration_s - total_clip_duration_s, 3),
            "saved_duration_fraction": round(
                (total_full_duration_s - total_clip_duration_s) / total_full_duration_s,
                6,
            )
            if total_full_duration_s
            else 0.0,
            "status_rows": status_rows,
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "source_ids": source_ids,
                "completed_clip_runs": completed_clip_runs,
                "status_csv": str(status_csv_path),
                "status_json": str(status_json_path),
                "full_video_duration_seconds": round(total_full_duration_s, 3),
                "clip_duration_seconds": round(total_clip_duration_s, 3),
                "saved_duration_seconds": round(total_full_duration_s - total_clip_duration_s, 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
