#!/usr/bin/env python3
"""Run the smallest practical multimodal sidecar slice for curated sources.

This helper does not download or discover anything. It only looks for manually
placed local files, runs the existing sidecar scripts when inputs are present,
skips missing modalities safely, and refreshes the multimodal eval summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from earnings_call_sentiment.optional_runtime import load_multimodal_config

DEFAULT_SOURCE_MANIFEST = Path("data/source_manifests/earnings_call_sources.csv")
DEFAULT_SEGMENT_MANIFEST = Path("data/source_manifests/earnings_call_segments.csv")
DEFAULT_INPUTS_ROOT = Path("cache/curated_multimodal_slice")
DEFAULT_STATUS_DIR = Path("data/processed/multimodal/eval")
DEFAULT_STATUS_BASENAME = "curated_slice_run_status"
DEFAULT_ALIGNMENT_OUTPUT_ROOT = Path("data/processed/multimodal/alignment")
DEFAULT_VISUAL_OUTPUT_ROOT = Path("data/processed/multimodal/visual")
DEFAULT_SOURCE_IDS = [
    "goog_q1_2025_example",
    "msft_fy26_q2_example",
    "bac_q4_2025_example",
    "dis_q1_fy26_example",
    "sbux_prepared_remarks_example",
]

AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".webm", ".m4v"]
TRANSCRIPT_FILENAMES = ["transcript.json", "transcript.txt"]
CHUNKS_FILENAMES = ["chunks_scored.csv", "chunks_scored.jsonl"]

STATUS_COLUMNS = [
    "source_id",
    "audio_found",
    "video_found",
    "transcript_found",
    "chunks_found",
    "alignment_ran",
    "visual_ran",
    "nlp_ran",
    "alignment_status",
    "visual_status",
    "nlp_status",
    "skip_reason",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run optional sidecar coverage for the curated real-source slice "
            "without downloading any inputs."
        )
    )
    parser.add_argument(
        "--inputs-root",
        default=str(DEFAULT_INPUTS_ROOT),
        help=(
            "Local manual-input root. Expected layout is "
            "cache/curated_multimodal_slice/<source_id>/."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        default=str(DEFAULT_SOURCE_MANIFEST),
        help="Source manifest path.",
    )
    parser.add_argument(
        "--segment-manifest",
        default=str(DEFAULT_SEGMENT_MANIFEST),
        help="Segment manifest path.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=DEFAULT_SOURCE_IDS,
        help="Curated source_ids to process. Defaults to the four prioritized real rows.",
    )
    parser.add_argument(
        "--openface-bin",
        default=None,
        help="Optional explicit OpenFace FeatureExtraction path or executable name.",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Pass through experimental pyannote diarization to the alignment sidecar.",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip WhisperX alignment attempts and only refresh other sidecar statuses.",
    )
    parser.add_argument(
        "--run-secondary-emotion",
        action="store_true",
        help="Also run the secondary emotion model during NLP sidecar scoring.",
    )
    parser.add_argument(
        "--status-dir",
        default=str(DEFAULT_STATUS_DIR),
        help="Directory for curated-slice status CSV/JSON outputs.",
    )
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _find_named_file(base_dir: Path, filenames: list[str]) -> Path | None:
    for filename in filenames:
        candidate = base_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _find_stemmed_file(base_dir: Path, stem: str, extensions: list[str]) -> Path | None:
    for extension in extensions:
        candidate = base_dir / f"{stem}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_segment_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source_id = _clean_text(row.get("source_id"))
        grouped.setdefault(source_id, []).append(row)
    return grouped


def _looks_like_placeholder(text: str) -> bool:
    lowered = _clean_text(text).lower()
    return lowered.startswith("planned:") or lowered.startswith("todo:")


def _has_visual_ready_segments(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        face_expected = _clean_text(row.get("face_expected")).lower()
        start_time = _clean_text(row.get("start_time"))
        end_time = _clean_text(row.get("end_time"))
        if face_expected == "true" and start_time and end_time:
            return True
    return False


def _has_nlp_ready_manifest_segments(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        transcript_ref = _clean_text(row.get("transcript_ref"))
        if transcript_ref and not _looks_like_placeholder(transcript_ref):
            return True
        start_time = _clean_text(row.get("start_time"))
        end_time = _clean_text(row.get("end_time"))
        if start_time and end_time:
            return True
    return False


def _ensure_source_ids_exist(source_rows: list[dict[str, str]], source_ids: list[str]) -> None:
    available = {_clean_text(row.get("source_id")) for row in source_rows}
    missing = [source_id for source_id in source_ids if source_id not in available]
    if missing:
        raise RuntimeError(
            "Requested source_ids are missing from the source manifest: "
            + ", ".join(missing)
        )


def _python_env(repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str((repo_dir / "src").resolve())
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def _run_sidecar_command(repo_dir: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=repo_dir,
        env=_python_env(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )


def _openface_is_configured(explicit_bin: str | None) -> bool:
    if explicit_bin and _clean_text(explicit_bin):
        return True
    config = load_multimodal_config()
    return bool(config.openface_bin)


def _existing_alignment_outputs_path(repo_dir: Path, source_id: str) -> Path:
    return (repo_dir / DEFAULT_ALIGNMENT_OUTPUT_ROOT / source_id / "alignment_summary.json").resolve()


def _existing_visual_outputs_path(repo_dir: Path, source_id: str) -> Path:
    return (repo_dir / DEFAULT_VISUAL_OUTPUT_ROOT / source_id / "segment_visual_features.csv").resolve()


def _status_row_base(source_id: str, *, audio_found: bool, video_found: bool, transcript_found: bool, chunks_found: bool) -> dict[str, str]:
    return {
        "source_id": source_id,
        "audio_found": _bool_text(audio_found),
        "video_found": _bool_text(video_found),
        "transcript_found": _bool_text(transcript_found),
        "chunks_found": _bool_text(chunks_found),
        "alignment_ran": "false",
        "visual_ran": "false",
        "nlp_ran": "false",
        "alignment_status": "",
        "visual_status": "",
        "nlp_status": "",
        "skip_reason": "",
    }


def _join_reasons(*values: str) -> str:
    reasons = [value for value in values if value and value != "ran"]
    return "; ".join(reasons)


def _write_status_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_status_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_dir = repo_root()
    source_manifest_path = (repo_dir / args.source_manifest).resolve()
    segment_manifest_path = (repo_dir / args.segment_manifest).resolve()
    inputs_root = (repo_dir / args.inputs_root).resolve()
    status_dir = (repo_dir / args.status_dir).resolve()
    status_csv_path = status_dir / f"{DEFAULT_STATUS_BASENAME}.csv"
    status_json_path = status_dir / f"{DEFAULT_STATUS_BASENAME}.json"

    source_rows = _load_manifest_rows(source_manifest_path)
    segment_rows_by_source = _group_segment_rows(_load_manifest_rows(segment_manifest_path))
    source_ids = [str(source_id).strip() for source_id in args.source_ids if str(source_id).strip()]
    _ensure_source_ids_exist(source_rows, source_ids)

    status_rows: list[dict[str, str]] = []

    for source_id in source_ids:
        source_dir = inputs_root / source_id
        segment_rows = segment_rows_by_source.get(source_id, [])

        audio_path = _find_stemmed_file(source_dir, "audio", AUDIO_EXTENSIONS)
        video_path = _find_stemmed_file(source_dir, "video", VIDEO_EXTENSIONS)
        transcript_path = _find_named_file(source_dir, TRANSCRIPT_FILENAMES)
        chunks_path = _find_named_file(source_dir, CHUNKS_FILENAMES)

        row = _status_row_base(
            source_id,
            audio_found=audio_path is not None,
            video_found=video_path is not None,
            transcript_found=transcript_path is not None,
            chunks_found=chunks_path is not None,
        )

        existing_alignment_outputs = _existing_alignment_outputs_path(repo_dir, source_id)
        if existing_alignment_outputs.exists():
            row["alignment_ran"] = "true"
            row["alignment_status"] = "existing_outputs"
        elif args.skip_alignment:
            row["alignment_status"] = "skipped_by_flag"
        elif audio_path is None:
            row["alignment_status"] = "skipped_missing_audio"
        else:
            alignment_cmd = [
                sys.executable,
                "scripts/run_whisperx_alignment.py",
                "--source-id",
                source_id,
                "--audio-path",
                str(audio_path),
            ]
            if transcript_path is not None:
                alignment_cmd.extend(["--transcript-path", str(transcript_path)])
            if args.enable_diarization:
                alignment_cmd.append("--enable-diarization")

            alignment_proc = _run_sidecar_command(repo_dir, alignment_cmd)
            if alignment_proc.returncode == 0:
                row["alignment_ran"] = "true"
                row["alignment_status"] = "ran"
            else:
                row["alignment_status"] = f"error_exit_{alignment_proc.returncode}"

        visual_ready = _has_visual_ready_segments(segment_rows)
        existing_visual_outputs = _existing_visual_outputs_path(repo_dir, source_id)
        if existing_visual_outputs.exists():
            row["visual_ran"] = "true"
            row["visual_status"] = "existing_outputs"
        elif video_path is None:
            row["visual_status"] = "skipped_missing_video"
        elif not _openface_is_configured(args.openface_bin):
            row["visual_status"] = "skipped_openface_unconfigured"
        elif not visual_ready:
            row["visual_status"] = "skipped_no_visual_eligible_segments"
        else:
            visual_cmd = [
                sys.executable,
                "scripts/run_openface_features.py",
                "--source-id",
                source_id,
                "--video-path",
                str(video_path),
                "--segment-manifest",
                str(segment_manifest_path),
                "--source-manifest",
                str(source_manifest_path),
            ]
            if args.openface_bin:
                visual_cmd.extend(["--openface-bin", str(args.openface_bin)])

            visual_proc = _run_sidecar_command(repo_dir, visual_cmd)
            if visual_proc.returncode == 0:
                row["visual_ran"] = "true"
                row["visual_status"] = "ran"
            else:
                row["visual_status"] = f"error_exit_{visual_proc.returncode}"

        nlp_manifest_ready = _has_nlp_ready_manifest_segments(segment_rows)
        if chunks_path is not None:
            nlp_cmd = [
                sys.executable,
                "scripts/run_nlp_segment_scoring.py",
                "--source-id",
                source_id,
            ]
            if chunks_path.suffix.lower() == ".csv":
                nlp_cmd.extend(["--chunks-csv", str(chunks_path)])
            else:
                nlp_cmd.extend(["--chunks-jsonl", str(chunks_path)])
            if args.run_secondary_emotion:
                nlp_cmd.append("--run-secondary-emotion")

            nlp_proc = _run_sidecar_command(repo_dir, nlp_cmd)
            if nlp_proc.returncode == 0:
                row["nlp_ran"] = "true"
                row["nlp_status"] = "ran"
            else:
                row["nlp_status"] = f"error_exit_{nlp_proc.returncode}"
        elif transcript_path is not None and transcript_path.suffix.lower() == ".json" and nlp_manifest_ready:
            nlp_cmd = [
                sys.executable,
                "scripts/run_nlp_segment_scoring.py",
                "--source-id",
                source_id,
                "--segment-manifest",
                str(segment_manifest_path),
                "--transcript-path",
                str(transcript_path),
            ]
            if args.run_secondary_emotion:
                nlp_cmd.append("--run-secondary-emotion")

            nlp_proc = _run_sidecar_command(repo_dir, nlp_cmd)
            if nlp_proc.returncode == 0:
                row["nlp_ran"] = "true"
                row["nlp_status"] = "ran"
            else:
                row["nlp_status"] = f"error_exit_{nlp_proc.returncode}"
        elif transcript_path is None and chunks_path is None:
            row["nlp_status"] = "skipped_missing_transcript_or_chunks"
        else:
            row["nlp_status"] = "skipped_segment_manifest_not_ready"

        row["skip_reason"] = _join_reasons(
            row["alignment_status"],
            row["visual_status"],
            row["nlp_status"],
        )
        status_rows.append(row)

    summary_cmd = [
        sys.executable,
        "scripts/build_multimodal_eval_summary.py",
        "--source-manifest",
        str(source_manifest_path),
        "--segment-manifest",
        str(segment_manifest_path),
    ]
    summary_proc = _run_sidecar_command(repo_dir, summary_cmd)
    summary_status = "ran" if summary_proc.returncode == 0 else f"error_exit_{summary_proc.returncode}"

    _write_status_csv(status_csv_path, status_rows)
    _write_status_json(
        status_json_path,
        {
            "status": "ok" if summary_proc.returncode == 0 else "error",
            "inputs_root": str(inputs_root),
            "source_manifest": str(source_manifest_path),
            "segment_manifest": str(segment_manifest_path),
            "source_ids": source_ids,
            "summary_status": summary_status,
            "status_csv": str(status_csv_path),
            "status_json": str(status_json_path),
            "status_rows": status_rows,
            "notes": [
                "Missing local files skip safely per modality.",
                "Transcript remains primary; sidecar outputs are supporting evidence only.",
            ],
        },
    )

    print(
        json.dumps(
            {
                "status": "ok" if summary_proc.returncode == 0 else "error",
                "inputs_root": str(inputs_root),
                "source_ids": source_ids,
                "status_csv": str(status_csv_path),
                "status_json": str(status_json_path),
                "summary_status": summary_status,
                "sources_with_any_sidecar_run": sum(
                    row["alignment_ran"] == "true"
                    or row["visual_ran"] == "true"
                    or row["nlp_ran"] == "true"
                    for row in status_rows
                ),
            },
            indent=2,
        )
    )
    return 0 if summary_proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
