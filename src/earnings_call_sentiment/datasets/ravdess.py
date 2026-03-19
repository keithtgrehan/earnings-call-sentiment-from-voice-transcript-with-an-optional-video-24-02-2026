"""RAVDESS placeholder adapter.

RAVDESS is treated as a secondary calibration/sanity-check resource only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import DatasetRegistryEntry, NormalizedDatasetRecord

DATASET_ID = "ravdess"
ENV_VAR = "EARNINGS_CALL_RAVDESS_ROOT"
DEFAULT_RELATIVE_ROOT = "data/external_datasets/ravdess"

EXPECTED_LAYOUT = (
    "Place the extracted dataset root under the configured path or data/external_datasets/ravdess/.",
    "Expected media layout: Actor_01/, Actor_02/, ... with .wav and/or .mp4 files inside.",
    "The adapter parses standard RAVDESS file names such as 03-01-05-01-02-01-12.wav.",
)

_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

_MODALITY_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _media_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("Actor_*/*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp4"}
    )


def validate_local_structure(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {
            "present": False,
            "structure_ok": False,
            "notes": [
                "Dataset root is not present locally.",
                "This is acceptable; RAVDESS remains a secondary sanity-check dataset only.",
            ],
            "actor_dirs_found": [],
            "media_file_count": 0,
        }

    actor_dirs = sorted(path.name for path in root.glob("Actor_*") if path.is_dir())
    media_files = _media_files(root)
    notes = []
    if not actor_dirs:
        notes.append("No Actor_* folders were detected.")
    if not media_files:
        notes.append("No .wav or .mp4 media files were detected under Actor_* folders.")

    return {
        "present": True,
        "structure_ok": bool(actor_dirs and media_files),
        "notes": notes or ["RAVDESS media layout looks usable for placeholder loading."],
        "actor_dirs_found": actor_dirs[:10],
        "media_file_count": int(len(media_files)),
    }


def _parse_filename(path: Path) -> dict[str, str]:
    stem_parts = path.stem.split("-")
    if len(stem_parts) != 7:
        return {}
    modality_code, _, emotion_code, _, _, _, actor_code = stem_parts
    return {
        "modality_code": modality_code,
        "emotion_code": emotion_code,
        "actor_code": actor_code,
    }


def load_normalized_records(root: Path, limit: int | None = None) -> list[NormalizedDatasetRecord]:
    records: list[NormalizedDatasetRecord] = []
    for path in _media_files(root):
        parsed = _parse_filename(path)
        records.append(
            NormalizedDatasetRecord(
                dataset_id=DATASET_ID,
                record_id=path.stem,
                split="",
                modality=_MODALITY_MAP.get(parsed.get("modality_code", ""), "unknown"),
                text="",
                audio_path=str(path) if path.suffix.lower() == ".wav" else "",
                video_path=str(path) if path.suffix.lower() == ".mp4" else "",
                transcript_path="",
                speaker_id=parsed.get("actor_code", ""),
                speaker_role="actor",
                label_namespace="emotion",
                original_label=_EMOTION_MAP.get(parsed.get("emotion_code", ""), ""),
                normalized_label=_EMOTION_MAP.get(parsed.get("emotion_code", ""), ""),
                source_context="acted_expression",
                is_finance_domain=False,
                notes="RAVDESS is secondary only here: feature sanity checks, not earnings-call evidence.",
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


REGISTRY_ENTRY = DatasetRegistryEntry(
    dataset_id=DATASET_ID,
    display_name="RAVDESS",
    priority_level="secondary_calibration_only",
    env_var=ENV_VAR,
    default_relative_root=DEFAULT_RELATIVE_ROOT,
    purpose="Secondary audio/video sanity-check dataset only.",
    should_not_use_for=(
        "finance-domain benchmarking",
        "earnings-call label truth",
        "product claims about management sentiment or behavior",
    ),
    expected_layout=EXPECTED_LAYOUT,
    validate_fn=validate_local_structure,
    load_fn=load_normalized_records,
)
