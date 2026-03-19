"""MAEC staging adapter.

This adapter accepts a conservative local staging layout rather than assuming
the dataset has already been unpacked into one exact upstream file tree.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .registry import DatasetRegistryEntry, NormalizedDatasetRecord

DATASET_ID = "maec"
ENV_VAR = "EARNINGS_CALL_MAEC_ROOT"
DEFAULT_RELATIVE_ROOT = "data/external_datasets/maec"

EXPECTED_LAYOUT = (
    "Place the staged dataset under the configured root or data/external_datasets/maec/.",
    "Provide one metadata file such as metadata/records.csv, metadata/records.jsonl, records.csv, or records.jsonl.",
    "Optional supporting folders: transcripts/, audio/, video/.",
)

_CANDIDATE_METADATA_FILES = (
    "metadata/records.csv",
    "metadata/records.jsonl",
    "metadata/maec.csv",
    "metadata/maec.jsonl",
    "records.csv",
    "records.jsonl",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _find_metadata_file(root: Path) -> Path | None:
    for relative in _CANDIDATE_METADATA_FILES:
        candidate = root / relative
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def validate_local_structure(root: Path) -> dict[str, Any]:
    metadata_path = _find_metadata_file(root)
    transcript_dir = root / "transcripts"
    audio_dir = root / "audio"
    video_dir = root / "video"

    if not root.exists():
        return {
            "present": False,
            "structure_ok": False,
            "notes": [
                "Dataset root is not present locally.",
                "This is acceptable; the adapter is only a placeholder until MAEC is staged locally.",
            ],
            "metadata_file": "",
            "detected_support_dirs": [],
        }

    detected_support_dirs = [
        name
        for name, path in (
            ("transcripts", transcript_dir),
            ("audio", audio_dir),
            ("video", video_dir),
        )
        if path.exists() and path.is_dir()
    ]
    notes = []
    if metadata_path is None:
        notes.append("No MAEC metadata manifest was found in the expected staging locations.")
    if not detected_support_dirs:
        notes.append("No transcripts/, audio/, or video/ staging folders were detected.")

    return {
        "present": True,
        "structure_ok": metadata_path is not None,
        "notes": notes or ["MAEC staging layout looks usable for placeholder metadata loading."],
        "metadata_file": str(metadata_path) if metadata_path is not None else "",
        "detected_support_dirs": detected_support_dirs,
    }


def _rows_from_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _first_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and _clean_text(row.get(key)):
            return _clean_text(row.get(key))
    return ""


def load_normalized_records(root: Path, limit: int | None = None) -> list[NormalizedDatasetRecord]:
    metadata_path = _find_metadata_file(root)
    if metadata_path is None:
        return []

    rows = _rows_from_metadata(metadata_path)
    records: list[NormalizedDatasetRecord] = []
    for index, row in enumerate(rows, start=1):
        record_id = _first_value(row, "record_id", "source_id", "id") or f"maec_{index:05d}"
        records.append(
            NormalizedDatasetRecord(
                dataset_id=DATASET_ID,
                record_id=record_id,
                split=_first_value(row, "split", "partition"),
                modality=_first_value(row, "modality") or "multimodal",
                text=_first_value(row, "text", "transcript", "utterance"),
                audio_path=_first_value(row, "audio_path", "audio_file"),
                video_path=_first_value(row, "video_path", "video_file"),
                transcript_path=_first_value(row, "transcript_path", "transcript_file"),
                speaker_id=_first_value(row, "speaker_id", "speaker"),
                speaker_role=_first_value(row, "speaker_role", "role"),
                label_namespace=_first_value(row, "label_namespace", "target_namespace"),
                original_label=_first_value(row, "label", "target", "sentiment"),
                normalized_label=_first_value(row, "normalized_label", "label", "target", "sentiment"),
                source_context=_first_value(row, "source_context", "event_type") or "earnings_call",
                is_finance_domain=True,
                notes="MAEC is the only external dataset in this scaffold treated as domain-relevant.",
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


REGISTRY_ENTRY = DatasetRegistryEntry(
    dataset_id=DATASET_ID,
    display_name="MAEC",
    priority_level="primary_domain_reference",
    env_var=ENV_VAR,
    default_relative_root=DEFAULT_RELATIVE_ROOT,
    purpose="Domain-relevant external reference for future earnings-call calibration or analysis.",
    should_not_use_for=(
        "replacing deterministic repo outputs",
        "claiming broader benchmark coverage than the repo currently has",
        "auto-training or auto-downloading in this scaffold",
    ),
    expected_layout=EXPECTED_LAYOUT,
    validate_fn=validate_local_structure,
    load_fn=load_normalized_records,
)
