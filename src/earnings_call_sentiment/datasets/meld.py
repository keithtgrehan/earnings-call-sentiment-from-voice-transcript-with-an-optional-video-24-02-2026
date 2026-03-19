"""MELD placeholder adapter.

MELD is kept secondary in this repo: useful for generic conversation-style
calibration or sanity checks, not as finance-domain evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .registry import DatasetRegistryEntry, NormalizedDatasetRecord

DATASET_ID = "meld"
ENV_VAR = "EARNINGS_CALL_MELD_ROOT"
DEFAULT_RELATIVE_ROOT = "data/external_datasets/meld"

EXPECTED_LAYOUT = (
    "Place the extracted MELD root under the configured path or data/external_datasets/meld/.",
    "Expected metadata files: train_sent_emo.csv, dev_sent_emo.csv, and/or test_sent_emo.csv.",
    "Optional video folders may be present, such as train_splits/, dev_splits_complete/, or output_repeated_splits_test/.",
)

_SPLIT_FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

_VIDEO_DIRS = (
    "train_splits",
    "dev_splits_complete",
    "output_repeated_splits_test",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def validate_local_structure(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {
            "present": False,
            "structure_ok": False,
            "notes": [
                "Dataset root is not present locally.",
                "This is acceptable; MELD remains an optional secondary calibration dataset only.",
            ],
            "metadata_files_found": [],
            "video_dirs_found": [],
        }

    metadata_files_found = [
        filename
        for filename in _SPLIT_FILES.values()
        if (root / filename).exists()
    ]
    video_dirs_found = [name for name in _VIDEO_DIRS if (root / name).exists()]
    notes = []
    if not metadata_files_found:
        notes.append("No MELD split metadata CSVs were found.")

    return {
        "present": True,
        "structure_ok": bool(metadata_files_found),
        "notes": notes or ["MELD metadata layout looks usable for placeholder loading."],
        "metadata_files_found": metadata_files_found,
        "video_dirs_found": video_dirs_found,
    }


def _guess_video_path(root: Path, split: str, dialogue_id: str, utterance_id: str) -> str:
    filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    for directory in _VIDEO_DIRS:
        candidate = root / directory / filename
        if candidate.exists():
            return str(candidate)
    return ""


def _label_namespace(row: pd.Series) -> str:
    if _clean_text(row.get("Emotion")):
        return "emotion"
    if _clean_text(row.get("Sentiment")):
        return "sentiment"
    return ""


def _original_label(row: pd.Series) -> str:
    if _clean_text(row.get("Emotion")):
        return _clean_text(row.get("Emotion"))
    return _clean_text(row.get("Sentiment"))


def load_normalized_records(root: Path, limit: int | None = None) -> list[NormalizedDatasetRecord]:
    records: list[NormalizedDatasetRecord] = []
    for split, filename in _SPLIT_FILES.items():
        path = root / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        for _, row in frame.iterrows():
            dialogue_id = _clean_text(row.get("Dialogue_ID"))
            utterance_id = _clean_text(row.get("Utterance_ID"))
            speaker = _clean_text(row.get("Speaker"))
            records.append(
                NormalizedDatasetRecord(
                    dataset_id=DATASET_ID,
                    record_id=f"meld_{split}_d{dialogue_id or 'x'}_u{utterance_id or 'x'}",
                    split=split,
                    modality="video_text" if _guess_video_path(root, split, dialogue_id, utterance_id) else "text",
                    text=_clean_text(row.get("Utterance") or row.get("Dialogue")),
                    audio_path="",
                    video_path=_guess_video_path(root, split, dialogue_id, utterance_id),
                    transcript_path="",
                    speaker_id=speaker,
                    speaker_role=speaker,
                    label_namespace=_label_namespace(row),
                    original_label=_original_label(row),
                    normalized_label=_original_label(row).lower(),
                    source_context="dialogue_emotion",
                    is_finance_domain=False,
                    notes="MELD is secondary only here: calibration or sanity checks, not earnings-call truth.",
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


REGISTRY_ENTRY = DatasetRegistryEntry(
    dataset_id=DATASET_ID,
    display_name="MELD",
    priority_level="secondary_calibration_only",
    env_var=ENV_VAR,
    default_relative_root=DEFAULT_RELATIVE_ROOT,
    purpose="Secondary conversational calibration or sanity-check dataset only.",
    should_not_use_for=(
        "finance-domain benchmarking",
        "replacing earnings-call evidence",
        "product truth claims about management behavior",
    ),
    expected_layout=EXPECTED_LAYOUT,
    validate_fn=validate_local_structure,
    load_fn=load_normalized_records,
)
