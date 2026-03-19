"""Registry and shared record shape for external multimodal datasets.

These adapters are intentionally conservative:
- no downloads
- no training
- no benchmark rewrites
- no assumption that datasets exist locally
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from earnings_call_sentiment.optional_runtime import load_multimodal_config


@dataclass(frozen=True)
class NormalizedDatasetRecord:
    dataset_id: str
    record_id: str
    split: str
    modality: str
    text: str
    audio_path: str
    video_path: str
    transcript_path: str
    speaker_id: str
    speaker_role: str
    label_namespace: str
    original_label: str
    normalized_label: str
    source_context: str
    is_finance_domain: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    display_name: str
    priority_level: str
    env_var: str
    default_relative_root: str
    purpose: str
    should_not_use_for: tuple[str, ...]
    expected_layout: tuple[str, ...]
    validate_fn: Callable[[Path], dict[str, Any]]
    load_fn: Callable[[Path, int | None], list[NormalizedDatasetRecord]]

    def default_root(self) -> Path:
        return repo_root() / self.default_relative_root


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dataset_roots_from_config() -> dict[str, Path | None]:
    config = load_multimodal_config()
    return {
        "maec": config.datasets.maec_root,
        "meld": config.datasets.meld_root,
        "ravdess": config.datasets.ravdess_root,
    }


def configured_dataset_root(dataset_id: str) -> Path:
    entry = get_dataset_entry(dataset_id)
    configured = _dataset_roots_from_config().get(dataset_id)
    if configured is not None:
        return configured.expanduser().resolve()
    return entry.default_root().resolve()


def _load_registry() -> dict[str, DatasetRegistryEntry]:
    from .maec import REGISTRY_ENTRY as MAEC_ENTRY
    from .meld import REGISTRY_ENTRY as MELD_ENTRY
    from .ravdess import REGISTRY_ENTRY as RAVDESS_ENTRY

    return {
        MAEC_ENTRY.dataset_id: MAEC_ENTRY,
        MELD_ENTRY.dataset_id: MELD_ENTRY,
        RAVDESS_ENTRY.dataset_id: RAVDESS_ENTRY,
    }


def list_dataset_entries() -> list[DatasetRegistryEntry]:
    return [entry for _, entry in sorted(_load_registry().items())]


def get_dataset_entry(dataset_id: str) -> DatasetRegistryEntry:
    normalized = str(dataset_id).strip().lower()
    registry = _load_registry()
    if normalized not in registry:
        raise KeyError(f"Unknown external dataset '{dataset_id}'.")
    return registry[normalized]


def validate_dataset_presence(
    dataset_id: str,
    *,
    root_override: str | Path | None = None,
) -> dict[str, Any]:
    entry = get_dataset_entry(dataset_id)
    root = Path(root_override).expanduser().resolve() if root_override is not None else configured_dataset_root(dataset_id)
    report = entry.validate_fn(root)
    report.update(
        {
            "dataset_id": entry.dataset_id,
            "display_name": entry.display_name,
            "priority_level": entry.priority_level,
            "env_var": entry.env_var,
            "configured_root": str(root),
            "default_root": str(entry.default_root().resolve()),
            "purpose": entry.purpose,
            "should_not_use_for": list(entry.should_not_use_for),
            "expected_layout": list(entry.expected_layout),
        }
    )
    return report


def normalize_dataset_records(
    dataset_id: str,
    *,
    root_override: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    entry = get_dataset_entry(dataset_id)
    root = Path(root_override).expanduser().resolve() if root_override is not None else configured_dataset_root(dataset_id)
    return [record.to_dict() for record in entry.load_fn(root, limit)]
