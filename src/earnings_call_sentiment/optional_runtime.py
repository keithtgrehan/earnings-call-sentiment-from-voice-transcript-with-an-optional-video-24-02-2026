"""Optional multimodal dependency and environment scaffolding.

This module is intentionally lightweight:
- no heavy optional imports at module import time
- no automatic downloads or credential handling
- no behavior changes to the current default pipeline
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

HF_TOKEN_ENV_CANDIDATES = (
    "EARNINGS_CALL_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MULTIMODAL_DEVICE = "cpu"
DEFAULT_OPENFACE_BINARY_NAME = "FeatureExtraction"


def _clean_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _env_flag(name: str, default: bool = False) -> bool:
    value = _clean_env(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return default


def _env_path(name: str) -> Path | None:
    value = _clean_env(name)
    if value is None:
        return None
    return Path(value).expanduser()


def _resolve_hf_token_env() -> str | None:
    for env_name in HF_TOKEN_ENV_CANDIDATES:
        if _clean_env(env_name) is not None:
            return env_name
    return None


def _resolve_openface_bin() -> str | None:
    direct = _clean_env("EARNINGS_CALL_OPENFACE_BIN")
    if direct is not None:
        return direct

    root = _env_path("EARNINGS_CALL_OPENFACE_ROOT")
    if root is not None:
        return str(root / DEFAULT_OPENFACE_BINARY_NAME)

    if _env_flag("EARNINGS_CALL_OPENFACE_ENABLED", default=False):
        return DEFAULT_OPENFACE_BINARY_NAME
    return None


def optional_dependency_available(module_name: str) -> bool:
    """Return True when a Python module can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def load_optional_dependency(module_name: str, *, package_name: str | None = None) -> Any:
    """Import an optional dependency with a clear error message."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised via callers
        install_hint = package_name or module_name
        raise RuntimeError(
            f"Optional dependency '{module_name}' is not available. "
            f"Install '{install_hint}' to enable this sidecar."
        ) from exc


def load_whisperx() -> Any:
    return load_optional_dependency("whisperx")


def load_pyannote_audio() -> Any:
    return load_optional_dependency("pyannote.audio", package_name="pyannote-audio")


@dataclass(frozen=True)
class DatasetRootConfig:
    maec_root: Path | None
    meld_root: Path | None
    ravdess_root: Path | None


@dataclass(frozen=True)
class MultimodalConfig:
    whisperx_enabled: bool
    whisperx_diarization_enabled: bool
    pyannote_enabled: bool
    pyannote_model: str
    multimodal_device: str
    hf_token_env: str | None
    model_cache_dir: Path | None
    hf_home: Path | None
    transformers_cache: Path | None
    openface_enabled: bool
    openface_bin: str | None
    openface_root: Path | None
    openface_work_dir: Path | None
    finbert_model: str
    emotion_model: str
    datasets: DatasetRootConfig


def load_multimodal_config() -> MultimodalConfig:
    return MultimodalConfig(
        whisperx_enabled=_env_flag("EARNINGS_CALL_WHISPERX_ENABLED", default=False),
        whisperx_diarization_enabled=_env_flag(
            "EARNINGS_CALL_WHISPERX_DIARIZATION_ENABLED",
            default=False,
        ),
        pyannote_enabled=_env_flag("EARNINGS_CALL_PYANNOTE_ENABLED", default=False),
        pyannote_model=_clean_env("EARNINGS_CALL_PYANNOTE_MODEL")
        or DEFAULT_PYANNOTE_MODEL,
        multimodal_device=_clean_env("EARNINGS_CALL_MULTIMODAL_DEVICE")
        or DEFAULT_MULTIMODAL_DEVICE,
        hf_token_env=_resolve_hf_token_env(),
        model_cache_dir=_env_path("EARNINGS_CALL_MODEL_CACHE_DIR"),
        hf_home=_env_path("HF_HOME"),
        transformers_cache=_env_path("TRANSFORMERS_CACHE"),
        openface_enabled=_env_flag("EARNINGS_CALL_OPENFACE_ENABLED", default=False),
        openface_bin=_resolve_openface_bin(),
        openface_root=_env_path("EARNINGS_CALL_OPENFACE_ROOT"),
        openface_work_dir=_env_path("EARNINGS_CALL_OPENFACE_WORK_DIR"),
        finbert_model=_clean_env("EARNINGS_CALL_FINBERT_MODEL")
        or DEFAULT_FINBERT_MODEL,
        emotion_model=_clean_env("EARNINGS_CALL_EMOTION_MODEL")
        or DEFAULT_EMOTION_MODEL,
        datasets=DatasetRootConfig(
            maec_root=_env_path("EARNINGS_CALL_MAEC_ROOT"),
            meld_root=_env_path("EARNINGS_CALL_MELD_ROOT"),
            ravdess_root=_env_path("EARNINGS_CALL_RAVDESS_ROOT"),
        ),
    )


def multimodal_config_status() -> dict[str, Any]:
    """Return a serializable config/status view without exposing token values."""
    config = load_multimodal_config()
    payload = asdict(config)
    payload["hf_token_present"] = config.hf_token_env is not None
    payload["whisperx_installed"] = optional_dependency_available("whisperx")
    payload["pyannote_audio_installed"] = optional_dependency_available("pyannote.audio")
    payload["openface_configured"] = bool(config.openface_bin)
    return payload
