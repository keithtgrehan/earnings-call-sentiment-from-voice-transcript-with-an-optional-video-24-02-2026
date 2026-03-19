from __future__ import annotations

from earnings_call_sentiment.optional_runtime import (
    DEFAULT_EMOTION_MODEL,
    DEFAULT_FINBERT_MODEL,
    DEFAULT_MULTIMODAL_DEVICE,
    DEFAULT_PYANNOTE_MODEL,
    load_multimodal_config,
    multimodal_config_status,
)


def test_multimodal_config_defaults(monkeypatch) -> None:
    for name in [
        "EARNINGS_CALL_WHISPERX_ENABLED",
        "EARNINGS_CALL_WHISPERX_DIARIZATION_ENABLED",
        "EARNINGS_CALL_PYANNOTE_ENABLED",
        "EARNINGS_CALL_PYANNOTE_MODEL",
        "EARNINGS_CALL_MULTIMODAL_DEVICE",
        "EARNINGS_CALL_HF_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "EARNINGS_CALL_MODEL_CACHE_DIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "EARNINGS_CALL_OPENFACE_ENABLED",
        "EARNINGS_CALL_OPENFACE_BIN",
        "EARNINGS_CALL_OPENFACE_ROOT",
        "EARNINGS_CALL_OPENFACE_WORK_DIR",
        "EARNINGS_CALL_FINBERT_MODEL",
        "EARNINGS_CALL_EMOTION_MODEL",
        "EARNINGS_CALL_MAEC_ROOT",
        "EARNINGS_CALL_MELD_ROOT",
        "EARNINGS_CALL_RAVDESS_ROOT",
    ]:
        monkeypatch.delenv(name, raising=False)

    config = load_multimodal_config()

    assert config.whisperx_enabled is False
    assert config.whisperx_diarization_enabled is False
    assert config.pyannote_enabled is False
    assert config.pyannote_model == DEFAULT_PYANNOTE_MODEL
    assert config.multimodal_device == DEFAULT_MULTIMODAL_DEVICE
    assert config.hf_token_env is None
    assert config.openface_enabled is False
    assert config.openface_bin is None
    assert config.finbert_model == DEFAULT_FINBERT_MODEL
    assert config.emotion_model == DEFAULT_EMOTION_MODEL
    assert config.datasets.maec_root is None
    assert config.datasets.meld_root is None
    assert config.datasets.ravdess_root is None


def test_multimodal_config_reads_project_env(monkeypatch) -> None:
    monkeypatch.setenv("EARNINGS_CALL_WHISPERX_ENABLED", "1")
    monkeypatch.setenv("EARNINGS_CALL_WHISPERX_DIARIZATION_ENABLED", "true")
    monkeypatch.setenv("EARNINGS_CALL_PYANNOTE_ENABLED", "yes")
    monkeypatch.setenv("EARNINGS_CALL_PYANNOTE_MODEL", "custom/pyannote-model")
    monkeypatch.setenv("EARNINGS_CALL_MULTIMODAL_DEVICE", "cuda")
    monkeypatch.setenv("EARNINGS_CALL_HF_TOKEN", "secret-token")
    monkeypatch.setenv("EARNINGS_CALL_OPENFACE_ENABLED", "1")
    monkeypatch.setenv("EARNINGS_CALL_OPENFACE_BIN", "/tmp/openface-bin")
    monkeypatch.setenv("EARNINGS_CALL_FINBERT_MODEL", "custom/finbert")
    monkeypatch.setenv("EARNINGS_CALL_EMOTION_MODEL", "custom/emotion")
    monkeypatch.setenv("EARNINGS_CALL_MAEC_ROOT", "./data/maec")
    monkeypatch.setenv("EARNINGS_CALL_MELD_ROOT", "./data/meld")
    monkeypatch.setenv("EARNINGS_CALL_RAVDESS_ROOT", "./data/ravdess")

    config = load_multimodal_config()

    assert config.whisperx_enabled is True
    assert config.whisperx_diarization_enabled is True
    assert config.pyannote_enabled is True
    assert config.pyannote_model == "custom/pyannote-model"
    assert config.multimodal_device == "cuda"
    assert config.hf_token_env == "EARNINGS_CALL_HF_TOKEN"
    assert config.openface_enabled is True
    assert config.openface_bin == "/tmp/openface-bin"
    assert config.finbert_model == "custom/finbert"
    assert config.emotion_model == "custom/emotion"
    assert str(config.datasets.maec_root) == "data/maec"
    assert str(config.datasets.meld_root) == "data/meld"
    assert str(config.datasets.ravdess_root) == "data/ravdess"


def test_openface_root_fallback_builds_feature_extraction_path(monkeypatch) -> None:
    monkeypatch.delenv("EARNINGS_CALL_OPENFACE_BIN", raising=False)
    monkeypatch.setenv("EARNINGS_CALL_OPENFACE_ENABLED", "1")
    monkeypatch.setenv("EARNINGS_CALL_OPENFACE_ROOT", "/opt/openface/bin")

    config = load_multimodal_config()

    assert config.openface_bin == "/opt/openface/bin/FeatureExtraction"


def test_hf_token_fallback_prefers_standard_env_when_project_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("EARNINGS_CALL_HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "fallback-token")

    config = load_multimodal_config()

    assert config.hf_token_env == "HF_TOKEN"


def test_status_never_exposes_token_value(monkeypatch) -> None:
    monkeypatch.setenv("EARNINGS_CALL_HF_TOKEN", "secret-token")

    status = multimodal_config_status()

    assert status["hf_token_present"] is True
    assert status["hf_token_env"] == "EARNINGS_CALL_HF_TOKEN"
    assert "secret-token" not in str(status)
