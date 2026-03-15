from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .pause_features import AudioEnvelope

_FEATURE_MAP = {
    "pitch_variability": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_mean": "loudness_sma3_amean",
    "loudness_variability": "loudness_sma3_stddevNorm",
    "spectral_flux": "spectralFlux_sma3_amean",
}


@lru_cache(maxsize=1)
def _smile() -> Any:
    try:
        import opensmile  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "openSMILE is required for eGeMAPSv02 extraction. Install opensmile to enable it."
        ) from exc
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps_slice(
    envelope: AudioEnvelope,
    start_time_s: float,
    end_time_s: float,
) -> dict[str, float | None]:
    if end_time_s <= start_time_s:
        return {key: None for key in _FEATURE_MAP}

    start_idx = max(0, int(round(float(start_time_s) * envelope.metadata.sample_rate)))
    end_idx = min(len(envelope.waveform), int(round(float(end_time_s) * envelope.metadata.sample_rate)))
    if end_idx - start_idx < int(envelope.metadata.sample_rate * 0.5):
        return {key: None for key in _FEATURE_MAP}

    signal = np.asarray(envelope.waveform[start_idx:end_idx], dtype=np.float32)
    if signal.size == 0:
        return {key: None for key in _FEATURE_MAP}

    try:
        features_df = _smile().process_signal(signal, envelope.metadata.sample_rate)
    except Exception:
        return {key: None for key in _FEATURE_MAP}

    if features_df.empty:
        return {key: None for key in _FEATURE_MAP}

    row = features_df.iloc[0]
    payload: dict[str, float | None] = {}
    for key, column in _FEATURE_MAP.items():
        value = row.get(column)
        try:
            payload[key] = float(value)
        except (TypeError, ValueError):
            payload[key] = None
    return payload


def opensmile_available() -> bool:
    try:
        _smile()
    except Exception:
        return False
    return True
