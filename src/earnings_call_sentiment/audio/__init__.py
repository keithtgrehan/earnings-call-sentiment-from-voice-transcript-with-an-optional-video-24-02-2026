"""Deterministic audio behavior helpers for earnings-call runs."""

from .pause_features import AudioEnvelope, AudioMetadata, load_audio_envelope
from .segment_aggregate import aggregate_audio_segments
from .summary import compute_audio_behavior_outputs, write_audio_behavior_outputs

__all__ = [
    "AudioEnvelope",
    "AudioMetadata",
    "load_audio_envelope",
    "aggregate_audio_segments",
    "compute_audio_behavior_outputs",
    "write_audio_behavior_outputs",
]
