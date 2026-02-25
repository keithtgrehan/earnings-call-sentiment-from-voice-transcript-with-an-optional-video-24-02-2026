from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from earnings_call_sentiment.transcriber import get_audio_duration_s


def test_get_audio_duration_s_for_generated_wav(tmp_path: Path) -> None:
    sample_rate = 16000
    duration_s = 1.0
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)

    wav_path = tmp_path / "tone.wav"
    sf.write(wav_path, signal, sample_rate)

    measured = get_audio_duration_s(wav_path)
    assert measured > 0.95
    assert measured < 1.05
