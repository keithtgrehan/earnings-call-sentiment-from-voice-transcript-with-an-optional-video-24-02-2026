from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from earnings_call_sentiment import cli
from earnings_call_sentiment.audio.summary import compute_audio_behavior_outputs


def _tone(sr: int, duration_s: float, *, freq_hz: float = 220.0) -> np.ndarray:
    times = np.arange(int(sr * duration_s), dtype=np.float32) / float(sr)
    return 0.2 * np.sin(2.0 * np.pi * float(freq_hz) * times)


def test_audio_behavior_summary_handles_missing_audio() -> None:
    payload = compute_audio_behavior_outputs(None, pd.DataFrame())
    assert payload["segments_df"].empty
    assert payload["summary"]["audio_available"] is False
    assert payload["summary"]["audio_features_available"] is False


def test_audio_behavior_outputs_capture_pause_and_fillers(tmp_path: Path) -> None:
    sr = 16000
    waveform = np.concatenate(
        [
            _tone(sr, 0.45),
            np.zeros(int(sr * 0.25), dtype=np.float32),
            _tone(sr, 0.35),
            np.zeros(int(sr * 0.35), dtype=np.float32),
            _tone(sr, 0.70),
        ]
    )
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, waveform, sr)

    qa_segments_df = pd.DataFrame(
        [
            {
                "segment_id": 0,
                "start": 0.0,
                "end": 0.45,
                "phase": "prepared_remarks",
                "speaker_role": "management",
                "qa_pair_id": 0,
                "text": "We remain confident in the setup.",
            },
            {
                "segment_id": 1,
                "start": 0.70,
                "end": 1.05,
                "phase": "q_and_a",
                "speaker_role": "analyst",
                "qa_pair_id": 1,
                "text": "What changed in the demand picture?",
            },
            {
                "segment_id": 2,
                "start": 1.40,
                "end": 2.10,
                "phase": "q_and_a",
                "speaker_role": "management",
                "qa_pair_id": 1,
                "text": "Um, we are kind of staying on track here.",
            },
        ]
    )

    payload = compute_audio_behavior_outputs(audio_path, qa_segments_df)
    segments_df = payload["segments_df"]
    answer_row = segments_df[segments_df["segment_id"] == 2].iloc[0]
    assert float(answer_row["pause_before_answer_ms"]) >= 250.0
    assert float(answer_row["answer_onset_delay_ms"]) >= 300.0
    assert int(answer_row["filler_count"]) >= 2
    assert "um" in str(answer_row["matched_fillers"])
    assert str(answer_row["hesitation_label"]) in {"medium", "high"}
    assert payload["summary"]["audio_available"] is True
    assert payload["summary"]["audio_features_available"] is True
    assert payload["summary"]["pauses_before_answers"]["level"] in {"medium", "high"}


def test_pause_before_answer_only_applies_to_first_answer_chunk(tmp_path: Path) -> None:
    sr = 16000
    waveform = np.concatenate(
        [
            _tone(sr, 0.35),
            np.zeros(int(sr * 0.30), dtype=np.float32),
            _tone(sr, 0.90),
        ]
    )
    audio_path = tmp_path / "chunked_answer.wav"
    sf.write(audio_path, waveform, sr)

    qa_segments_df = pd.DataFrame(
        [
            {
                "segment_id": 0,
                "start": 0.0,
                "end": 0.35,
                "phase": "q_and_a",
                "speaker_role": "analyst",
                "qa_pair_id": 3,
                "text": "Can you help us understand the timing?",
            },
            {
                "segment_id": 1,
                "start": 0.65,
                "end": 1.05,
                "phase": "q_and_a",
                "speaker_role": "management",
                "qa_pair_id": 3,
                "text": "We remain on track here.",
            },
            {
                "segment_id": 2,
                "start": 1.05,
                "end": 1.55,
                "phase": "q_and_a",
                "speaker_role": "management",
                "qa_pair_id": 3,
                "text": "And the cadence is largely unchanged.",
            },
        ]
    )

    payload = compute_audio_behavior_outputs(audio_path, qa_segments_df)
    segments_df = payload["segments_df"].set_index("segment_id")
    assert float(segments_df.loc[1, "pause_before_answer_ms"]) >= 250.0
    assert pd.isna(segments_df.loc[2, "pause_before_answer_ms"])
    assert payload["summary"]["pauses_before_answers"]["median_ms"] >= 250.0


def test_report_markdown_includes_audio_section_when_summary_present(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"
    cli._write_report_markdown(
        output_path=output_path,
        metrics_payload={"num_chunks_scored": 2, "sentiment_mean": 0.1, "sentiment_std": 0.2, "guidance": {"row_count": 1, "mean_strength": 0.4}},
        guidance_df=pd.DataFrame(),
        guidance_revision_df=pd.DataFrame(),
        behavioral_summary={
            "uncertainty_score_overall": {"level": "low"},
            "reassurance_score_management": {"level": "medium"},
            "analyst_skepticism_score": {"level": "high"},
            "strongest_evidence": {},
        },
        qa_shift_summary={
            "prepared_remarks_vs_q_and_a": {"label": "mixed"},
            "analyst_skepticism": {"level": "high"},
            "management_answers_vs_prepared_uncertainty": {"label": "mixed"},
            "early_vs_late_q_and_a": {"label": "low"},
            "strongest_evidence": {},
        },
        audio_summary={
            "audio_features_available": True,
            "hesitation_overall": {"level": "medium"},
            "pauses_before_answers": {"level": "high"},
            "prepared_baseline_audio_stability": {"level": "medium"},
            "qa_hesitation_shift": {"level": "medium"},
            "most_hesitant_answers": [
                {
                    "section": "q_and_a",
                    "start_time_s": 10.0,
                    "end_time_s": 18.0,
                    "hesitation_label": "high",
                    "pause_before_answer_ms": 820.0,
                }
            ],
            "low_confidence_segments": [
                {"confidence_note": "short segment limits audio confidence"}
            ],
        },
        visual_summary=None,
    )
    report = output_path.read_text(encoding="utf-8")
    assert "## Audio Behavior Signals" in report
    assert "- hesitation: medium" in report
    assert "- pauses before answers: high" in report
