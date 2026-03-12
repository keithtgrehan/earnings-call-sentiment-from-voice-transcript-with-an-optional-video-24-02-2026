from __future__ import annotations

from pathlib import Path

import pandas as pd

from earnings_call_sentiment import cli
from earnings_call_sentiment.media_quality import build_media_quality_summary
from earnings_call_sentiment.multimodal_support import build_multimodal_support_summary


def test_media_quality_summary_combines_audio_and_video_gates() -> None:
    summary = build_media_quality_summary(
        audio_summary={
            "audio_quality_ok": False,
            "limitations": ["Audio SNR proxy is weak."],
            "quality_gate": {"suppression_recommended": True},
        },
        visual_summary={
            "video_quality_ok": True,
            "limitations": ["Q&A visual shift is based on few segments."],
            "quality_gate": {"suppression_recommended": False},
        },
    )

    assert summary["audio_quality_ok"] is False
    assert summary["video_quality_ok"] is True
    assert summary["suppression_flags"]["audio_support_suppressed"] is True
    assert summary["suppression_flags"]["video_support_suppressed"] is False
    assert any("audio:" in note for note in summary["quality_notes"])


def test_multimodal_support_is_cautionary_when_audio_conflicts_with_green_transcript() -> None:
    media_quality = {
        "audio_quality_ok": True,
        "video_quality_ok": False,
    }
    summary = build_multimodal_support_summary(
        metrics_payload={"overall_review_signal": "green"},
        qa_shift_summary={"prepared_remarks_vs_q_and_a": {"label": "weaker"}},
        audio_summary={
            "hesitation_level": {"level": "high"},
            "pause_pressure_delta": {"label": "more_paused_under_questions"},
            "answer_latency_pressure": {"level": "medium"},
        },
        visual_summary=None,
        media_quality=media_quality,
    )

    assert summary["audio_support_direction"] == "cautionary"
    assert summary["video_support_direction"] == "unavailable"
    assert summary["multimodal_alignment"] == "low"
    assert summary["multimodal_confidence_adjustment"] < 0


def test_multimodal_support_stays_unavailable_when_media_is_missing() -> None:
    summary = build_multimodal_support_summary(
        metrics_payload={"overall_review_signal": "amber"},
        qa_shift_summary={"prepared_remarks_vs_q_and_a": {"label": "mixed"}},
        audio_summary=None,
        visual_summary=None,
        media_quality={"audio_quality_ok": False, "video_quality_ok": False},
    )

    assert summary["audio_support_direction"] == "unavailable"
    assert summary["video_support_direction"] == "unavailable"
    assert summary["multimodal_confidence_adjustment"] == 0


def test_multimodal_support_prefers_model_backed_audio_when_available() -> None:
    summary = build_multimodal_support_summary(
        metrics_payload={"overall_review_signal": "green"},
        qa_shift_summary={"prepared_remarks_vs_q_and_a": {"label": "weaker"}},
        audio_summary={
            "model_support": {
                "available": True,
                "support_direction": "cautionary",
                "calibrated_support_score": 0.42,
            }
        },
        visual_summary={
            "model_support": {
                "available": False,
            }
        },
        media_quality={"audio_quality_ok": True, "video_quality_ok": True},
    )

    assert summary["audio_support_direction"] == "cautionary"
    assert summary["fusion_mode"] == "hybrid"
    assert summary["calibrated_support_score"] > 0
    assert summary["multimodal_confidence_adjustment"] < 0


def test_report_markdown_includes_media_quality_and_multimodal_sections(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"
    cli._write_report_markdown(
        output_path=output_path,
        metrics_payload={"num_chunks_scored": 2, "sentiment_mean": 0.1, "sentiment_std": 0.2, "guidance": {"row_count": 1, "mean_strength": 0.4}},
        guidance_df=pd.DataFrame(),
        guidance_revision_df=pd.DataFrame(),
        behavioral_summary={
            "uncertainty_score_overall": {"level": "low"},
            "reassurance_score_management": {"level": "medium"},
            "analyst_skepticism_score": {"level": "medium"},
            "strongest_evidence": {},
        },
        qa_shift_summary={
            "prepared_remarks_vs_q_and_a": {"label": "mixed"},
            "analyst_skepticism": {"level": "medium"},
            "management_answers_vs_prepared_uncertainty": {"label": "mixed"},
            "early_vs_late_q_and_a": {"label": "mixed"},
            "strongest_evidence": {},
        },
        audio_summary={
            "audio_features_available": True,
            "hesitation_overall": {"level": "medium"},
            "pauses_before_answers": {"level": "medium"},
            "prepared_baseline_audio_stability": {"level": "medium"},
            "qa_hesitation_shift": {"level": "medium"},
            "answer_latency_pressure": {"level": "medium"},
            "audio_confidence_support": {"level": "low", "suppressed": True, "reason": "quality gate suppressed confidence uplift"},
            "support_mode": "model_backed",
            "model_support": {"available": True, "support_direction": "cautionary", "calibrated_support_score": 0.33},
            "strongest_audio_evidence": [
                {
                    "segment_id": 3,
                    "start_time_s": 10.0,
                    "end_time_s": 18.0,
                    "hesitation_label": "high",
                    "pause_before_answer_ms": 820.0,
                    "answer_onset_delay_ms": 710.0,
                }
            ],
            "low_confidence_segments": [{"confidence_note": "short segment limits audio confidence"}],
        },
        visual_summary={
            "visual_features_available": True,
            "face_visibility_overall": {"level": "medium"},
            "prepared_baseline_visual_stability": {"level": "medium"},
            "qa_visual_shift_score": {"level": "medium"},
            "facial_tension_level": {"level": "medium"},
            "head_motion_pressure": {"level": "high"},
            "visual_confidence_support": {"level": "low", "suppressed": True, "reason": "quality gate suppressed visual confidence uplift"},
            "support_mode": "heuristic_fallback",
            "strongest_visual_evidence": [
                {
                    "segment_id": 4,
                    "start_time_s": 12.0,
                    "end_time_s": 21.0,
                    "visual_change_score": 0.22,
                    "head_motion_energy": 0.18,
                }
            ],
            "notable_low_confidence_segments": [{"confidence_note": "low face visibility reduces confidence"}],
        },
        media_quality={
            "audio_quality_ok": False,
            "video_quality_ok": False,
            "quality_notes": ["audio: short answer windows reduce confidence"],
        },
        multimodal_summary={
            "transcript_primary_assessment": "amber",
            "audio_support_direction": "cautionary",
            "video_support_direction": "unavailable",
            "fusion_mode": "hybrid",
            "calibrated_support_score": 0.33,
            "multimodal_alignment": "medium",
            "multimodal_confidence_adjustment": -2,
            "notes": ["Transcript-first signal remains amber."],
        },
    )
    report = output_path.read_text(encoding="utf-8")
    assert "## Media Quality" in report
    assert "## Multimodal Support" in report
    assert "- audio support direction: cautionary" in report
    assert "- video support direction: unavailable" in report
    assert "- fusion mode: hybrid" in report
    assert "- support mode: model_backed" in report
