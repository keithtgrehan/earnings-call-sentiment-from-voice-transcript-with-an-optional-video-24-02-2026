from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from earnings_call_sentiment import cli
from earnings_call_sentiment.visual.segment_aggregate import aggregate_visual_segments
from earnings_call_sentiment.visual.summary import compute_visual_behavior_outputs


def test_visual_behavior_summary_handles_missing_video() -> None:
    payload = compute_visual_behavior_outputs(None, pd.DataFrame())
    assert payload["frames_df"].empty
    assert payload["segments_df"].empty
    assert payload["summary"]["video_available"] is False
    assert payload["summary"]["visual_features_available"] is False


def test_visual_segment_aggregation_builds_labels_and_notes() -> None:
    frames_df = pd.DataFrame(
        [
            {
                "timestamp_s": 0.0,
                "frame_index": 0,
                "face_visible": True,
                "landmark_confidence": 0.9,
                "motion_score": 0.02,
                "head_shift_score": 0.01,
                "gaze_shift_proxy": 0.01,
                "blink_proxy": 0.0,
                "shoulder_shift_score": 0.01,
            },
            {
                "timestamp_s": 1.0,
                "frame_index": 30,
                "face_visible": True,
                "landmark_confidence": 0.92,
                "motion_score": 0.2,
                "head_shift_score": 0.18,
                "gaze_shift_proxy": 0.14,
                "blink_proxy": 1.0,
                "shoulder_shift_score": 0.15,
            },
            {
                "timestamp_s": 2.0,
                "frame_index": 60,
                "face_visible": False,
                "landmark_confidence": 0.0,
                "motion_score": 0.0,
                "head_shift_score": 0.0,
                "gaze_shift_proxy": 0.0,
                "blink_proxy": 0.0,
                "shoulder_shift_score": 0.0,
            },
        ]
    )
    qa_segments_df = pd.DataFrame(
        [
            {"segment_id": 0, "start": 0.0, "end": 0.9, "phase": "prepared_remarks", "speaker_role": "management", "text": "Prepared."},
            {"segment_id": 1, "start": 1.0, "end": 2.1, "phase": "q_and_a", "speaker_role": "management", "text": "Answer."},
        ]
    )
    segments_df = aggregate_visual_segments(frames_df, qa_segments_df)
    assert list(segments_df["visual_stability_label"]) == ["stable", "somewhat_changed"]
    assert segments_df.iloc[0]["confidence_note"] == "usable visual segment"
    assert segments_df.iloc[1]["confidence_note"] == "usable visual segment"


def test_report_markdown_includes_visual_section_when_summary_present(tmp_path: Path) -> None:
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
        visual_summary={
            "visual_features_available": True,
            "face_visibility_overall": {"level": "high"},
            "prepared_baseline_visual_stability": {"level": "medium"},
            "qa_visual_shift_score": {"level": "high"},
            "most_visually_changed_segments": [
                {
                    "section": "q_and_a",
                    "start_time_s": 10.0,
                    "end_time_s": 20.0,
                    "visual_change_score": 0.2,
                }
            ],
            "notable_low_confidence_segments": [
                {"confidence_note": "low face visibility reduces confidence"}
            ],
        },
    )
    report = output_path.read_text(encoding="utf-8")
    assert "## Visual Behavior Signals" in report
    assert "- face visibility: high" in report
    assert "- Q&A visual shift: high" in report

