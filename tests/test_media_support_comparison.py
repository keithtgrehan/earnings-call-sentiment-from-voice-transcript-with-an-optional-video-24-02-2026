from __future__ import annotations

import json

import pandas as pd

from earnings_call_sentiment import media_support_comparison
from earnings_call_sentiment.media_support_comparison import (
    aggregate_support_target,
    evaluate_downstream_decision_cases,
    summarize_task_impact_results,
)


def test_aggregate_support_target_uses_signed_mean_threshold() -> None:
    labels = pd.DataFrame(
        {
            "multimodal_support_direction": [
                "cautionary",
                "cautionary",
                "supportive",
                "neutral",
            ]
        }
    )

    result = aggregate_support_target(labels)

    assert result["counts"]["cautionary"] == 2
    assert result["counts"]["supportive"] == 1
    assert result["target_support_signed_mean"] == 0.25
    assert result["target_support_direction"] == "cautionary"


def test_evaluate_downstream_decision_cases_tracks_unscored_rows_honestly() -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "call01",
                "metrics_path": "",
                "qa_shift_path": "",
                "audio_summary_path": "",
                "visual_summary_path": "",
                "saved_multimodal_summary_path": "",
                "target_support_direction": "supportive",
                "target_support_signed_mean": "-0.3",
                "gold_guidance_label": "raised",
            },
            {
                "case_id": "call02",
                "metrics_path": "",
                "qa_shift_path": "",
                "audio_summary_path": "",
                "visual_summary_path": "",
                "saved_multimodal_summary_path": "",
                "target_support_direction": "",
                "target_support_signed_mean": "",
                "gold_guidance_label": "unclear",
            },
        ]
    )

    result_frame, summary = evaluate_downstream_decision_cases(cases)

    assert len(result_frame) == 2
    assert summary["case_count"] == 2
    assert summary["case_count_with_support_targets"] == 1
    assert summary["case_count_without_support_targets"] == 1
    assert summary["gold_guidance_label_distribution"]["unclear"] == 1


def test_evaluate_downstream_decision_cases_resolves_repo_relative_artifact_paths(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "outputs" / "case01").mkdir(parents=True)
    metrics_path = tmp_path / "outputs" / "case01" / "metrics.json"
    qa_shift_path = tmp_path / "outputs" / "case01" / "qa_shift_summary.json"
    audio_path = tmp_path / "outputs" / "case01" / "audio_behavior_summary.json"
    saved_path = tmp_path / "outputs" / "case01" / "multimodal_support_summary.json"

    metrics_path.write_text(json.dumps({"overall_review_signal": "green"}), encoding="utf-8")
    qa_shift_path.write_text(
        json.dumps({"prepared_remarks_vs_q_and_a": {"label": "weaker"}}),
        encoding="utf-8",
    )
    audio_path.write_text(
        json.dumps(
            {
                "audio_quality_ok": True,
                "model_support": {
                    "available": True,
                    "support_direction": "cautionary",
                    "calibrated_support_score": 0.42,
                    "reliability_weight": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    saved_path.write_text(json.dumps({"calibrated_support_score": 0.15}), encoding="utf-8")

    monkeypatch.setattr(media_support_comparison, "repo_root", lambda: tmp_path)

    cases = pd.DataFrame(
        [
            {
                "case_id": "call01",
                "metrics_path": "outputs/case01/metrics.json",
                "qa_shift_path": "outputs/case01/qa_shift_summary.json",
                "audio_summary_path": "outputs/case01/audio_behavior_summary.json",
                "visual_summary_path": "",
                "saved_multimodal_summary_path": "outputs/case01/multimodal_support_summary.json",
                "target_support_direction": "cautionary",
                "target_support_signed_mean": "0.3",
                "gold_guidance_label": "raised",
            }
        ]
    )

    result_frame, summary = evaluate_downstream_decision_cases(cases)

    assert len(result_frame) == 1
    assert result_frame.iloc[0]["current_direction"] == "cautionary"
    assert result_frame.iloc[0]["saved_direction"] == "cautionary"
    assert summary["case_count_with_support_targets"] == 1


def test_summarize_task_impact_results_scores_against_case_labels() -> None:
    cases = pd.DataFrame(
        [
            {"case_id": "task01", "gold_guidance_label": "raised"},
            {"case_id": "task02", "gold_guidance_label": "unclear"},
        ]
    )
    results = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "case_id": "task01",
                "condition": "control",
                "completion_seconds": "120",
                "predicted_guidance_label": "raised",
                "clarity_rating": "3",
                "evidence_quality_score": "2",
            },
            {
                "participant_id": "p1",
                "case_id": "task02",
                "condition": "assisted",
                "completion_seconds": "90",
                "predicted_guidance_label": "unclear",
                "clarity_rating": "4",
                "evidence_quality_score": "4",
            },
        ]
    )

    summary = summarize_task_impact_results(cases, results)

    assert summary["submission_count"] == 2
    assert summary["conditions"]["control"]["label_accuracy"] == 1.0
    assert summary["conditions"]["assisted"]["mean_completion_seconds"] == 90.0
