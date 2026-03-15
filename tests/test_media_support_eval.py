from __future__ import annotations

from pathlib import Path

from earnings_call_sentiment.media_support_eval import (
    build_visual_trainability_report,
    load_runtime_smoke_manifest,
    load_segment_labels,
)
from earnings_call_sentiment.media_support_casepack import (
    build_downstream_case_frame,
    build_task_impact_case_frame,
)
from earnings_call_sentiment.visual import runtime as visual_runtime


def test_media_support_eval_seed_files_have_expected_counts() -> None:
    labels = load_segment_labels()

    assert len(labels) >= 64
    counts = labels["feature_modality"].value_counts().to_dict()
    assert counts["audio"] >= 52
    assert counts["video"] >= 12

    runtime_smoke = load_runtime_smoke_manifest()
    assert len(runtime_smoke) >= 3
    assert set(runtime_smoke["runtime_success"].astype(str).str.lower()) == {"true"}


def test_visual_trainability_report_flags_remaining_group_gap_honestly() -> None:
    report = build_visual_trainability_report()

    assert report["video_label_rows_total"] >= 18
    assert report["video_label_rows_with_visual_tension"] >= 12
    assert report["source_groups_with_visual_tension_labels"] == 2
    assert report["basic_grouped_eval_ready"] is True
    assert report["defensible_grouped_eval_ready"] is False
    assert report["minimum_next_data"]["additional_groups_for_basic_grouped_eval"] == 0
    assert report["minimum_next_data"]["additional_groups_for_defensible_grouped_eval"] == 1


def test_multimodal_runtime_status_reports_expected_keys(monkeypatch) -> None:
    class DummyCV2:
        __version__ = "4.10.0"

    class DummyMP:
        __version__ = "0.10.32"
        solutions = object()

    class DummyVision:
        FaceLandmarker = object
        PoseLandmarker = object

    monkeypatch.setattr(visual_runtime, "load_cv2", lambda: DummyCV2())
    monkeypatch.setattr(visual_runtime, "load_mediapipe", lambda: DummyMP())
    monkeypatch.setattr(visual_runtime, "mediapipe_tasks_vision", lambda: DummyVision())

    status = visual_runtime.multimodal_runtime_status()
    assert status["cv2_import_ok"] is True
    assert status["mediapipe_import_ok"] is True
    assert status["face_landmarker_available"] is True
    assert status["pose_landmarker_available"] is True


def test_media_support_casepack_paths_are_repo_relative() -> None:
    downstream = build_downstream_case_frame()
    task_impact = build_task_impact_case_frame()

    downstream_columns = [
        "source_path",
        "metrics_path",
        "qa_shift_path",
        "audio_summary_path",
        "visual_summary_path",
        "saved_multimodal_summary_path",
    ]
    task_columns = [
        "baseline_transcript_path",
        "treatment_report_path",
        "treatment_metrics_path",
        "treatment_transcript_path",
    ]

    for column in downstream_columns:
        values = downstream[column].astype(str).str.strip()
        assert not any(Path(value).is_absolute() for value in values if value)

    for column in task_columns:
        values = task_impact[column].astype(str).str.strip()
        assert not any(Path(value).is_absolute() for value in values if value)
