from __future__ import annotations

from earnings_call_sentiment.media_support_eval import validate_media_support_eval
from earnings_call_sentiment.visual import runtime as visual_runtime


def test_media_support_eval_seed_set_validates() -> None:
    summary = validate_media_support_eval()
    assert summary["status"] == "ok"
    assert summary["manifest_rows"] >= 2
    assert summary["label_rows"] >= 20
    assert summary["label_counts_by_modality"]["audio"] >= 10


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
