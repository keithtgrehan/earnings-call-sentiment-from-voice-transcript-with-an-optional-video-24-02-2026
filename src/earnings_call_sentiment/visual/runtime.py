from __future__ import annotations

from typing import Any


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenCV is required for visual behavior analysis. Install opencv-python-headless."
        ) from exc
    return cv2


def load_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MediaPipe is required for visual behavior analysis. Install mediapipe.") from exc
    return mp


def mediapipe_solutions() -> Any | None:
    try:
        return getattr(load_mediapipe(), "solutions", None)
    except RuntimeError:
        return None


def mediapipe_tasks_vision() -> Any | None:
    try:
        from mediapipe.tasks.python import vision  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    return vision


def multimodal_runtime_status() -> dict[str, Any]:
    notes: list[str] = []
    fallback_notes: list[str] = []

    cv2_version: str | None = None
    cv2_ok = False
    try:
        cv2 = load_cv2()
        cv2_version = str(getattr(cv2, "__version__", "unknown"))
        cv2_ok = True
        notes.append(f"cv2 import ok ({cv2_version}).")
    except RuntimeError as exc:
        fallback_notes.append(str(exc))

    mediapipe_version: str | None = None
    mediapipe_ok = False
    try:
        mp = load_mediapipe()
        mediapipe_version = str(getattr(mp, "__version__", "unknown"))
        mediapipe_ok = True
        notes.append(f"mediapipe import ok ({mediapipe_version}).")
    except RuntimeError as exc:
        fallback_notes.append(str(exc))

    vision = mediapipe_tasks_vision()
    face_landmarker_ok = bool(vision is not None and hasattr(vision, "FaceLandmarker"))
    pose_landmarker_ok = bool(vision is not None and hasattr(vision, "PoseLandmarker"))
    if face_landmarker_ok and pose_landmarker_ok:
        notes.append("MediaPipe task classes for face and pose landmarking are available.")
    elif mediapipe_ok:
        fallback_notes.append(
            "MediaPipe is installed, but FaceLandmarker/PoseLandmarker task classes were not both available."
        )

    opensmile_ok = False
    opensmile_version: str | None = None
    try:
        import opensmile  # type: ignore

        opensmile_version = str(getattr(opensmile, "__version__", "unknown"))
        opensmile_ok = True
        notes.append(f"opensmile import ok ({opensmile_version}).")
    except Exception as exc:  # pragma: no cover - optional dependency
        fallback_notes.append(f"openSMILE unavailable: {exc}")

    if mediapipe_ok and not face_landmarker_ok:
        fallback_notes.append(
            "Visual extraction will continue to use the FaceMesh/Pose solutions path unless task model assets are configured."
        )

    return {
        "cv2_import_ok": cv2_ok,
        "cv2_version": cv2_version,
        "mediapipe_import_ok": mediapipe_ok,
        "mediapipe_version": mediapipe_version,
        "face_landmarker_available": face_landmarker_ok,
        "pose_landmarker_available": pose_landmarker_ok,
        "opensmile_available": opensmile_ok,
        "opensmile_version": opensmile_version,
        "runtime_notes": notes,
        "fallback_notes": fallback_notes,
    }
