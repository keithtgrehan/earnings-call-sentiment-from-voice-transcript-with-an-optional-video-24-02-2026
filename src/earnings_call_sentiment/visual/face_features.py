from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_EPS = 1e-6


def _mp_solutions() -> Any | None:
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("MediaPipe is required for visual behavior analysis. Install mediapipe.") from exc
    return getattr(mp, "solutions", None)


@dataclass
class FaceState:
    center_xy: tuple[float, float] | None = None
    gaze_offset: float | None = None


class FaceFeatureExtractor:
    def __init__(self, *, min_detection_confidence: float = 0.5) -> None:
        solutions = _mp_solutions()
        self._backend = "mediapipe" if solutions is not None else "opencv"
        self._face_detection = None
        self._face_mesh = None
        self._cascade = None
        if solutions is not None:
            self._face_detection = solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_detection_confidence,
            )
            self._face_mesh = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError(f"Unable to load OpenCV face cascade: {cascade_path}")
            self._cascade = cascade
        self._state = FaceState()

    def close(self) -> None:
        if self._face_detection is not None:
            self._face_detection.close()
        if self._face_mesh is not None:
            self._face_mesh.close()

    def __enter__(self) -> "FaceFeatureExtractor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if self._backend == "opencv":
            return self._process_opencv(frame_bgr)
        return self._process_mediapipe(frame_bgr)

    def _process_mediapipe(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import cv2  # type: ignore

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection_result = self._face_detection.process(rgb)
        mesh_result = self._face_mesh.process(rgb)
        detections = list(detection_result.detections or [])
        landmarks = list(mesh_result.multi_face_landmarks or [])
        face_detected = bool(detections)
        face_count = len(detections)
        detection_score = 0.0
        if detections:
            score_list = detections[0].score or [0.0]
            detection_score = float(score_list[0])

        feature_note = ""
        if not face_detected:
            feature_note = "no_face_detected"
        elif not landmarks:
            feature_note = "landmarks_unavailable"

        face_visible = bool(face_detected and landmarks and detection_score >= 0.45)
        if not face_visible:
            self._state = FaceState()
            return {
                "face_detected": face_detected,
                "face_count": int(face_count),
                "face_visible": False,
                "landmark_confidence": round(detection_score, 4),
                "head_shift_score": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "gaze_shift_proxy": 0.0,
                "blink_proxy": 0.0,
                "feature_note": feature_note or "low_face_visibility",
            }

        lmk = landmarks[0].landmark
        left_face = lmk[234]
        right_face = lmk[454]
        nose_tip = lmk[1]
        left_eye_outer = lmk[33]
        left_eye_inner = lmk[133]
        right_eye_inner = lmk[362]
        right_eye_outer = lmk[263]
        mouth_top = lmk[13]
        mouth_bottom = lmk[14]

        face_width = max(abs(right_face.x - left_face.x), _EPS)
        left_width = max(nose_tip.x - left_face.x, 0.0)
        right_width = max(right_face.x - nose_tip.x, 0.0)
        yaw = ((left_width - right_width) / face_width)

        eye_line_y = (left_eye_outer.y + right_eye_outer.y) / 2.0
        mouth_y = (mouth_top.y + mouth_bottom.y) / 2.0
        pitch = ((nose_tip.y - eye_line_y) / max(mouth_y - eye_line_y, _EPS)) - 0.5

        face_center = ((left_face.x + right_face.x) / 2.0, (left_face.y + right_face.y) / 2.0)
        previous_center = self._state.center_xy
        head_shift_score = 0.0
        if previous_center is not None:
            head_shift_score = float(
                math.hypot(face_center[0] - previous_center[0], face_center[1] - previous_center[1])
                / max(face_width, _EPS)
            )

        gaze_offset = self._gaze_offset(lmk)
        previous_gaze = self._state.gaze_offset
        gaze_shift = 0.0 if previous_gaze is None else abs(gaze_offset - previous_gaze)

        blink_proxy = self._blink_proxy(lmk)
        self._state = FaceState(center_xy=face_center, gaze_offset=gaze_offset)

        return {
            "face_detected": True,
            "face_count": int(face_count),
            "face_visible": True,
            "landmark_confidence": round(detection_score, 4),
            "head_shift_score": round(head_shift_score, 4),
            "head_yaw": round(float(yaw), 4),
            "head_pitch": round(float(pitch), 4),
            "gaze_shift_proxy": round(float(gaze_shift), 4),
            "blink_proxy": round(float(blink_proxy), 4),
            "feature_note": feature_note or "ok",
        }

    def _process_opencv(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import cv2  # type: ignore

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        face_detected = len(faces) > 0
        if not face_detected:
            self._state = FaceState()
            return {
                "face_detected": False,
                "face_count": 0,
                "face_visible": False,
                "landmark_confidence": 0.0,
                "head_shift_score": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "gaze_shift_proxy": 0.0,
                "blink_proxy": 0.0,
                "feature_note": "no_face_detected",
            }

        frame_h, frame_w = gray.shape[:2]
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center_xy = ((x + (w / 2.0)) / max(frame_w, 1), (y + (h / 2.0)) / max(frame_h, 1))
        face_width = max(float(w) / max(frame_w, 1), _EPS)
        previous_center = self._state.center_xy
        head_shift_score = 0.0
        if previous_center is not None:
            head_shift_score = float(
                math.hypot(center_xy[0] - previous_center[0], center_xy[1] - previous_center[1])
                / face_width
            )
        self._state = FaceState(center_xy=center_xy, gaze_offset=0.0)
        return {
            "face_detected": True,
            "face_count": int(len(faces)),
            "face_visible": True,
            "landmark_confidence": 0.6,
            "head_shift_score": round(head_shift_score, 4),
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "gaze_shift_proxy": 0.0,
            "blink_proxy": 0.0,
            "feature_note": "opencv_face_detection_only",
        }

    def _gaze_offset(self, lmk: list[Any]) -> float:
        if len(lmk) < 478:
            return 0.0
        left_iris = np.mean([[lmk[idx].x, lmk[idx].y] for idx in range(468, 473)], axis=0)
        right_iris = np.mean([[lmk[idx].x, lmk[idx].y] for idx in range(473, 478)], axis=0)

        left_offset = self._iris_offset(left_iris[0], lmk[33].x, lmk[133].x)
        right_offset = self._iris_offset(right_iris[0], lmk[362].x, lmk[263].x)
        return float((abs(left_offset) + abs(right_offset)) / 2.0)

    @staticmethod
    def _iris_offset(iris_x: float, a_x: float, b_x: float) -> float:
        low = min(a_x, b_x)
        high = max(a_x, b_x)
        width = max(high - low, _EPS)
        return ((iris_x - low) / width) - 0.5

    def _blink_proxy(self, lmk: list[Any]) -> float:
        left_ear = self._eye_aspect_ratio(lmk, 33, 133, 159, 145, 158, 153)
        right_ear = self._eye_aspect_ratio(lmk, 362, 263, 386, 374, 387, 373)
        ear = (left_ear + right_ear) / 2.0
        return 1.0 if ear < 0.18 else 0.0

    @staticmethod
    def _eye_aspect_ratio(
        lmk: list[Any],
        left_idx: int,
        right_idx: int,
        top_a: int,
        bottom_a: int,
        top_b: int,
        bottom_b: int,
    ) -> float:
        left = np.array([lmk[left_idx].x, lmk[left_idx].y])
        right = np.array([lmk[right_idx].x, lmk[right_idx].y])
        top_one = np.array([lmk[top_a].x, lmk[top_a].y])
        bottom_one = np.array([lmk[bottom_a].x, lmk[bottom_a].y])
        top_two = np.array([lmk[top_b].x, lmk[top_b].y])
        bottom_two = np.array([lmk[bottom_b].x, lmk[bottom_b].y])
        vertical = (np.linalg.norm(top_one - bottom_one) + np.linalg.norm(top_two - bottom_two)) / 2.0
        horizontal = max(np.linalg.norm(left - right), _EPS)
        return float(vertical / horizontal)
