from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_EPS = 1e-6


def _mp() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "MediaPipe is required for visual behavior analysis. Install mediapipe."
        ) from exc
    return getattr(mp, "solutions", None)


@dataclass
class PoseState:
    shoulder_center: tuple[float, float] | None = None


class PoseFeatureExtractor:
    def __init__(self, *, min_detection_confidence: float = 0.5) -> None:
        solutions = _mp()
        self._pose = None
        if solutions is not None:
            self._pose = solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
        self._state = PoseState()

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()

    def __enter__(self) -> "PoseFeatureExtractor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if self._pose is None:
            self._state = PoseState()
            return {
                "pose_visible": False,
                "hand_visible": False,
                "shoulder_shift_score": 0.0,
            }
        import cv2  # type: ignore

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        landmarks = getattr(result, "pose_landmarks", None)
        if landmarks is None:
            self._state = PoseState()
            return {
                "pose_visible": False,
                "hand_visible": False,
                "shoulder_shift_score": 0.0,
            }

        lmk = landmarks.landmark
        left_shoulder = lmk[11]
        right_shoulder = lmk[12]
        shoulder_visibility = (float(left_shoulder.visibility) + float(right_shoulder.visibility)) / 2.0
        pose_visible = shoulder_visibility >= 0.45
        if not pose_visible:
            self._state = PoseState()
            return {
                "pose_visible": False,
                "hand_visible": False,
                "shoulder_shift_score": 0.0,
            }

        shoulder_center = (
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        )
        shoulder_width = max(abs(right_shoulder.x - left_shoulder.x), _EPS)
        previous_center = self._state.shoulder_center
        shoulder_shift = 0.0
        if previous_center is not None:
            shoulder_shift = float(
                math.hypot(shoulder_center[0] - previous_center[0], shoulder_center[1] - previous_center[1])
                / shoulder_width
            )
        self._state = PoseState(shoulder_center=shoulder_center)

        left_wrist = lmk[15]
        right_wrist = lmk[16]
        hand_visible = bool(float(left_wrist.visibility) >= 0.4 or float(right_wrist.visibility) >= 0.4)
        return {
            "pose_visible": True,
            "hand_visible": hand_visible,
            "shoulder_shift_score": round(float(shoulder_shift), 4),
        }
