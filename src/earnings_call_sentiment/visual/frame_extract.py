from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterator

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int


def is_video_path(path: Path | str | None) -> bool:
    if path is None:
        return False
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def _cv2() -> object:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency checked in runtime integration
        raise RuntimeError(
            "OpenCV is required for visual behavior analysis. Install opencv-python-headless."
        ) from exc
    return cv2


def _probe_with_ffprobe(video_path: Path) -> VideoMetadata | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height,nb_frames:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    fmt = payload.get("format") or {}
    duration_s = float(fmt.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    nb_frames_raw = stream.get("nb_frames")
    frame_count = int(nb_frames_raw) if str(nb_frames_raw or "").isdigit() else 0
    avg_frame_rate = str(stream.get("avg_frame_rate") or "0/1")
    try:
        num, den = avg_frame_rate.split("/", 1)
        fps = float(num) / float(den)
    except Exception:
        fps = 0.0
    return VideoMetadata(
        path=video_path,
        duration_s=duration_s,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def probe_video_metadata(video_path: Path) -> VideoMetadata:
    resolved = video_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"Video path not found: {resolved}")

    ffprobe_meta = _probe_with_ffprobe(resolved)
    if ffprobe_meta is not None:
        return ffprobe_meta

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {resolved}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_s = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return VideoMetadata(
            path=resolved,
            duration_s=duration_s,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def iter_sampled_frames(
    video_path: Path,
    *,
    sample_fps: float = 1.0,
    max_width: int = 640,
) -> Iterator[dict[str, object]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    cv2 = _cv2()
    metadata = probe_video_metadata(video_path)
    capture = cv2.VideoCapture(str(metadata.path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {metadata.path}")

    native_fps = metadata.fps if metadata.fps > 0 else 30.0
    stride = max(1, int(round(native_fps / sample_fps)))
    frame_index = 0
    sampled_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            timestamp_s = float(frame_index / native_fps)
            height, width = frame.shape[:2]
            if max_width > 0 and width > max_width:
                scale = float(max_width) / float(width)
                frame = cv2.resize(frame, (max_width, max(1, int(round(height * scale)))))
            yield {
                "sample_index": sampled_index,
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "frame_bgr": frame,
                "metadata": metadata,
            }
            sampled_index += 1
            frame_index += 1
    finally:
        capture.release()
