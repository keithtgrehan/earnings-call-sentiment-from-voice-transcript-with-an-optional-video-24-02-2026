from __future__ import annotations

from typing import Any

import pandas as pd

from .pause_features import (
    AudioEnvelope,
    count_fillers,
    leading_silence_before,
    silence_ratio,
    speech_rate_wpm,
)

SEGMENT_COLUMNS = [
    "segment_id",
    "section",
    "speaker_role",
    "qa_pair_id",
    "start_time_s",
    "end_time_s",
    "pause_before_answer_ms",
    "answer_onset_delay_ms",
    "silence_ratio",
    "filler_count",
    "filler_density",
    "matched_fillers",
    "speech_rate_wpm",
    "hesitation_score",
    "hesitation_label",
    "confidence_note",
    "frame_count",
    "word_count",
    "text",
]


def _hesitation_label(score: int) -> str:
    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _pause_score(pause_ms: float | None) -> int:
    if pause_ms is None or pd.isna(pause_ms):
        return 0
    value = float(pause_ms)
    if value >= 900.0:
        return 3
    if value >= 450.0:
        return 2
    if value >= 200.0:
        return 1
    return 0


def _silence_score(ratio: float) -> int:
    if ratio >= 0.25:
        return 3
    if ratio >= 0.15:
        return 2
    if ratio >= 0.08:
        return 1
    return 0


def _filler_score(density: float) -> int:
    if density >= 6.0:
        return 3
    if density >= 2.5:
        return 2
    if density > 0.0:
        return 1
    return 0


def _speech_rate_score(words_per_minute: float, word_count: int) -> int:
    if word_count < 12 or words_per_minute <= 0.0:
        return 0
    if words_per_minute < 85.0:
        return 1
    return 0


def _confidence_note(duration_s: float, frame_count: int, word_count: int) -> str:
    if frame_count <= 0:
        return "no sampled audio frames in segment"
    if duration_s < 1.0:
        return "short segment limits audio confidence"
    if word_count < 7:
        return "low transcript word count limits audio confidence"
    if frame_count < 5:
        return "sparse sampled audio frames limit confidence"
    return "usable audio segment"


def aggregate_audio_segments(
    envelope: AudioEnvelope,
    qa_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    if qa_segments_df.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    rows: list[dict[str, Any]] = []
    question_end_by_pair: dict[int, float] = {}
    seen_answer_pair_ids: set[int] = set()

    for _, row in qa_segments_df.sort_values("start").iterrows():
        start_time_s = float(row.get("start", 0.0))
        end_time_s = float(row.get("end", start_time_s))
        duration_s = max(0.0, end_time_s - start_time_s)
        text = str(row.get("text", "") or "").strip()
        section = str(row.get("phase", "prepared_remarks") or "prepared_remarks")
        speaker_role = str(row.get("speaker_role", "management") or "management")
        qa_pair_id = int(row.get("qa_pair_id", 0) or 0)

        if speaker_role == "analyst" and qa_pair_id > 0:
            question_end_by_pair[qa_pair_id] = end_time_s

        segment_silence_ratio, frame_count = silence_ratio(envelope, start_time_s, end_time_s)
        filler_count, filler_matches = count_fillers(text)
        speech_rate, word_count = speech_rate_wpm(text, duration_s)
        filler_density = float((filler_count / word_count) * 100.0) if word_count > 0 else 0.0

        pause_before_answer_ms = float("nan")
        answer_onset_delay_ms = float("nan")
        if (
            speaker_role == "management"
            and section == "q_and_a"
            and qa_pair_id > 0
            and qa_pair_id not in seen_answer_pair_ids
        ):
            seen_answer_pair_ids.add(qa_pair_id)
            pause_before_answer_ms = leading_silence_before(envelope, start_time_s)
            question_end = question_end_by_pair.get(qa_pair_id)
            if question_end is not None:
                answer_onset_delay_ms = max(0.0, start_time_s - float(question_end)) * 1000.0

        hesitation_score = (
            _pause_score(None if pd.isna(pause_before_answer_ms) else float(pause_before_answer_ms))
            + _silence_score(float(segment_silence_ratio))
            + _filler_score(float(filler_density))
            + _speech_rate_score(float(speech_rate), int(word_count))
        )

        rows.append(
            {
                "segment_id": int(row.get("segment_id", len(rows))),
                "section": section,
                "speaker_role": speaker_role,
                "qa_pair_id": qa_pair_id,
                "start_time_s": round(start_time_s, 4),
                "end_time_s": round(end_time_s, 4),
                "pause_before_answer_ms": round(float(pause_before_answer_ms), 1)
                if not pd.isna(pause_before_answer_ms)
                else None,
                "answer_onset_delay_ms": round(float(answer_onset_delay_ms), 1)
                if not pd.isna(answer_onset_delay_ms)
                else None,
                "silence_ratio": round(float(segment_silence_ratio), 4),
                "filler_count": int(filler_count),
                "filler_density": round(float(filler_density), 4),
                "matched_fillers": ", ".join(filler_matches),
                "speech_rate_wpm": round(float(speech_rate), 2),
                "hesitation_score": int(hesitation_score),
                "hesitation_label": _hesitation_label(int(hesitation_score)),
                "confidence_note": _confidence_note(duration_s, frame_count, word_count),
                "frame_count": int(frame_count),
                "word_count": int(word_count),
                "text": text,
            }
        )

    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
