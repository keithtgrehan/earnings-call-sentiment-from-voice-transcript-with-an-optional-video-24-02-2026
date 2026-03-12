from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pause_features import AudioEnvelope, AudioMetadata, load_audio_envelope
from .segment_aggregate import SEGMENT_COLUMNS, aggregate_audio_segments

AUDIO_SCHEMA_VERSION = "1.0.0"


def _empty_segments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEGMENT_COLUMNS)


def _level_from_mean(score: float) -> str:
    if score >= 4.0:
        return "high"
    if score >= 2.0:
        return "medium"
    return "low"


def _stability_level(score: float) -> str:
    if score <= 1.2:
        return "high"
    if score <= 2.5:
        return "medium"
    return "low"


def _pause_level(median_ms: float) -> str:
    if median_ms >= 650.0:
        return "high"
    if median_ms >= 250.0:
        return "medium"
    return "low"


def _shift_level(delta: float) -> str:
    if abs(delta) >= 1.5:
        return "high"
    if abs(delta) >= 0.6:
        return "medium"
    return "low"


def _shift_direction(delta: float) -> str:
    if delta >= 0.6:
        return "more_hesitant"
    if delta <= -0.6:
        return "less_hesitant"
    return "mixed"


def _truncate(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _summary_unavailable(reason: str, *, metadata: AudioMetadata | None = None) -> dict[str, Any]:
    limitations = [reason]
    if metadata is not None:
        limitations.append(
            f"audio metadata: duration={metadata.duration_s:.2f}s, sample_rate={metadata.sample_rate}Hz"
        )
    return {
        "schema_version": AUDIO_SCHEMA_VERSION,
        "audio_available": metadata is not None,
        "audio_features_available": False,
        "hesitation_overall": {"score": 0.0, "level": "low"},
        "pauses_before_answers": {"median_ms": 0.0, "level": "low"},
        "prepared_baseline_audio_stability": {"score": 0.0, "level": "high"},
        "qa_hesitation_shift": {"delta": 0.0, "level": "low", "direction": "mixed"},
        "most_hesitant_answers": [],
        "low_confidence_segments": [],
        "limitations": limitations,
        "notes": [
            "Audio behavior signals are observational proxies only; they are not emotion or deception inference.",
            "Pause and hesitation features are most useful on management answers with clear Q&A timing.",
        ],
    }


def _segment_items(frame: pd.DataFrame, *, limit: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, row in frame.head(limit).iterrows():
        items.append(
            {
                "segment_id": int(row["segment_id"]),
                "section": str(row["section"]),
                "speaker_role": str(row["speaker_role"]),
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "hesitation_label": str(row["hesitation_label"]),
                "hesitation_score": int(row["hesitation_score"]),
                "pause_before_answer_ms": (
                    float(row["pause_before_answer_ms"])
                    if pd.notna(row["pause_before_answer_ms"])
                    else None
                ),
                "filler_density": float(row["filler_density"]),
                "confidence_note": str(row["confidence_note"]),
                "text": _truncate(str(row["text"])),
            }
        )
    return items


def _build_summary(segments_df: pd.DataFrame, envelope: AudioEnvelope) -> dict[str, Any]:
    if segments_df.empty:
        return _summary_unavailable(
            "No transcript-aligned audio segments were available for this run.",
            metadata=envelope.metadata,
        )

    management_df = segments_df[segments_df["speaker_role"] == "management"].copy()
    prepared_df = management_df[management_df["section"] == "prepared_remarks"].copy()
    answer_df = management_df[management_df["section"] == "q_and_a"].copy()

    prepared_score = float(prepared_df["hesitation_score"].mean()) if not prepared_df.empty else 0.0
    answer_score = float(answer_df["hesitation_score"].mean()) if not answer_df.empty else 0.0
    pause_source = answer_df[answer_df["pause_before_answer_ms"].notna()].copy() if not answer_df.empty else pd.DataFrame()
    pause_series = pause_source["pause_before_answer_ms"].dropna() if not pause_source.empty else pd.Series(dtype="float64")
    pause_median = float(pause_series.median()) if not pause_series.empty else 0.0
    hesitation_overall = answer_score if not answer_df.empty else float(management_df["hesitation_score"].mean() if not management_df.empty else 0.0)

    changed_answers = (
        answer_df.assign(_has_pause=answer_df["pause_before_answer_ms"].notna().astype(int))
        .sort_values(
            ["_has_pause", "hesitation_score", "pause_before_answer_ms", "filler_density"],
            ascending=[False, False, False, False],
        )
        .drop(columns="_has_pause")
    )
    usable_answers = changed_answers[changed_answers["confidence_note"] == "usable audio segment"].copy()
    low_confidence = segments_df[segments_df["confidence_note"] != "usable audio segment"].copy()

    limitations: list[str] = []
    if answer_df.empty:
        limitations.append("No management answer segments were available, so Q&A hesitation shift is limited.")
    if not low_confidence.empty:
        limitations.append("Some segments are short or have sparse sampled frames, which lowers confidence.")
    if pause_series.empty:
        limitations.append("Pause-before-answer metrics were unavailable because answer/question pairing was sparse.")

    return {
        "schema_version": AUDIO_SCHEMA_VERSION,
        "audio_available": True,
        "audio_features_available": True,
        "audio_metadata": {
            "path": str(envelope.metadata.path),
            "duration_s": round(float(envelope.metadata.duration_s), 3),
            "sample_rate": int(envelope.metadata.sample_rate),
            "frame_length_s": round(float(envelope.metadata.frame_length_s), 4),
            "hop_length_s": round(float(envelope.metadata.hop_length_s), 4),
            "silence_threshold_db": float(envelope.metadata.silence_threshold_db),
        },
        "hesitation_overall": {
            "score": round(hesitation_overall, 4),
            "level": _level_from_mean(hesitation_overall),
        },
        "pauses_before_answers": {
            "median_ms": round(pause_median, 1),
            "level": _pause_level(pause_median),
        },
        "prepared_baseline_audio_stability": {
            "score": round(prepared_score, 4),
            "level": _stability_level(prepared_score),
        },
        "qa_hesitation_shift": {
            "delta": round(answer_score - prepared_score, 4),
            "prepared_mean": round(prepared_score, 4),
            "q_and_a_mean": round(answer_score, 4),
            "direction": _shift_direction(answer_score - prepared_score),
            "level": _shift_level(answer_score - prepared_score),
        },
        "most_hesitant_answers": _segment_items(usable_answers if not usable_answers.empty else changed_answers),
        "low_confidence_segments": _segment_items(low_confidence.sort_values(["hesitation_score", "start_time_s"], ascending=[False, True])),
        "limitations": limitations,
        "notes": [
            "Audio hesitation combines pause-before-answer, within-segment silence, filler markers, and a lightweight speech-rate proxy.",
            "These are observational timing proxies for review support, not mental-state or deception inference.",
        ],
    }


def compute_audio_behavior_outputs(
    audio_path: Path | None,
    qa_segments_df: pd.DataFrame,
) -> dict[str, Any]:
    if audio_path is None:
        return {
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable("No audio source was available for this run."),
        }

    envelope = load_audio_envelope(audio_path)
    segments_df = aggregate_audio_segments(envelope, qa_segments_df)
    summary = _build_summary(segments_df, envelope)
    return {
        "segments_df": segments_df,
        "summary": summary,
    }


def write_audio_behavior_outputs(
    audio_path: Path | None,
    qa_segments_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    payload = compute_audio_behavior_outputs(audio_path, qa_segments_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_path = out_dir / "audio_behavior_segments.csv"
    summary_path = out_dir / "audio_behavior_summary.json"
    payload["segments_df"].to_csv(segments_path, index=False)
    summary_path.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    return {
        **payload,
        "segments_path": segments_path,
        "summary_path": summary_path,
    }
