from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.media_support_models import score_audio_support

from .pause_features import AudioEnvelope, AudioMetadata, load_audio_envelope, snr_proxy_db
from .segment_aggregate import SEGMENT_COLUMNS, aggregate_audio_segments

AUDIO_SCHEMA_VERSION = "1.2.0"


def _empty_segments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEGMENT_COLUMNS)


def _level_from_mean(score: float) -> str:
    if score >= 5.0:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


def _stability_level(score: float) -> str:
    if score <= 1.4:
        return "high"
    if score <= 3.0:
        return "medium"
    return "low"


def _pause_level(median_ms: float) -> str:
    if median_ms >= 650.0:
        return "high"
    if median_ms >= 250.0:
        return "medium"
    return "low"


def _shift_level(delta: float) -> str:
    if abs(delta) >= 2.0:
        return "high"
    if abs(delta) >= 0.8:
        return "medium"
    return "low"


def _shift_direction(delta: float) -> str:
    if delta >= 0.8:
        return "more_hesitant"
    if delta <= -0.8:
        return "less_hesitant"
    return "mixed"


def _pause_pressure_label(delta: float) -> str:
    if delta >= 0.5:
        return "more_paused_under_questions"
    if delta <= -0.5:
        return "less_paused_under_questions"
    return "mixed"


def _answer_latency_level(median_ms: float) -> str:
    if median_ms >= 650.0:
        return "high"
    if median_ms >= 250.0:
        return "medium"
    return "low"


def _truncate(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _summary_unavailable(reason: str, *, metadata: AudioMetadata | None = None) -> dict[str, Any]:
    limitations = [reason]
    if metadata is not None:
        limitations.append(
            f"audio metadata: duration={metadata.duration_s:.2f}s, sample_rate={metadata.sample_rate}Hz"
        )
    quality_gate = {
        "quality_ok": False,
        "suppression_recommended": True,
        "snr_proxy_db": 0.0,
        "snr_proxy_ok": False,
        "management_speech_duration_s": 0.0,
        "enough_speech_duration_ok": False,
        "usable_management_segments": 0,
        "usable_answer_segments": 0,
        "alignment_quality_ratio": 0.0,
        "alignment_quality_ok": False,
    }
    return {
        "schema_version": AUDIO_SCHEMA_VERSION,
        "audio_available": metadata is not None,
        "audio_features_available": False,
        "audio_quality_ok": False,
        "quality_gate": quality_gate,
        "hesitation_overall": {"score": 0.0, "level": "low"},
        "hesitation_level": {"score": 0.0, "level": "low"},
        "pauses_before_answers": {"median_ms": 0.0, "level": "low"},
        "prepared_baseline_audio_stability": {"score": 0.0, "level": "high"},
        "qa_hesitation_shift": {"delta": 0.0, "level": "low", "direction": "mixed"},
        "pause_pressure_delta": {
            "delta": 0.0,
            "prepared_density": 0.0,
            "q_and_a_density": 0.0,
            "level": "low",
            "label": "mixed",
        },
        "answer_latency_pressure": {"median_ms": 0.0, "level": "low"},
        "audio_confidence_support": {
            "level": "low",
            "suppressed": True,
            "reason": reason,
        },
        "support_mode": "heuristic_fallback",
        "model_support": {
            "available": False,
            "mode": "heuristic_fallback",
            "support_direction": "unavailable",
            "calibrated_support_score": 0.0,
            "reason": reason,
        },
        "egemaps_v02_summary": {"available": False},
        "strongest_audio_evidence": [],
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
                "answer_onset_delay_ms": (
                    float(row["answer_onset_delay_ms"])
                    if pd.notna(row["answer_onset_delay_ms"])
                    else None
                ),
                "pause_density_per_10s": float(row.get("pause_density_per_10s", 0.0)),
                "articulation_rate_wpm": float(row.get("articulation_rate_wpm", 0.0)),
                "confidence_note": str(row["confidence_note"]),
                "text": _truncate(str(row["text"])),
            }
        )
    return items


def _quality_gate(
    *,
    segments_df: pd.DataFrame,
    management_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    envelope: AudioEnvelope,
) -> dict[str, Any]:
    usable_management = management_df[management_df["confidence_note"] == "usable audio segment"].copy()
    usable_answers = answer_df[answer_df["confidence_note"] == "usable audio segment"].copy()
    management_speech_duration = float(management_df.get("speech_duration_s", pd.Series(dtype="float64")).sum())
    snr_value = snr_proxy_db(envelope)
    alignment_ratio = (
        float(len(usable_management) / len(management_df))
        if not management_df.empty
        else 0.0
    )
    quality_ok = (
        snr_value >= 8.0
        and management_speech_duration >= 2.0
        and len(usable_management) >= 2
        and len(usable_answers) >= 1
        and alignment_ratio >= 0.5
    )
    return {
        "quality_ok": bool(quality_ok),
        "suppression_recommended": not bool(quality_ok),
        "snr_proxy_db": round(float(snr_value), 4),
        "snr_proxy_ok": bool(snr_value >= 8.0),
        "management_speech_duration_s": round(management_speech_duration, 4),
        "enough_speech_duration_ok": bool(management_speech_duration >= 2.0),
        "usable_management_segments": int(len(usable_management)),
        "usable_answer_segments": int(len(usable_answers)),
        "alignment_quality_ratio": round(alignment_ratio, 4),
        "alignment_quality_ok": bool(alignment_ratio >= 0.5),
    }


def _confidence_support_level(quality_gate: dict[str, Any], answer_df: pd.DataFrame, egemaps_available: bool) -> str:
    score = 0
    if bool(quality_gate.get("quality_ok")):
        score += 2
    if _as_int(quality_gate.get("usable_answer_segments")) >= 1:
        score += 1
    if bool(quality_gate.get("snr_proxy_ok")):
        score += 1
    if egemaps_available:
        score += 1
    if len(answer_df) >= 2:
        score += 1
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _egemaps_summary(answer_df: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "egemaps_pitch_variability",
        "egemaps_loudness_mean",
        "egemaps_loudness_variability",
        "egemaps_spectral_flux",
    ]
    available = all(column in answer_df.columns and answer_df[column].notna().any() for column in columns)
    if not available:
        return {"available": False}
    return {
        "available": True,
        "pitch_variability_mean": round(float(answer_df["egemaps_pitch_variability"].dropna().mean()), 4),
        "loudness_mean": round(float(answer_df["egemaps_loudness_mean"].dropna().mean()), 4),
        "loudness_variability_mean": round(float(answer_df["egemaps_loudness_variability"].dropna().mean()), 4),
        "spectral_flux_mean": round(float(answer_df["egemaps_spectral_flux"].dropna().mean()), 4),
    }


def _strongest_audio_evidence(answer_df: pd.DataFrame) -> list[dict[str, Any]]:
    if answer_df.empty:
        return []
    ranked = answer_df.sort_values(
        [
            "hesitation_score",
            "pause_before_answer_ms",
            "pause_density_per_10s",
            "filler_density",
        ],
        ascending=[False, False, False, False],
    )
    items: list[dict[str, Any]] = []
    for _, row in ranked.head(3).iterrows():
        items.append(
            {
                "segment_id": int(row["segment_id"]),
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "hesitation_label": str(row["hesitation_label"]),
                "pause_before_answer_ms": (
                    float(row["pause_before_answer_ms"])
                    if pd.notna(row["pause_before_answer_ms"])
                    else None
                ),
                "answer_onset_delay_ms": (
                    float(row["answer_onset_delay_ms"])
                    if pd.notna(row["answer_onset_delay_ms"])
                    else None
                ),
                "pause_density_per_10s": float(row.get("pause_density_per_10s", 0.0)),
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
    hesitation_overall = (
        answer_score
        if not answer_df.empty
        else float(management_df["hesitation_score"].mean() if not management_df.empty else 0.0)
    )

    pause_source = answer_df[answer_df["pause_before_answer_ms"].notna()].copy() if not answer_df.empty else pd.DataFrame()
    pause_series = pause_source["pause_before_answer_ms"].dropna() if not pause_source.empty else pd.Series(dtype="float64")
    latency_series = (
        answer_df["answer_onset_delay_ms"].dropna()
        if not answer_df.empty and "answer_onset_delay_ms" in answer_df.columns
        else pd.Series(dtype="float64")
    )
    pause_median = float(pause_series.median()) if not pause_series.empty else 0.0
    latency_median = float(latency_series.median()) if not latency_series.empty else 0.0
    prepared_pause_density = (
        float(prepared_df["pause_density_per_10s"].mean())
        if not prepared_df.empty and "pause_density_per_10s" in prepared_df.columns
        else 0.0
    )
    answer_pause_density = (
        float(answer_df["pause_density_per_10s"].mean())
        if not answer_df.empty and "pause_density_per_10s" in answer_df.columns
        else 0.0
    )
    quality_gate = _quality_gate(
        segments_df=segments_df,
        management_df=management_df,
        answer_df=answer_df,
        envelope=envelope,
    )
    egemaps_summary = _egemaps_summary(answer_df)
    audio_confidence_level = _confidence_support_level(quality_gate, answer_df, bool(egemaps_summary.get("available")))

    changed_answers = answer_df.sort_values(
        ["hesitation_score", "pause_before_answer_ms", "pause_density_per_10s", "filler_density"],
        ascending=[False, False, False, False],
    )
    usable_answers = changed_answers[changed_answers["confidence_note"] == "usable audio segment"].copy()
    low_confidence = segments_df[segments_df["confidence_note"] != "usable audio segment"].copy()

    limitations: list[str] = []
    if answer_df.empty:
        limitations.append("No management answer segments were available, so Q&A hesitation shift is limited.")
    if not bool(quality_gate["snr_proxy_ok"]):
        limitations.append("Audio SNR proxy is weak, so hesitation support should be treated cautiously.")
    if not bool(quality_gate["alignment_quality_ok"]):
        limitations.append("Transcript/audio alignment is sparse or short for several segments.")
    if not bool(quality_gate["enough_speech_duration_ok"]):
        limitations.append("Management speech duration is short, which limits audio confidence.")
    if pause_series.empty:
        limitations.append("Pause-before-answer metrics were unavailable because answer/question pairing was sparse.")

    model_support = score_audio_support(segments_df)

    return {
        "schema_version": AUDIO_SCHEMA_VERSION,
        "audio_available": True,
        "audio_features_available": True,
        "audio_quality_ok": bool(quality_gate["quality_ok"]),
        "quality_gate": quality_gate,
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
        "hesitation_level": {
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
        "pause_pressure_delta": {
            "delta": round(answer_pause_density - prepared_pause_density, 4),
            "prepared_density": round(prepared_pause_density, 4),
            "q_and_a_density": round(answer_pause_density, 4),
            "level": _shift_level(answer_pause_density - prepared_pause_density),
            "label": _pause_pressure_label(answer_pause_density - prepared_pause_density),
        },
        "answer_latency_pressure": {
            "median_ms": round(latency_median, 1),
            "level": _answer_latency_level(latency_median),
        },
        "audio_confidence_support": {
            "level": audio_confidence_level,
            "suppressed": not bool(quality_gate["quality_ok"]),
            "reason": (
                "quality gate suppressed confidence uplift"
                if not bool(quality_gate["quality_ok"])
                else "usable answer-level audio support"
            ),
        },
        "support_mode": str(model_support.get("mode", "heuristic_fallback")),
        "model_support": model_support,
        "egemaps_v02_summary": egemaps_summary,
        "strongest_audio_evidence": _strongest_audio_evidence(
            usable_answers if not usable_answers.empty else changed_answers
        ),
        "most_hesitant_answers": _segment_items(
            usable_answers if not usable_answers.empty else changed_answers
        ),
        "low_confidence_segments": _segment_items(
            low_confidence.sort_values(["hesitation_score", "start_time_s"], ascending=[False, True])
        ),
        "limitations": limitations,
        "notes": [
            "Audio hesitation combines pause-before-answer, answer latency, within-segment pause density, filler markers, articulation rate, and loudness dynamics.",
            "These are observational timing proxies for review support, not mental-state or deception inference.",
        ],
    }


def compute_audio_behavior_outputs(
    audio_path: Path | None,
    qa_segments_df: pd.DataFrame,
    *,
    use_opensmile: bool = True,
) -> dict[str, Any]:
    if audio_path is None:
        return {
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable("No audio source was available for this run."),
        }

    try:
        envelope = load_audio_envelope(audio_path)
    except Exception as exc:
        return {
            "segments_df": _empty_segments_df(),
            "summary": _summary_unavailable(f"Audio analysis unavailable: {exc}"),
        }

    segments_df = aggregate_audio_segments(envelope, qa_segments_df, use_opensmile=use_opensmile)
    summary = _build_summary(segments_df, envelope)
    return {
        "segments_df": segments_df,
        "summary": summary,
    }


def write_audio_behavior_outputs(
    audio_path: Path | None,
    qa_segments_df: pd.DataFrame,
    out_dir: Path,
    *,
    use_opensmile: bool = True,
) -> dict[str, Any]:
    payload = compute_audio_behavior_outputs(audio_path, qa_segments_df, use_opensmile=use_opensmile)
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
