"""Optional WhisperX alignment sidecars and supporting helpers.

These utilities are intentionally separate from the default transcription
pipeline. They write inspectable sidecar artifacts only and do not modify
`transcript.json`, `transcript.txt`, or deterministic signal outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from earnings_call_sentiment.optional_runtime import (
    load_multimodal_config,
    load_pyannote_audio,
    load_whisperx,
)
from earnings_call_sentiment.transcriber import get_audio_duration_s

DEFAULT_ALIGNMENT_DIR = Path("data/processed/multimodal/alignment")
DEFAULT_WHISPERX_MODEL = "small"
DEFAULT_LANGUAGE = "en"
DEFAULT_BATCH_SIZE = 4
DEFAULT_COMPUTE_TYPE = "int8"


@dataclass(frozen=True)
class AlignmentRunConfig:
    source_id: str
    audio_path: Path
    transcript_path: Path | None
    transcript_text: str | None
    language: str
    device: str
    whisperx_model: str
    compute_type: str
    batch_size: int
    out_dir: Path
    enable_diarization: bool
    min_speakers: int | None
    max_speakers: int | None
    return_char_alignments: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_alignment_dir(*, source_id: str, out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        base = Path(out_dir)
    else:
        base = repo_root() / DEFAULT_ALIGNMENT_DIR
    return base.expanduser().resolve() / source_id


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_segment_rows(
    raw_segments: list[dict[str, Any]],
    *,
    audio_duration_s: float,
) -> tuple[list[dict[str, Any]], str]:
    normalized: list[dict[str, Any]] = []
    all_have_times = True
    text_parts: list[str] = []

    for row in raw_segments:
        text = _clean_text(row.get("text"))
        if not text:
            continue
        start = row.get("start")
        end = row.get("end")
        if start is None or end is None:
            all_have_times = False
        else:
            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                all_have_times = False
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )
        text_parts.append(text)

    if not normalized:
        return [], "empty"

    if all_have_times and all(
        isinstance(item["start"], float)
        and isinstance(item["end"], float)
        and float(item["end"]) > float(item["start"])
        for item in normalized
    ):
        return [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": item["text"],
            }
            for item in normalized
        ], "provided_segment_times"

    return [
        {
            "start": 0.0,
            "end": float(audio_duration_s),
            "text": " ".join(text_parts).strip(),
        }
    ], "collapsed_full_text"


def load_reference_segments(
    *,
    audio_path: Path,
    transcript_path: Path | None = None,
    transcript_text: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str]:
    audio_duration_s = get_audio_duration_s(audio_path)

    if transcript_text is not None and transcript_text.strip():
        return (
            [
                {
                    "start": 0.0,
                    "end": float(audio_duration_s),
                    "text": transcript_text.strip(),
                }
            ],
            "provided_text",
        )

    if transcript_path is None:
        return None, "whisperx_asr"

    resolved = transcript_path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
            raw_segments = [item for item in raw["segments"] if isinstance(item, dict)]
        elif isinstance(raw, list):
            raw_segments = [item for item in raw if isinstance(item, dict)]
        else:
            raise RuntimeError(f"Unsupported transcript JSON format: {resolved}")
        segments, mode = _normalize_segment_rows(
            raw_segments,
            audio_duration_s=audio_duration_s,
        )
        if not segments:
            raise RuntimeError(f"No usable transcript segments found in {resolved}")
        return segments, f"transcript_json:{mode}"

    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Transcript text is empty: {resolved}")
    return (
        [
            {
                "start": 0.0,
                "end": float(audio_duration_s),
                "text": text,
            }
        ],
        f"transcript_text:{resolved.suffix.lower() or 'plain'}",
    )


def _segment_confidence(words: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for word in words:
        score = word.get("score")
        try:
            if score is not None:
                scores.append(float(score))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return round(float(mean(scores)), 4)


def _word_rows_from_segments(
    *,
    source_id: str,
    aligned_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for utterance_index, segment in enumerate(aligned_segments, start=1):
        utterance_id = f"{source_id}_utt_{utterance_index:04d}"
        words = segment.get("words") or []
        for word_index, word in enumerate(words, start=1):
            rows.append(
                {
                    "source_id": source_id,
                    "utterance_id": utterance_id,
                    "word_index": word_index,
                    "word": _clean_text(word.get("word")),
                    "start_time_s": word.get("start"),
                    "end_time_s": word.get("end"),
                    "speaker_label": word.get("speaker"),
                    "alignment_confidence": word.get("score"),
                }
            )
    return rows


def _segment_rows_from_aligned(
    *,
    source_id: str,
    aligned_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for utterance_index, segment in enumerate(aligned_segments, start=1):
        utterance_id = f"{source_id}_utt_{utterance_index:04d}"
        words = segment.get("words") or []
        rows.append(
            {
                "source_id": source_id,
                "utterance_id": utterance_id,
                "start_time_s": segment.get("start"),
                "end_time_s": segment.get("end"),
                "utterance_text": _clean_text(segment.get("text")),
                "speaker_label": segment.get("speaker"),
                "alignment_confidence": _segment_confidence(words),
                "word_count": len(words),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _diarization_rows(annotation: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append(
            {
                "start_time_s": round(float(turn.start), 4),
                "end_time_s": round(float(turn.end), 4),
                "speaker_label": str(speaker),
            }
        )
    return rows


def _overlap_seconds(
    start_a: float | None,
    end_a: float | None,
    start_b: float | None,
    end_b: float | None,
) -> float:
    if None in {start_a, end_a, start_b, end_b}:
        return 0.0
    left = max(float(start_a), float(start_b))
    right = min(float(end_a), float(end_b))
    return max(0.0, right - left)


def _assign_speaker_label(
    *,
    start_time_s: float | None,
    end_time_s: float | None,
    diarization_rows: list[dict[str, Any]],
) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for row in diarization_rows:
        overlap = _overlap_seconds(
            start_time_s,
            end_time_s,
            row.get("start_time_s"),
            row.get("end_time_s"),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(row.get("speaker_label") or "")
    return best_speaker or None


def _attach_diarization_to_alignment(
    *,
    aligned_result: dict[str, Any],
    diarization_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    segments = aligned_result.get("segments") or []
    updated_segments: list[dict[str, Any]] = []
    for segment in segments:
        updated = dict(segment)
        segment_speaker = _assign_speaker_label(
            start_time_s=_as_float_or_none(segment.get("start")),
            end_time_s=_as_float_or_none(segment.get("end")),
            diarization_rows=diarization_rows,
        )
        if segment_speaker is not None:
            updated["speaker"] = segment_speaker

        updated_words: list[dict[str, Any]] = []
        for word in segment.get("words") or []:
            updated_word = dict(word)
            word_speaker = _assign_speaker_label(
                start_time_s=_as_float_or_none(word.get("start")),
                end_time_s=_as_float_or_none(word.get("end")),
                diarization_rows=diarization_rows,
            )
            if word_speaker is not None:
                updated_word["speaker"] = word_speaker
            updated_words.append(updated_word)
        if updated_words:
            updated["words"] = updated_words
        updated_segments.append(updated)

    result = dict(aligned_result)
    result["segments"] = updated_segments
    return result


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_hf_token(config_hf_env: str | None) -> str | None:
    if config_hf_env is None:
        return None
    value = os.getenv(config_hf_env, "").strip()
    return value or None


def run_whisperx_alignment(config: AlignmentRunConfig) -> dict[str, Any]:
    whisperx = load_whisperx()
    multimodal_config = load_multimodal_config()
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    provided_segments, transcript_mode = load_reference_segments(
        audio_path=config.audio_path,
        transcript_path=config.transcript_path,
        transcript_text=config.transcript_text,
    )

    audio = whisperx.load_audio(str(config.audio_path))

    alignment_source = "provided_transcript"
    language = config.language
    result: dict[str, Any]

    if provided_segments is None:
        alignment_source = "whisperx_asr"
        model_kwargs: dict[str, Any] = {}
        if multimodal_config.model_cache_dir is not None:
            model_kwargs["download_root"] = str(multimodal_config.model_cache_dir)
        model = whisperx.load_model(
            config.whisperx_model,
            config.device,
            compute_type=config.compute_type,
            **model_kwargs,
        )
        transcribe_kwargs: dict[str, Any] = {
            "batch_size": config.batch_size,
        }
        if config.language:
            transcribe_kwargs["language"] = config.language
        result = model.transcribe(audio, **transcribe_kwargs)
        language = str(result.get("language") or config.language or DEFAULT_LANGUAGE)
        segments_to_align = result.get("segments") or []
    else:
        result = {
            "language": language,
            "segments": provided_segments,
        }
        segments_to_align = provided_segments

    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=config.device,
    )
    aligned_result = whisperx.align(
        segments_to_align,
        align_model,
        align_metadata,
        audio,
        config.device,
        return_char_alignments=config.return_char_alignments,
    )
    aligned_result["language"] = language

    diarization_rows: list[dict[str, Any]] = []
    diarization_enabled = False
    diarization_reason = "not_requested"

    if config.enable_diarization:
        if not multimodal_config.pyannote_enabled:
            raise RuntimeError(
                "Diarization was requested, but pyannote is disabled. "
                "Set EARNINGS_CALL_PYANNOTE_ENABLED=1 to enable experimental diarization."
            )
        hf_token = _resolve_hf_token(multimodal_config.hf_token_env)
        if hf_token is None:
            raise RuntimeError(
                "Diarization was requested, but no Hugging Face token env is configured. "
                "Set EARNINGS_CALL_HF_TOKEN, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN."
            )
        pyannote_audio = load_pyannote_audio()
        pipeline = pyannote_audio.Pipeline.from_pretrained(
            multimodal_config.pyannote_model,
            token=hf_token,
        )
        if config.device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        diarize_kwargs: dict[str, Any] = {}
        if config.min_speakers is not None:
            diarize_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers is not None:
            diarize_kwargs["max_speakers"] = config.max_speakers
        annotation = pipeline(str(config.audio_path), **diarize_kwargs)
        diarization_rows = _diarization_rows(annotation)
        aligned_result = _attach_diarization_to_alignment(
            aligned_result=aligned_result,
            diarization_rows=diarization_rows,
        )
        diarization_enabled = True
        diarization_reason = "applied"

    aligned_segments = aligned_result.get("segments") or []
    segment_rows = _segment_rows_from_aligned(
        source_id=config.source_id,
        aligned_segments=aligned_segments,
    )
    word_rows = _word_rows_from_segments(
        source_id=config.source_id,
        aligned_segments=aligned_segments,
    )

    alignment_payload = {
        "source_id": config.source_id,
        "audio_path": str(config.audio_path),
        "transcript_source": transcript_mode,
        "alignment_source": alignment_source,
        "language": language,
        "device": config.device,
        "whisperx_model": config.whisperx_model,
        "compute_type": config.compute_type,
        "batch_size": config.batch_size,
        "diarization_requested": bool(config.enable_diarization),
        "diarization_applied": diarization_enabled,
        "diarization_reason": diarization_reason,
        "segments": aligned_segments,
        "diarization_segments": diarization_rows,
    }

    _write_json(out_dir / "aligned_transcript.json", alignment_payload)
    _write_csv(
        out_dir / "aligned_segments.csv",
        segment_rows,
        fieldnames=[
            "source_id",
            "utterance_id",
            "start_time_s",
            "end_time_s",
            "utterance_text",
            "speaker_label",
            "alignment_confidence",
            "word_count",
        ],
    )
    _write_csv(
        out_dir / "aligned_words.csv",
        word_rows,
        fieldnames=[
            "source_id",
            "utterance_id",
            "word_index",
            "word",
            "start_time_s",
            "end_time_s",
            "speaker_label",
            "alignment_confidence",
        ],
    )
    if diarization_rows:
        _write_csv(
            out_dir / "diarization_segments.csv",
            diarization_rows,
            fieldnames=["start_time_s", "end_time_s", "speaker_label"],
        )

    summary = {
        "source_id": config.source_id,
        "output_dir": str(out_dir),
        "audio_path": str(config.audio_path),
        "transcript_source": transcript_mode,
        "alignment_source": alignment_source,
        "language": language,
        "device": config.device,
        "diarization_requested": bool(config.enable_diarization),
        "diarization_applied": diarization_enabled,
        "diarization_reason": diarization_reason,
        "segment_count": len(segment_rows),
        "word_count": len(word_rows),
        "diarization_segment_count": len(diarization_rows),
        "artifacts": {
            "aligned_transcript_json": str(out_dir / "aligned_transcript.json"),
            "aligned_segments_csv": str(out_dir / "aligned_segments.csv"),
            "aligned_words_csv": str(out_dir / "aligned_words.csv"),
            "diarization_segments_csv": (
                str(out_dir / "diarization_segments.csv") if diarization_rows else None
            ),
        },
        "notes": [
            "This sidecar is optional and does not replace the repo's default transcription path.",
            "Diarization remains supporting/experimental and is not required for normal runs.",
        ],
    }
    _write_json(out_dir / "alignment_summary.json", summary)
    return summary


def build_segment_candidates_from_alignment(
    *,
    alignment_payload: dict[str, Any],
    speaker_role_hint: str = "unknown",
    labeling_status: str = "pending_review",
) -> list[dict[str, Any]]:
    source_id = str(alignment_payload.get("source_id") or "").strip()
    segments = alignment_payload.get("segments") or []
    diarization_applied = bool(alignment_payload.get("diarization_applied"))
    rows: list[dict[str, Any]] = []

    for index, segment in enumerate(segments, start=1):
        rows.append(
            {
                "candidate_segment_id": f"{source_id}_candidate_{index:04d}",
                "source_id": source_id,
                "start_time_s": segment.get("start"),
                "end_time_s": segment.get("end"),
                "utterance_text": _clean_text(segment.get("text")),
                "speaker_label": segment.get("speaker"),
                "speaker_role_hint": speaker_role_hint,
                "transcript_ref": f"aligned_transcript.json:segments[{index - 1}]",
                "alignment_confidence": _segment_confidence(segment.get("words") or []),
                "diarization_available": diarization_applied,
                "labeling_status": labeling_status,
                "notes": "Alignment-derived candidate segment for later manual review.",
            }
        )
    return rows


def write_segment_candidates_csv(
    *,
    alignment_json_path: Path,
    out_path: Path | None = None,
    speaker_role_hint: str = "unknown",
    labeling_status: str = "pending_review",
) -> Path:
    payload = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    rows = build_segment_candidates_from_alignment(
        alignment_payload=payload,
        speaker_role_hint=speaker_role_hint,
        labeling_status=labeling_status,
    )
    target = out_path or alignment_json_path.with_name("segment_candidates.csv")
    _write_csv(
        target,
        rows,
        fieldnames=[
            "candidate_segment_id",
            "source_id",
            "start_time_s",
            "end_time_s",
            "utterance_text",
            "speaker_label",
            "speaker_role_hint",
            "transcript_ref",
            "alignment_confidence",
            "diarization_available",
            "labeling_status",
            "notes",
        ],
    )
    return target
