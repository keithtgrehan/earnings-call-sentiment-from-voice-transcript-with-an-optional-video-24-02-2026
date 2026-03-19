"""Optional NLP assist sidecars for transcript chunks and segment windows.

These helpers write inspectable comparison artifacts only. They do not replace
the existing deterministic labels, summaries, or score outputs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from earnings_call_sentiment.optional_runtime import load_multimodal_config

DEFAULT_NLP_DIR = Path("data/processed/multimodal/nlp")
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 256
DEFAULT_SEGMENT_MANIFEST = Path("data/source_manifests/earnings_call_segments.csv")

CHUNK_TYPE_TRANSCRIPT = "transcript_chunk"
CHUNK_TYPE_PREPARED = "prepared_remarks"
CHUNK_TYPE_QA_QUESTION = "q_and_a_question"
CHUNK_TYPE_QA_ANSWER = "q_and_a_answer"
CHUNK_TYPE_OTHER = "other"

SUPPORTED_CHUNK_TYPES = {
    CHUNK_TYPE_TRANSCRIPT,
    CHUNK_TYPE_PREPARED,
    CHUNK_TYPE_QA_QUESTION,
    CHUNK_TYPE_QA_ANSWER,
    CHUNK_TYPE_OTHER,
}


@dataclass(frozen=True)
class NlpSegmentInput:
    source_id: str
    segment_id: str
    text: str
    chunk_type: str
    start_time_s: float | None
    end_time_s: float | None
    speaker_role: str
    input_row_index: int
    deterministic_label: str | None
    deterministic_score: float | None
    deterministic_signed_score: float | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_nlp_dir(*, source_id: str, out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        base = Path(out_dir)
    else:
        base = repo_root() / DEFAULT_NLP_DIR
    return base.expanduser().resolve() / source_id


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _coerce_float(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_present(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        if key not in row.index:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _text_looks_like_placeholder(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("planned:") or lowered.startswith("todo:")


def _normalize_chunk_type(
    *,
    explicit_chunk_type: str | None = None,
    phase: str | None = None,
    speaker_role: str | None = None,
    segment_type: str | None = None,
) -> str:
    if explicit_chunk_type:
        normalized = explicit_chunk_type.strip().lower()
        return normalized if normalized in SUPPORTED_CHUNK_TYPES else CHUNK_TYPE_OTHER

    if segment_type:
        normalized = segment_type.strip().lower()
        if normalized in SUPPORTED_CHUNK_TYPES:
            return normalized

    normalized_phase = _clean_text(phase).lower()
    normalized_role = _clean_text(speaker_role).lower()
    if normalized_phase == "prepared_remarks":
        return CHUNK_TYPE_PREPARED
    if normalized_phase in {"q_and_a", "q&a", "qa"}:
        if normalized_role == "analyst":
            return CHUNK_TYPE_QA_QUESTION
        if normalized_role == "management":
            return CHUNK_TYPE_QA_ANSWER
        return CHUNK_TYPE_OTHER
    return CHUNK_TYPE_TRANSCRIPT


def _resolve_device(device: str) -> int:
    normalized = _clean_text(device).lower() or "cpu"
    if normalized == "cpu":
        return -1
    if normalized.startswith("cuda"):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - import failure is environment-specific
            raise RuntimeError(
                "CUDA was requested for NLP sidecars, but torch could not be imported."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for NLP sidecars, but no CUDA device is available."
            )
        return 0
    raise RuntimeError(
        f"Unsupported NLP sidecar device '{device}'. Use 'cpu' or 'cuda'."
    )


def _cache_dir_from_config() -> str | None:
    config = load_multimodal_config()
    for candidate in (
        config.model_cache_dir,
        config.transformers_cache,
        config.hf_home,
    ):
        if candidate is not None:
            return str(candidate)
    return None


def _build_text_classifier(*, model_name: str, device: str):
    from transformers import pipeline

    cache_dir = _cache_dir_from_config()
    kwargs: dict[str, Any] = {
        "task": "text-classification",
        "model": model_name,
        "device": _resolve_device(device),
    }
    if cache_dir:
        kwargs["model_kwargs"] = {"cache_dir": cache_dir}
    return pipeline(**kwargs)


def _normalize_all_scores(raw_scores: Any) -> list[dict[str, Any]]:
    if isinstance(raw_scores, dict):
        normalized = [raw_scores]
    elif isinstance(raw_scores, list):
        normalized = [item for item in raw_scores if isinstance(item, dict)]
    else:
        normalized = []
    ordered = sorted(
        normalized,
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return [
        {
            "label": _clean_text(item.get("label")),
            "score": round(float(item.get("score", 0.0) or 0.0), 6),
        }
        for item in ordered
    ]


def _load_transcript_segments(transcript_path: Path) -> list[dict[str, Any]]:
    suffix = transcript_path.suffix.lower()
    if suffix != ".json":
        raise RuntimeError(
            "Segment-manifest text extraction expects a JSON transcript with segment "
            f"timings. Got: {transcript_path}"
        )

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_segments = payload.get("segments", [])
    elif isinstance(payload, list):
        raw_segments = payload
    else:
        raise RuntimeError(f"Unsupported transcript JSON format: {transcript_path}")

    segments: list[dict[str, Any]] = []
    for row in raw_segments:
        if not isinstance(row, dict):
            continue
        text = _clean_text(row.get("text"))
        if not text:
            continue
        start_time = _coerce_float(row.get("start"))
        end_time = _coerce_float(row.get("end"))
        segments.append(
            {
                "start": start_time,
                "end": end_time,
                "text": text,
            }
        )
    return segments


def _extract_text_window(
    *,
    start_time_s: float | None,
    end_time_s: float | None,
    transcript_segments: list[dict[str, Any]],
) -> str:
    if start_time_s is None or end_time_s is None:
        return ""

    parts: list[str] = []
    for row in transcript_segments:
        seg_start = _coerce_float(row.get("start"))
        seg_end = _coerce_float(row.get("end"))
        if seg_start is None or seg_end is None:
            continue
        if seg_end <= start_time_s or seg_start >= end_time_s:
            continue
        text = _clean_text(row.get("text"))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _build_segment_input_rows(
    frame: pd.DataFrame,
    *,
    source_id: str,
    chunk_types: set[str] | None = None,
) -> list[NlpSegmentInput]:
    rows: list[NlpSegmentInput] = []
    for input_row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        text = _clean_text(row.get("text"))
        if not text:
            continue
        chunk_type = _normalize_chunk_type(
            explicit_chunk_type=_clean_text(row.get("chunk_type")) or None,
            phase=_clean_text(row.get("phase")) or None,
            speaker_role=_clean_text(row.get("speaker_role")) or None,
            segment_type=_clean_text(row.get("segment_type")) or None,
        )
        if chunk_types is not None and chunk_type not in chunk_types:
            continue
        segment_id = _clean_text(row.get("segment_id")) or f"{source_id}_seg_{input_row_index:04d}"
        rows.append(
            NlpSegmentInput(
                source_id=source_id,
                segment_id=segment_id,
                text=text,
                chunk_type=chunk_type,
                start_time_s=_coerce_float(
                    _first_present(row, "start", "start_time_s", "start_time")
                ),
                end_time_s=_coerce_float(
                    _first_present(row, "end", "end_time_s", "end_time")
                ),
                speaker_role=_clean_text(row.get("speaker_role")),
                input_row_index=input_row_index,
                deterministic_label=_clean_text(
                    _first_present(row, "sentiment", "sentiment_label")
                )
                or None,
                deterministic_score=_coerce_float(
                    _first_present(row, "score", "sentiment_score")
                ),
                deterministic_signed_score=_coerce_float(
                    _first_present(
                        row,
                        "signed_score",
                        "signed_sentiment",
                        "sentiment_signed",
                    )
                ),
            )
        )
    return rows


def load_segment_inputs_from_chunks_csv(
    *,
    source_id: str,
    path: str | Path,
    chunk_types: set[str] | None = None,
) -> list[NlpSegmentInput]:
    frame = pd.read_csv(Path(path), keep_default_na=False)
    return _build_segment_input_rows(frame, source_id=source_id, chunk_types=chunk_types)


def load_segment_inputs_from_chunks_jsonl(
    *,
    source_id: str,
    path: str | Path,
    chunk_types: set[str] | None = None,
) -> list[NlpSegmentInput]:
    frame = pd.read_json(Path(path), lines=True)
    return _build_segment_input_rows(frame, source_id=source_id, chunk_types=chunk_types)


def load_segment_inputs_from_manifest(
    *,
    source_id: str,
    manifest_path: str | Path = DEFAULT_SEGMENT_MANIFEST,
    transcript_path: str | Path | None = None,
    chunk_types: set[str] | None = None,
) -> tuple[list[NlpSegmentInput], dict[str, int]]:
    frame = pd.read_csv(Path(manifest_path), dtype=str, keep_default_na=False)
    source_rows = frame[frame["source_id"] == source_id].copy()
    transcript_segments: list[dict[str, Any]] = []
    if transcript_path is not None:
        transcript_segments = _load_transcript_segments(Path(transcript_path))

    built_rows: list[NlpSegmentInput] = []
    skipped_placeholder = 0
    skipped_missing_text = 0

    for input_row_index, (_, row) in enumerate(source_rows.iterrows(), start=1):
        chunk_type = _normalize_chunk_type(
            segment_type=_clean_text(row.get("segment_type")) or None,
            speaker_role=_clean_text(row.get("speaker_role")) or None,
        )
        if chunk_types is not None and chunk_type not in chunk_types:
            continue

        transcript_ref = _clean_text(row.get("transcript_ref"))
        if transcript_ref and not _text_looks_like_placeholder(transcript_ref):
            text = transcript_ref
        else:
            text = _extract_text_window(
                start_time_s=_coerce_float(row.get("start_time")),
                end_time_s=_coerce_float(row.get("end_time")),
                transcript_segments=transcript_segments,
            )

        if transcript_ref and _text_looks_like_placeholder(transcript_ref) and not text:
            skipped_placeholder += 1
            continue
        if not text:
            skipped_missing_text += 1
            continue

        built_rows.append(
            NlpSegmentInput(
                source_id=source_id,
                segment_id=_clean_text(row.get("segment_id")) or f"{source_id}_seg_{input_row_index:04d}",
                text=text,
                chunk_type=chunk_type,
                start_time_s=_coerce_float(row.get("start_time")),
                end_time_s=_coerce_float(row.get("end_time")),
                speaker_role=_clean_text(row.get("speaker_role")),
                input_row_index=input_row_index,
                deterministic_label=None,
                deterministic_score=None,
                deterministic_signed_score=None,
            )
        )

    return built_rows, {
        "manifest_rows_considered": int(len(source_rows)),
        "manifest_rows_scored": int(len(built_rows)),
        "manifest_rows_skipped_placeholder_text": int(skipped_placeholder),
        "manifest_rows_skipped_missing_text": int(skipped_missing_text),
    }


def score_segments_with_model(
    *,
    segment_rows: list[NlpSegmentInput],
    model_name: str,
    model_role: str,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    if not segment_rows:
        return []

    classifier = _build_text_classifier(model_name=model_name, device=device)
    texts = [row.text for row in segment_rows]
    raw_outputs = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
        top_k=None,
    )

    scored_rows: list[dict[str, Any]] = []
    for row, raw_output in zip(segment_rows, raw_outputs, strict=True):
        all_scores = _normalize_all_scores(raw_output)
        if not all_scores:
            raise RuntimeError(
                f"Model '{model_name}' returned no label scores for segment {row.segment_id}."
            )
        top_score = all_scores[0]
        scored_rows.append(
            {
                "source_id": row.source_id,
                "segment_id": row.segment_id,
                "input_row_index": row.input_row_index,
                "start_time_s": row.start_time_s,
                "end_time_s": row.end_time_s,
                "speaker_role": row.speaker_role,
                "chunk_type": row.chunk_type,
                "text": row.text,
                "text_char_len": len(row.text),
                "model_role": model_role,
                "model_name": model_name,
                "predicted_label": top_score["label"],
                "predicted_score": top_score["score"],
                "label_scores_json": _json_dumps(all_scores),
                "deterministic_label": row.deterministic_label or "",
                "deterministic_score": row.deterministic_score,
                "deterministic_signed_score": row.deterministic_signed_score,
            }
        )
    return scored_rows


def build_nlp_scoring_summary(
    *,
    source_id: str,
    input_mode: str,
    rows: list[dict[str, Any]],
    notes: list[str] | None = None,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    by_model: dict[str, Any] = {}
    for model_name, model_frame in frame.groupby("model_name") if not frame.empty else []:
        by_model[str(model_name)] = {
            "row_count": int(len(model_frame)),
            "predicted_label_counts": {
                str(key): int(value)
                for key, value in model_frame["predicted_label"].value_counts().items()
            },
            "chunk_type_counts": {
                str(key): int(value)
                for key, value in model_frame["chunk_type"].value_counts().items()
            },
            "mean_predicted_score": round(
                float(model_frame["predicted_score"].mean()),
                4,
            ),
        }

    return {
        "source_id": source_id,
        "input_mode": input_mode,
        "rows_written": int(len(rows)),
        "models_run": list(by_model.keys()),
        "input_metadata": input_metadata or {},
        "by_model": by_model,
        "notes": notes or [
            "Deterministic labels remain the source of truth.",
            "These NLP outputs are sidecar evidence for inspection and disagreement analysis only.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_nlp_sidecar_outputs(
    *,
    source_id: str,
    input_mode: str,
    rows: list[dict[str, Any]],
    out_dir: str | Path | None = None,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    target_dir = default_nlp_dir(source_id=source_id, out_dir=out_dir)
    csv_path = target_dir / "nlp_segment_scores.csv"
    json_path = target_dir / "nlp_segment_scores.json"
    summary_path = target_dir / "nlp_scoring_summary.json"

    fieldnames = [
        "source_id",
        "segment_id",
        "input_row_index",
        "start_time_s",
        "end_time_s",
        "speaker_role",
        "chunk_type",
        "text",
        "text_char_len",
        "model_role",
        "model_name",
        "predicted_label",
        "predicted_score",
        "label_scores_json",
        "deterministic_label",
        "deterministic_score",
        "deterministic_signed_score",
    ]
    _write_csv(csv_path, rows, fieldnames=fieldnames)
    _write_json(
        json_path,
        {
            "source_id": source_id,
            "rows": rows,
        },
    )
    _write_json(
        summary_path,
        build_nlp_scoring_summary(
            source_id=source_id,
            input_mode=input_mode,
            rows=rows,
            input_metadata=input_metadata,
        ),
    )
    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "summary_path": summary_path,
    }


def _normalize_polarity_label(label: str) -> str:
    lowered = _clean_text(label).lower()
    if "positive" in lowered:
        return "positive"
    if "negative" in lowered:
        return "negative"
    if "neutral" in lowered:
        return "neutral"
    return lowered or "unknown"


def build_nlp_disagreement_summary(
    *,
    score_rows: list[dict[str, Any]],
    deterministic_out_dir: str | Path | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame(score_rows)
    if frame.empty:
        return {
            "status": "no_rows",
            "notes": [
                "No NLP sidecar rows were available for comparison.",
                "Deterministic labels remain the source of truth.",
            ],
        }

    source_id = str(frame.iloc[0]["source_id"])
    finbert_frame = frame[frame["model_role"] == "primary_finbert"].copy()
    emotion_frame = frame[frame["model_role"] == "secondary_emotion"].copy()

    def _counts(series: pd.Series) -> dict[str, int]:
        return {str(key): int(value) for key, value in series.value_counts().items()}

    def _chunk_means(model_frame: pd.DataFrame) -> dict[str, float]:
        if model_frame.empty:
            return {}
        return {
            str(chunk_type): round(float(chunk_frame["predicted_score"].mean()), 4)
            for chunk_type, chunk_frame in model_frame.groupby("chunk_type")
        }

    disagreement_examples: list[dict[str, Any]] = []
    comparable_count = 0
    disagreement_count = 0
    by_chunk_type: dict[str, dict[str, int]] = {}

    if not finbert_frame.empty and "deterministic_label" in finbert_frame.columns:
        comparable = finbert_frame[finbert_frame["deterministic_label"].astype(str).str.strip() != ""].copy()
        comparable_count = int(len(comparable))
        if comparable_count:
            comparable["deterministic_polarity"] = comparable["deterministic_label"].map(_normalize_polarity_label)
            comparable["finbert_polarity"] = comparable["predicted_label"].map(_normalize_polarity_label)
            comparable["agrees"] = comparable["deterministic_polarity"] == comparable["finbert_polarity"]
            disagreement_count = int((~comparable["agrees"]).sum())
            for chunk_type, chunk_frame in comparable.groupby("chunk_type"):
                by_chunk_type[str(chunk_type)] = {
                    "comparable_rows": int(len(chunk_frame)),
                    "disagreements": int((~chunk_frame["agrees"]).sum()),
                }
            disagreeing = comparable[~comparable["agrees"]].head(5)
            disagreement_examples = [
                {
                    "segment_id": str(row["segment_id"]),
                    "chunk_type": str(row["chunk_type"]),
                    "deterministic_label": str(row["deterministic_label"]),
                    "finbert_label": str(row["predicted_label"]),
                    "finbert_score": round(float(row["predicted_score"]), 4),
                    "text": _clean_text(row["text"])[:220],
                }
                for _, row in disagreeing.iterrows()
            ]

    summary: dict[str, Any] = {
        "status": "ok",
        "source_id": source_id,
        "notes": [
            "Deterministic outputs remain the source of truth.",
            "This summary is for side-by-side inspection and disagreement analysis only.",
            "FinBERT is finance-domain text classification support, not a final decision layer.",
            "The generic emotion model is supporting context only and is not treated as finance-ground-truth emotion labeling.",
        ],
        "finbert": {
            "row_count": int(len(finbert_frame)),
            "predicted_label_counts": _counts(finbert_frame["predicted_label"]) if not finbert_frame.empty else {},
            "chunk_type_mean_score": _chunk_means(finbert_frame),
        },
        "emotion_model": {
            "row_count": int(len(emotion_frame)),
            "predicted_label_counts": _counts(emotion_frame["predicted_label"]) if not emotion_frame.empty else {},
            "chunk_type_mean_score": _chunk_means(emotion_frame),
        },
        "deterministic_vs_finbert": {
            "comparable_rows": comparable_count,
            "disagreement_rows": disagreement_count,
            "agreement_rows": max(0, comparable_count - disagreement_count),
            "disagreement_rate": round(disagreement_count / comparable_count, 4) if comparable_count else 0.0,
            "by_chunk_type": by_chunk_type,
            "examples": disagreement_examples,
        },
    }

    if deterministic_out_dir is not None:
        target_dir = Path(deterministic_out_dir)
        guidance_path = target_dir / "guidance_revision.csv"
        qa_shift_path = target_dir / "qa_shift_summary.json"
        chunks_path = target_dir / "chunks_scored.csv"

        if chunks_path.exists():
            chunks_frame = pd.read_csv(chunks_path)
            label_column = "sentiment" if "sentiment" in chunks_frame.columns else "sentiment_label"
            if label_column in chunks_frame.columns:
                summary["deterministic_tone"] = {
                    "chunk_row_count": int(len(chunks_frame)),
                    "label_counts": _counts(chunks_frame[label_column]),
                }

        if guidance_path.exists():
            guidance_frame = pd.read_csv(guidance_path, keep_default_na=False)
            if "revision_label" in guidance_frame.columns:
                summary["deterministic_guidance"] = {
                    "row_count": int(len(guidance_frame)),
                    "revision_label_counts": _counts(guidance_frame["revision_label"]),
                }

        if qa_shift_path.exists():
            qa_payload = json.loads(qa_shift_path.read_text(encoding="utf-8"))
            summary["deterministic_q_and_a_context"] = {
                "prepared_remarks_vs_q_and_a": qa_payload.get("prepared_remarks_vs_q_and_a", {}),
                "analyst_skepticism": qa_payload.get("analyst_skepticism", {}),
                "management_answers_vs_prepared_uncertainty": qa_payload.get(
                    "management_answers_vs_prepared_uncertainty",
                    {},
                ),
            }

    return summary


def write_nlp_disagreement_summary(
    *,
    score_rows: list[dict[str, Any]],
    out_path: str | Path,
    deterministic_out_dir: str | Path | None = None,
) -> Path:
    target = Path(out_path)
    _write_json(
        target,
        build_nlp_disagreement_summary(
            score_rows=score_rows,
            deterministic_out_dir=deterministic_out_dir,
        ),
    )
    return target


def mean_score_by_chunk_type(score_rows: list[dict[str, Any]], *, model_role: str) -> dict[str, float]:
    frame = pd.DataFrame(score_rows)
    filtered = frame[frame["model_role"] == model_role].copy()
    if filtered.empty:
        return {}
    return {
        str(chunk_type): round(float(chunk_frame["predicted_score"].mean()), 4)
        for chunk_type, chunk_frame in filtered.groupby("chunk_type")
    }


def multimodal_nlp_defaults() -> dict[str, Any]:
    config = load_multimodal_config()
    return {
        "device": config.multimodal_device,
        "finbert_model": config.finbert_model,
        "emotion_model": config.emotion_model,
        "supported_chunk_types": sorted(SUPPORTED_CHUNK_TYPES),
    }


def segment_input_to_dicts(rows: list[NlpSegmentInput]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def mean_text_length(rows: list[NlpSegmentInput]) -> float:
    if not rows:
        return 0.0
    return round(float(mean(len(row.text) for row in rows)), 2)
