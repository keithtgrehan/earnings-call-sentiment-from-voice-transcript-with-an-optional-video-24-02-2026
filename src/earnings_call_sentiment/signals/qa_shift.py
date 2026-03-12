"""Deterministic Q&A shift summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .behavior import compute_behavioral_outputs, _classify_context

QA_SHIFT_SCHEMA_VERSION = "1.0.0"


def _resolve_signed_score(row: pd.Series) -> float:
    for key in ("signed_score", "signed_sentiment", "signed", "sentiment_signed", "score"):
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return 0.0


def _shift_label(delta: float, *, threshold: float = 0.08) -> str:
    if delta >= threshold:
        return "stronger"
    if delta <= -threshold:
        return "weaker"
    return "mixed"


def _uncertainty_delta_label(prepared_score: int, answer_score: int) -> str:
    delta = int(answer_score) - int(prepared_score)
    if delta >= 2:
        return "more uncertain"
    if delta <= -2:
        return "less uncertain"
    return "mixed"


def _strongest_pair_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: (abs(float(item["answer_minus_question"])), float(item["question_start"])), reverse=True)
    examples: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in ordered:
        pair_id = int(row["qa_pair_id"])
        if pair_id in seen:
            continue
        seen.add(pair_id)
        examples.append(
            {
                "qa_pair_id": pair_id,
                "question_text": str(row["question_text"]),
                "answer_text": str(row["answer_text"]),
                "answer_minus_question": round(float(row["answer_minus_question"]), 4),
            }
        )
        if len(examples) >= 2:
            break
    return examples


def compute_qa_shift_outputs(chunks_scored: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "segment_id",
        "start",
        "end",
        "phase",
        "speaker_role",
        "qa_pair_id",
        "signed_score",
        "question_text",
        "text",
    ]
    if chunks_scored.empty:
        segments_df = pd.DataFrame(columns=columns)
        summary = {
            "schema_version": QA_SHIFT_SCHEMA_VERSION,
            "status": "no_segments",
            "prepared_remarks_vs_q_and_a": {"label": "mixed", "delta": 0.0},
            "analyst_skepticism": {"level": "low", "score": 0},
            "management_answers_vs_prepared_uncertainty": {"label": "mixed", "delta": 0},
            "early_vs_late_q_and_a": {"label": "mixed", "delta": 0.0},
            "strongest_evidence": {"skepticism": [], "answer_uncertainty": [], "qa_pairs": []},
            "notes": ["No scored segments were available for Q&A shift analysis."],
        }
        return {"segments_df": segments_df, "summary": summary}

    work = chunks_scored.copy().sort_values("start").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    qa_pair_id = 0
    current_question_text = ""
    in_q_and_a = False

    for idx, row in work.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        section, speaker_role, _ = _classify_context(text)
        if section == "q_and_a" and speaker_role == "analyst":
            in_q_and_a = True
            qa_pair_id += 1
            current_question_text = text
        phase = "q_and_a" if in_q_and_a else "prepared_remarks"
        rows.append(
            {
                "segment_id": int(idx),
                "start": float(row.get("start", 0.0)),
                "end": float(row.get("end", 0.0)),
                "phase": phase,
                "speaker_role": speaker_role,
                "qa_pair_id": int(qa_pair_id) if phase == "q_and_a" else 0,
                "signed_score": _resolve_signed_score(row),
                "question_text": current_question_text if phase == "q_and_a" else "",
                "text": text,
            }
        )

    segments_df = pd.DataFrame(rows, columns=columns)
    prepared_df = segments_df[segments_df["phase"] == "prepared_remarks"].copy()
    qa_df = segments_df[segments_df["phase"] == "q_and_a"].copy()
    question_df = qa_df[qa_df["speaker_role"] == "analyst"].copy()
    answer_df = qa_df[qa_df["speaker_role"] == "management"].copy()

    prepared_mean = float(prepared_df["signed_score"].mean()) if not prepared_df.empty else 0.0
    qa_mean = float(qa_df["signed_score"].mean()) if not qa_df.empty else 0.0
    prepared_vs_qa_delta = qa_mean - prepared_mean

    prepared_behavior = compute_behavioral_outputs(
        prepared_df[["start", "end", "text", "signed_score"]].rename(columns={"signed_score": "score"})
    )
    answer_behavior = compute_behavioral_outputs(
        answer_df[["start", "end", "text", "signed_score"]].rename(columns={"signed_score": "score"})
    )
    qa_behavior = compute_behavioral_outputs(
        qa_df[["start", "end", "text", "signed_score"]].rename(columns={"signed_score": "score"})
    )

    pair_rows: list[dict[str, Any]] = []
    for pair_id in sorted(int(value) for value in answer_df["qa_pair_id"].dropna().unique() if int(value) > 0):
        pair_answers = answer_df[answer_df["qa_pair_id"] == pair_id]
        pair_questions = question_df[question_df["qa_pair_id"] == pair_id]
        if pair_answers.empty or pair_questions.empty:
            continue
        question_row = pair_questions.iloc[0]
        pair_rows.append(
            {
                "qa_pair_id": pair_id,
                "question_start": float(question_row["start"]),
                "question_text": str(question_row["text"]),
                "answer_text": str(pair_answers.iloc[0]["text"]),
                "answer_minus_question": float(pair_answers["signed_score"].mean()) - float(question_row["signed_score"]),
            }
        )

    early_vs_late = {"label": "mixed", "delta": 0.0}
    if len(answer_df) >= 2:
        midpoint = max(1, len(answer_df) // 2)
        early_mean = float(answer_df.iloc[:midpoint]["signed_score"].mean())
        late_mean = float(answer_df.iloc[midpoint:]["signed_score"].mean())
        early_vs_late = {
            "label": _shift_label(late_mean - early_mean, threshold=0.06),
            "delta": round(late_mean - early_mean, 4),
        }

    prepared_uncertainty = int(
        prepared_behavior["summary"].get("uncertainty_score_overall", {}).get("score", 0)
    )
    answer_uncertainty = int(
        answer_behavior["summary"].get("uncertainty_score_overall", {}).get("score", 0)
    )
    skepticism_summary = qa_behavior["summary"].get("analyst_skepticism_score", {})

    notes = [
        "Prepared remarks vs Q&A is a simple signed-score mean comparison over deterministic transcript chunks.",
        "Management answer uncertainty compares deterministic hedging totals in prepared remarks versus Q&A answers.",
    ]
    if question_df.empty or answer_df.empty:
        notes.append("Q&A pairing is sparse, so pair-level examples may be limited.")

    summary = {
        "schema_version": QA_SHIFT_SCHEMA_VERSION,
        "status": "ok",
        "prepared_remarks_vs_q_and_a": {
            "label": _shift_label(prepared_vs_qa_delta),
            "delta": round(prepared_vs_qa_delta, 4),
            "prepared_mean": round(prepared_mean, 4),
            "q_and_a_mean": round(qa_mean, 4),
        },
        "analyst_skepticism": {
            "level": str(skepticism_summary.get("level", "low")),
            "score": int(skepticism_summary.get("score", 0)),
        },
        "management_answers_vs_prepared_uncertainty": {
            "label": _uncertainty_delta_label(prepared_uncertainty, answer_uncertainty),
            "delta": int(answer_uncertainty - prepared_uncertainty),
            "prepared_score": prepared_uncertainty,
            "q_and_a_answer_score": answer_uncertainty,
        },
        "early_vs_late_q_and_a": early_vs_late,
        "strongest_evidence": {
            "skepticism": qa_behavior["summary"].get("strongest_evidence", {}).get("analyst_skepticism", []),
            "answer_uncertainty": answer_behavior["summary"].get("strongest_evidence", {}).get("uncertainty", []),
            "qa_pairs": _strongest_pair_examples(pair_rows),
        },
        "counts": {
            "prepared_segments": int(len(prepared_df)),
            "q_and_a_segments": int(len(qa_df)),
            "analyst_questions": int(len(question_df)),
            "management_answers": int(len(answer_df)),
        },
        "notes": notes,
    }
    return {"segments_df": segments_df, "summary": summary}


def write_qa_shift_outputs(chunks_scored: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    payload = compute_qa_shift_outputs(chunks_scored)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_path = out_dir / "qa_shift_segments.csv"
    summary_path = out_dir / "qa_shift_summary.json"
    payload["segments_df"].to_csv(segments_path, index=False)
    summary_path.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    return {
        "segments_df": payload["segments_df"],
        "summary": payload["summary"],
        "segments_path": segments_path,
        "summary_path": summary_path,
    }
