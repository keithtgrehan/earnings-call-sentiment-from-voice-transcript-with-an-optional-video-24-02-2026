"""Deterministic tone and behavior signal extraction."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .rule_tables import (
    OPERATOR_PATTERNS,
    QUESTION_PATTERNS,
    REASSURANCE_RULES,
    SKEPTICISM_RULES,
    TOPIC_PATTERNS,
    UNCERTAINTY_RULES,
)

SUMMARY_SCHEMA_VERSION = "1.0.0"


def _iter_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.S):
        start, end = match.span()
        sentence = text[start:end].strip()
        if sentence:
            sentence_start = text.find(sentence, start, end)
            spans.append((sentence, sentence_start, sentence_start + len(sentence)))
    if spans:
        return spans
    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    return [(stripped, start, start + len(stripped))]


def _looks_like_operator(text: str) -> bool:
    lowered = text.strip().lower()
    return any(re.search(pattern, lowered) for pattern in OPERATOR_PATTERNS)


def _looks_like_question(text: str) -> bool:
    lowered = text.strip().lower()
    return any(re.search(pattern, lowered) for pattern in QUESTION_PATTERNS)


def _classify_context(text: str) -> tuple[str, str, str]:
    if _looks_like_operator(text):
        return "operator", "operator", ""
    if _looks_like_question(text):
        return "q_and_a", "analyst", "unknown"
    return "management", "management", ""


def _topic_hint(text: str) -> str:
    lowered = text.lower()
    best_topic = "general"
    best_count = 0
    for topic, keywords in TOPIC_PATTERNS.items():
        count = sum(1 for keyword in keywords if keyword in lowered)
        if count > best_count:
            best_topic = topic
            best_count = count
    return best_topic


def _level_from_score(score: int) -> str:
    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _strongest_items(rows: list[dict[str, Any]], *, text_key: str = "text") -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: (-int(item.get("strength", 0)), str(item.get(text_key, ""))))
    snippets: list[dict[str, Any]] = []
    for row in ordered[:2]:
        snippets.append(
            {
                "matched_phrase": str(row.get("matched_phrase", "")),
                "strength": int(row.get("strength", 0)),
                "text": str(row.get(text_key, "")),
            }
        )
    return snippets


def compute_behavioral_outputs(chunks_scored: pd.DataFrame) -> dict[str, Any]:
    uncertainty_rows: list[dict[str, Any]] = []
    reassurance_rows: list[dict[str, Any]] = []
    skepticism_rows: list[dict[str, Any]] = []
    question_id = 0

    if chunks_scored.empty:
        uncertainty_df = pd.DataFrame(
            columns=[
                "section",
                "speaker_role",
                "text",
                "matched_phrase",
                "signal_type",
                "strength",
                "start_char",
                "end_char",
                "notes",
            ]
        )
        reassurance_df = pd.DataFrame(
            columns=[
                "section",
                "speaker_role",
                "text",
                "matched_phrase",
                "strength",
                "topic_hint",
                "start_char",
                "end_char",
                "notes",
            ]
        )
        skepticism_df = pd.DataFrame(
            columns=[
                "question_id",
                "analyst_name",
                "text",
                "skepticism_label",
                "matched_phrase",
                "strength",
                "topic_hint",
                "notes",
            ]
        )
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "uncertainty_score_overall": {"score": 0, "level": "low"},
            "reassurance_score_management": {"score": 0, "level": "low"},
            "analyst_skepticism_score": {"score": 0, "level": "low"},
            "strongest_evidence": {"uncertainty": [], "reassurance": [], "analyst_skepticism": []},
            "notes": [
                "No behavioral matches were extracted from the current transcript chunks.",
                "Scores are deterministic phrase-weight totals, not model probabilities.",
            ],
        }
        return {
            "uncertainty_df": uncertainty_df,
            "reassurance_df": reassurance_df,
            "skepticism_df": skepticism_df,
            "summary": summary,
        }

    work = chunks_scored.copy()
    if "start" not in work.columns:
        work["start"] = 0.0
    if "end" not in work.columns:
        work["end"] = 0.0

    for _, row in work.sort_values("start").iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        section, speaker_role, analyst_name = _classify_context(text)
        sentence_spans = _iter_sentence_spans(text)

        if speaker_role == "management":
            for sentence, sent_start, _ in sentence_spans:
                lowered = sentence.lower()
                for rule in UNCERTAINTY_RULES:
                    for match in re.finditer(str(rule["pattern"]), lowered):
                        start_char, end_char = match.span()
                        uncertainty_rows.append(
                            {
                                "section": section,
                                "speaker_role": speaker_role,
                                "text": sentence,
                                "matched_phrase": str(rule["matched_phrase"]),
                                "signal_type": str(rule["signal_type"]),
                                "strength": int(rule["strength"]),
                                "start_char": int(start_char),
                                "end_char": int(end_char),
                                "notes": str(rule["notes"]),
                                "segment_start": float(row.get("start", 0.0)),
                                "segment_end": float(row.get("end", 0.0)),
                                "sentence_start": int(sent_start),
                            }
                        )
                for rule in REASSURANCE_RULES:
                    for match in re.finditer(str(rule["pattern"]), lowered):
                        start_char, end_char = match.span()
                        reassurance_rows.append(
                            {
                                "section": section,
                                "speaker_role": speaker_role,
                                "text": sentence,
                                "matched_phrase": str(rule["matched_phrase"]),
                                "strength": int(rule["strength"]),
                                "topic_hint": str(rule["topic_hint"]),
                                "start_char": int(start_char),
                                "end_char": int(end_char),
                                "notes": str(rule["notes"]),
                                "segment_start": float(row.get("start", 0.0)),
                                "segment_end": float(row.get("end", 0.0)),
                                "sentence_start": int(sent_start),
                            }
                        )

        if speaker_role == "analyst":
            question_id += 1
            question_matches: list[dict[str, Any]] = []
            lowered = text.lower()
            for rule in SKEPTICISM_RULES:
                for match in re.finditer(str(rule["pattern"]), lowered):
                    question_matches.append(
                        {
                            "matched_phrase": str(rule["matched_phrase"]),
                            "strength": int(rule["strength"]),
                            "topic_hint": str(rule["topic_hint"]),
                            "notes": str(rule["notes"]),
                        }
                    )
            if question_matches:
                total_strength = sum(int(item["strength"]) for item in question_matches)
                strongest = max(question_matches, key=lambda item: int(item["strength"]))
                skepticism_rows.append(
                    {
                        "question_id": int(question_id),
                        "analyst_name": analyst_name,
                        "text": text,
                        "skepticism_label": _level_from_score(total_strength),
                        "matched_phrase": str(strongest["matched_phrase"]),
                        "strength": int(total_strength),
                        "topic_hint": _topic_hint(text) if strongest["topic_hint"] == "general" else str(strongest["topic_hint"]),
                        "notes": (
                            f"{len(question_matches)} skeptical cue(s) matched. "
                            f"Strongest cue: {strongest['notes']}"
                        ),
                    }
                )

    uncertainty_df = pd.DataFrame(uncertainty_rows)
    reassurance_df = pd.DataFrame(reassurance_rows)
    skepticism_df = pd.DataFrame(skepticism_rows)

    uncertainty_score = int(uncertainty_df.get("strength", pd.Series(dtype="int64")).fillna(0).sum())
    reassurance_score = int(reassurance_df.get("strength", pd.Series(dtype="int64")).fillna(0).sum())
    skepticism_score = int(skepticism_df.get("strength", pd.Series(dtype="int64")).fillna(0).sum())

    notes = [
        "Scores are deterministic phrase-weight totals, not model probabilities.",
        "Analyst skepticism is only computed on question-like segments; management reassurance ignores question-like text.",
    ]
    if uncertainty_score > 0 and reassurance_score > 0:
        notes.append(
            "Management language contains both caution and reassurance, so both evidence sets should be reviewed together."
        )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "uncertainty_score_overall": {
            "score": uncertainty_score,
            "level": _level_from_score(uncertainty_score),
        },
        "reassurance_score_management": {
            "score": reassurance_score,
            "level": _level_from_score(reassurance_score),
        },
        "analyst_skepticism_score": {
            "score": skepticism_score,
            "level": _level_from_score(skepticism_score),
        },
        "strongest_evidence": {
            "uncertainty": _strongest_items(uncertainty_rows),
            "reassurance": _strongest_items(reassurance_rows),
            "analyst_skepticism": _strongest_items(skepticism_rows),
        },
        "notes": notes,
    }

    return {
        "uncertainty_df": uncertainty_df,
        "reassurance_df": reassurance_df,
        "skepticism_df": skepticism_df,
        "summary": summary,
    }


def write_behavioral_outputs(chunks_scored: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    payload = compute_behavioral_outputs(chunks_scored)
    out_dir.mkdir(parents=True, exist_ok=True)

    uncertainty_path = out_dir / "uncertainty_signals.csv"
    reassurance_path = out_dir / "reassurance_signals.csv"
    skepticism_path = out_dir / "analyst_skepticism.csv"
    summary_path = out_dir / "behavioral_summary.json"

    payload["uncertainty_df"].to_csv(uncertainty_path, index=False)
    payload["reassurance_df"].to_csv(reassurance_path, index=False)
    payload["skepticism_df"].to_csv(skepticism_path, index=False)
    summary_path.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")

    return {
        **payload,
        "uncertainty_path": uncertainty_path,
        "reassurance_path": reassurance_path,
        "skepticism_path": skepticism_path,
        "summary_path": summary_path,
    }
