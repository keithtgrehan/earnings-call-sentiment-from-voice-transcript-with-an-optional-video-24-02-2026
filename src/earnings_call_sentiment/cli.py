"""Canonical CLI entry point for earnings_call_sentiment."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import __version__
from earnings_call_sentiment.downloaders.youtube import download_audio, download_video
from earnings_call_sentiment.post_summary import generate_optional_summary
from earnings_call_sentiment.pipeline.run import (
    load_transcript_segments,
    normalize_audio_to_wav,
    run_pipeline,
    transcribe_audio,
    write_sentiment_artifacts,
    write_transcript_artifacts,
)
import earnings_call_sentiment.question_shifts as qs
from earnings_call_sentiment.signals import write_behavioral_outputs, write_qa_shift_outputs
from earnings_call_sentiment.summary_config import (
    SummaryConfig,
    load_summary_config,
    run_summary_preflight,
)
from earnings_call_sentiment.visual import is_video_path, write_visual_behavior_outputs


_GUIDANCE_CUES = (
    "guidance",
    "outlook",
    "we expect",
    "we project",
    "we forecast",
    "range",
    "year",
    "quarter",
    "fy",
    "q1",
    "q2",
    "q3",
    "q4",
    "margin",
    "eps",
    "revenue",
    "operating income",
)

_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "top line"),
    "eps": ("eps", "earnings per share"),
    "margin": ("margin", "gross margin", "operating margin"),
    "outlook": ("outlook", "guidance", "forecast", "expect"),
    "opex": ("opex", "operating expense", "operating expenses"),
    "capex": ("capex", "capital expenditure", "capital expenditures"),
}

METRICS_SCHEMA_VERSION = "1.0.0"
DEFAULT_SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SENTIMENT_MODEL_REVISION = "714eb0f"


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[verbose] {message}")


def _signed_score(label: str, score: float) -> float:
    normalized = label.strip().upper()
    if "NEG" in normalized:
        return -abs(score)
    if "POS" in normalized:
        return abs(score)
    return 0.0


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes:02d}:{remainder:02d}"


def _file_ok(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _strict_required_artifacts(out_dir: Path) -> list[Path]:
    return [
        out_dir / "transcript.json",
        out_dir / "transcript.txt",
        out_dir / "sentiment_segments.csv",
        out_dir / "sentiment_timeline.png",
        out_dir / "risk_metrics.json",
        out_dir / "guidance.csv",
        out_dir / "guidance_revision.csv",
        out_dir / "tone_changes.csv",
        out_dir / "metrics.json",
        out_dir / "report.md",
        out_dir / "run_meta.json",
    ]


def _validate_strict_outputs(out_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in _strict_required_artifacts(out_dir):
        if not path.exists() or not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if path.stat().st_size <= 0:
            errors.append(f"empty: {path}")
    return errors


def _enforce_strict_outputs(out_dir: Path, console: Console) -> bool:
    errors = _validate_strict_outputs(out_dir)
    if not errors:
        console.print("[green]Strict output contract passed.[/green]")
        return True
    console.print("[bold red]Strict output contract failed.[/bold red]")
    for item in errors:
        console.print(f"- {item}")
    return False


def _stage_should_run(
    stage: str, outputs: list[Path], *, resume: bool, force: bool
) -> bool:
    if force or not resume:
        print(f"RUN stage {stage}")
        return True
    if outputs and all(_file_ok(path) for path in outputs):
        print(f"SKIP stage {stage}")
        return False
    print(f"RUN stage {stage}")
    return True


def _resolve_source_audio(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.audio_path:
        audio_path = Path(args.audio_path).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise RuntimeError(f"--audio-path not found: {audio_path}")
        return audio_path
    if args.youtube_url:
        return download_audio(
            youtube_url=args.youtube_url,
            cache_dir=cache_dir,
            audio_format=args.audio_format,
        )
    raise RuntimeError("Provide either --youtube-url or --audio-path.")


def _build_chunks_scored_df(sentiment_segments: list[dict[str, Any]]) -> pd.DataFrame:
    chunks_scored = pd.DataFrame(sentiment_segments)
    if chunks_scored.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "text",
                "sentiment",
                "score",
                "signed_score",
                "positive_prob",
                "negative_prob",
            ]
        )

    for column in ("start", "end", "score"):
        chunks_scored[column] = pd.to_numeric(
            chunks_scored[column], errors="coerce"
        ).fillna(0.0)

    labels = chunks_scored["sentiment"].astype(str).fillna("")
    signed_scores: list[float] = []
    positive_probs: list[float] = []
    negative_probs: list[float] = []
    for label, score in zip(labels.tolist(), chunks_scored["score"].tolist()):
        numeric_score = float(score)
        signed = _signed_score(label, numeric_score)
        signed_scores.append(signed)
        upper = str(label).upper()
        if "NEG" in upper:
            negative_probs.append(max(0.0, min(1.0, numeric_score)))
            positive_probs.append(max(0.0, min(1.0, 1.0 - numeric_score)))
        elif "POS" in upper:
            positive_probs.append(max(0.0, min(1.0, numeric_score)))
            negative_probs.append(max(0.0, min(1.0, 1.0 - numeric_score)))
        else:
            positive_probs.append(0.5)
            negative_probs.append(0.5)

    chunks_scored["signed_score"] = signed_scores
    chunks_scored["positive_prob"] = positive_probs
    chunks_scored["negative_prob"] = negative_probs
    return chunks_scored


def _write_chunks_scored_jsonl(chunks_scored: pd.DataFrame, out_dir: Path) -> Path:
    jsonl_path = out_dir / "chunks_scored.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in chunks_scored.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonl_path


def _run_question_shift_analysis(
    segments: list[dict[str, Any]],
    chunks_scored_jsonl: Path,
    out_dir: Path,
    args: argparse.Namespace,
    console: Console,
) -> None:
    if not chunks_scored_jsonl.exists() or not chunks_scored_jsonl.is_file():
        raise RuntimeError(
            "Question shift analysis requires scored chunks jsonl, but it was not found: "
            f"{chunks_scored_jsonl}"
        )
    chunks_scored = pd.read_json(chunks_scored_jsonl, lines=True)
    question_df = qs.detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=float(args.pre_window_s),
        after_window=float(args.post_window_s),
        min_gap_s=float(args.min_gap_s),
        min_chars=int(args.min_chars),
    )
    question_csv_path = out_dir / "question_sentiment_shifts.csv"
    question_plot_path = out_dir / "question_shifts.png"
    question_df.to_csv(question_csv_path, index=False)
    fig = qs.plot_question_shifts(question_df, out_path=question_plot_path)
    fig.clear()

    console.print()
    console.print("[bold]Question Shift Artifacts[/bold]")
    console.print(f"[bold]Question Shift CSV:[/bold] {question_csv_path}")
    console.print(f"[bold]Question Shift Plot:[/bold] {question_plot_path}")
    console.print("[bold]Top 10 Most Negative Shifts[/bold]")
    if question_df.empty:
        console.print("No qualifying analyst questions detected.")
        return

    top_negative = question_df.nsmallest(10, "sentiment_shift")
    for _, row in top_negative.iterrows():
        time_label = _format_mmss(float(row["question_time"]))
        shift = float(row["sentiment_shift"])
        question_text = str(row["question_text"]).replace("\n", " ").strip()
        if len(question_text) > 140:
            question_text = f"{question_text[:137]}..."
        console.print(f"{time_label} | shift={shift:+.4f} | {question_text}")


def _extract_guidance_df(chunks_scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "start",
        "end",
        "text",
        "sentiment",
        "score",
        "topic",
        "period",
        "numbers",
        "numeric_signature",
        "midpoint_hint",
        "guidance_strength",
        "count_numbers",
        "has_percent",
        "has_range",
        "has_currency",
        "matched_cues",
    ]
    if chunks_scored.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, row in chunks_scored.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        lowered = text.lower()
        cues = [cue for cue in _GUIDANCE_CUES if cue in lowered]
        if not cues:
            continue

        numbers = _extract_numbers(text)
        count_numbers = len(numbers)
        has_percent = bool(re.search(r"%|percent|bps|basis points", lowered))
        range_values = _extract_range_values(text)
        has_range = range_values is not None
        midpoint_hint = _extract_midpoint_value(text, numbers)
        has_currency = bool(
            re.search(r"\$|€|£|million|billion|bn\b|mm\b", lowered)
        )
        topic = _topic_tag(text)
        period = _period_tag(text)
        numeric_signature = _numeric_signature(numbers)
        label = str(row.get("sentiment", "")).upper()
        score = float(row.get("score", 0.0) or 0.0)
        signed = _signed_score(label, score)
        guidance_strength = min(
            1.0,
            (0.4 * min(1.0, count_numbers / 4.0))
            + (0.15 if has_percent else 0.0)
            + (0.15 if has_range else 0.0)
            + (0.1 if has_currency else 0.0)
            + (0.2 * max(0.0, signed))
            + min(0.1, 0.02 * len(cues)),
        )

        rows.append(
            {
                "start": float(row.get("start", 0.0)),
                "end": float(row.get("end", 0.0)),
                "text": text,
                "sentiment": str(row.get("sentiment", "")),
                "score": score,
                "topic": topic,
                "period": period,
                "numbers": ";".join(f"{value:.6g}" for value in numbers),
                "numeric_signature": numeric_signature,
                "midpoint_hint": midpoint_hint,
                "guidance_strength": round(guidance_strength, 4),
                "count_numbers": int(count_numbers),
                "has_percent": bool(has_percent),
                "has_range": bool(has_range),
                "has_currency": bool(has_currency),
                "matched_cues": ";".join(cues),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        "guidance_strength", ascending=False
    ).reset_index(drop=True)


def _round_to_sig(value: float, sig: int = 3) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.{sig}g}")


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        token = raw.replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _parse_numbers_column(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    text = str(value or "").strip()
    if not text:
        return []
    out: list[float] = []
    for token in re.split(r"[;|]", text):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _extract_range_values(text: str) -> tuple[float, float] | None:
    lowered = text.lower()
    between_match = re.search(
        r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", lowered
    )
    if between_match:
        first = float(between_match.group(1))
        second = float(between_match.group(2))
        return (min(first, second), max(first, second))

    to_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|–|-)\s*(-?\d+(?:\.\d+)?)", lowered)
    if to_match:
        first = float(to_match.group(1))
        second = float(to_match.group(2))
        return (min(first, second), max(first, second))
    return None


def _extract_midpoint_value(text: str, numbers: list[float]) -> float | None:
    range_values = _extract_range_values(text)
    if range_values is not None:
        return round((range_values[0] + range_values[1]) / 2.0, 6)
    if numbers:
        return round(float(numbers[0]), 6)
    return None


def _topic_tag(text: str) -> str:
    lowered = text.lower()
    best_topic = "other"
    best_count = 0
    for topic, patterns in _TOPIC_PATTERNS.items():
        count = sum(1 for pattern in patterns if pattern in lowered)
        if count > best_count:
            best_topic = topic
            best_count = count
    return best_topic


def _period_tag(text: str) -> str:
    lowered = text.lower()
    quarter_match = re.search(r"\bq([1-4])\b", lowered)
    if quarter_match:
        return f"Q{quarter_match.group(1)}"
    if re.search(r"\bfy\b|fiscal year|full year", lowered):
        return "FY"
    if "year" in lowered:
        return "Year"
    if "quarter" in lowered:
        return "Quarter"
    return "Unknown"


def _numeric_signature(numbers: list[float]) -> str:
    if not numbers:
        return "none"
    rounded = sorted({_round_to_sig(value) for value in numbers})
    return "|".join(f"{value:.6g}" for value in rounded)


def _jaccard_numbers(left: list[float], right: list[float]) -> float:
    left_tokens = {f"{_round_to_sig(value):.6g}" for value in left}
    right_tokens = {f"{_round_to_sig(value):.6g}" for value in right}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _guidance_row_features(row: pd.Series, index: int) -> dict[str, Any]:
    text = str(row.get("text", "")).strip()
    numbers = _parse_numbers_column(row.get("numbers", ""))
    if not numbers:
        numbers = _extract_numbers(text)
    topic = str(row.get("topic", "") or _topic_tag(text))
    period = str(row.get("period", "") or _period_tag(text))
    if period == "":
        period = "Unknown"
    midpoint = row.get("midpoint_hint", None)
    midpoint_value: float | None
    if midpoint is None or (isinstance(midpoint, float) and pd.isna(midpoint)):
        midpoint_value = _extract_midpoint_value(text, numbers)
    else:
        try:
            midpoint_value = float(midpoint)
        except (TypeError, ValueError):
            midpoint_value = _extract_midpoint_value(text, numbers)
    return {
        "idx": int(index),
        "start": float(row.get("start", 0.0)),
        "end": float(row.get("end", 0.0)),
        "text": text,
        "topic": topic,
        "period": period,
        "numbers": numbers,
        "numeric_signature": str(row.get("numeric_signature", "") or _numeric_signature(numbers)),
        "midpoint": midpoint_value,
        "match_key": f"{topic}|{period}|{_numeric_signature(numbers)}",
    }


def match_guidance(
    prior_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    overlap_threshold: float = 0.75,
    topic_match_bonus: float = 0.35,
    period_match_bonus: float = 0.2,
) -> pd.DataFrame:
    """Deterministically match current guidance rows to prior rows."""
    columns = [
        "current_idx",
        "prior_idx",
        "is_matched",
        "overlap_score",
        "topic",
        "period",
        "current_match_key",
        "prior_match_key",
    ]
    if current_df.empty:
        return pd.DataFrame(columns=columns)

    prior_rows = [
        _guidance_row_features(row, idx)
        for idx, row in prior_df.reset_index(drop=True).iterrows()
    ]
    current_rows = [
        _guidance_row_features(row, idx)
        for idx, row in current_df.reset_index(drop=True).iterrows()
    ]

    used_prior: set[int] = set()
    results: list[dict[str, Any]] = []
    for current in current_rows:
        best_score = -1.0
        best_prior: dict[str, Any] | None = None

        for prior in prior_rows:
            if prior["idx"] in used_prior:
                continue
            score = _jaccard_numbers(current["numbers"], prior["numbers"])
            if current["topic"] == prior["topic"]:
                score += topic_match_bonus if current["topic"] != "other" else (topic_match_bonus / 2.0)
            if current["period"] == prior["period"]:
                score += period_match_bonus if current["period"] != "Unknown" else (period_match_bonus / 2.0)
            if score > best_score:
                best_score = score
                best_prior = prior

        is_matched = best_prior is not None and best_score >= float(overlap_threshold)
        if is_matched and best_prior is not None:
            used_prior.add(int(best_prior["idx"]))

        results.append(
            {
                "current_idx": int(current["idx"]),
                "prior_idx": (int(best_prior["idx"]) if is_matched and best_prior is not None else pd.NA),
                "is_matched": bool(is_matched),
                "overlap_score": round(max(0.0, best_score), 6),
                "topic": current["topic"],
                "period": current["period"],
                "current_match_key": current["match_key"],
                "prior_match_key": (
                    str(best_prior["match_key"])
                    if is_matched and best_prior is not None
                    else ""
                ),
            }
        )

    return pd.DataFrame(results, columns=columns)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _classify_revision_label(
    *,
    prior_numbers: list[float],
    current_numbers: list[float],
    prior_midpoint: float | None,
    current_midpoint: float | None,
    epsilon: float,
) -> tuple[str, float | None]:
    if prior_midpoint is None or current_midpoint is None:
        return "unclear", None

    diff = float(current_midpoint) - float(prior_midpoint)
    pair_count = min(len(prior_numbers), len(current_numbers))
    if pair_count >= 2:
        pair_diffs = [current_numbers[i] - prior_numbers[i] for i in range(pair_count)]
        has_pos = any(item >= epsilon for item in pair_diffs)
        has_neg = any(item <= -epsilon for item in pair_diffs)
        if has_pos and has_neg:
            return "mixed", diff

    if diff >= epsilon:
        return "raised", diff
    if diff <= -epsilon:
        return "lowered", diff
    if abs(diff) < epsilon:
        return "reaffirmed", diff
    return "unclear", diff


def _compute_guidance_revision(
    current_guidance: pd.DataFrame,
    *,
    prior_guidance_path: Path | None,
    output_path: Path,
    overlap_threshold: float = 0.75,
    epsilon: float = 0.05,
) -> pd.DataFrame:
    columns = [
        "row_id",
        "is_matched",
        "revision_label",
        "topic",
        "period",
        "current_start",
        "current_end",
        "prior_start",
        "prior_end",
        "current_text_snippet",
        "prior_text_snippet",
        "current_numbers",
        "prior_numbers",
        "current_midpoint",
        "prior_midpoint",
        "diff",
        "overlap_score",
        "current_match_key",
        "prior_match_key",
    ]
    empty_df = pd.DataFrame(columns=columns)
    if prior_guidance_path is None or current_guidance.empty:
        empty_df.to_csv(output_path, index=False)
        return empty_df

    prior_path = prior_guidance_path.expanduser().resolve()
    if not prior_path.exists() or not prior_path.is_file():
        raise RuntimeError(f"--prior-guidance file not found: {prior_path}")

    prior_guidance = _read_csv_or_empty(prior_path)
    matches_df = match_guidance(
        prior_guidance,
        current_guidance,
        overlap_threshold=overlap_threshold,
    )

    rows: list[dict[str, Any]] = []
    for _, match_row in matches_df.iterrows():
        current_idx = int(match_row["current_idx"])
        current_raw = current_guidance.reset_index(drop=True).iloc[current_idx]
        current_features = _guidance_row_features(current_raw, current_idx)
        prior_idx_value = match_row.get("prior_idx", pd.NA)
        is_matched = bool(match_row.get("is_matched", False)) and pd.notna(prior_idx_value)

        prior_features: dict[str, Any] | None = None
        if is_matched:
            prior_idx = int(prior_idx_value)
            prior_raw = prior_guidance.reset_index(drop=True).iloc[prior_idx]
            prior_features = _guidance_row_features(prior_raw, prior_idx)

        if prior_features is None:
            prior_numbers: list[float] = []
            prior_midpoint: float | None = None
            prior_start = float("nan")
            prior_end = float("nan")
            prior_text = ""
            label = "unclear"
            diff: float | None = None
            prior_match_key = ""
        else:
            prior_numbers = prior_features["numbers"]
            prior_midpoint = prior_features["midpoint"]
            prior_start = float(prior_features["start"])
            prior_end = float(prior_features["end"])
            prior_text = prior_features["text"]
            label, diff = _classify_revision_label(
                prior_numbers=prior_numbers,
                current_numbers=current_features["numbers"],
                prior_midpoint=prior_midpoint,
                current_midpoint=current_features["midpoint"],
                epsilon=float(epsilon),
            )
            prior_match_key = str(prior_features["match_key"])

        rows.append(
            {
                "row_id": int(current_idx),
                "is_matched": bool(is_matched),
                "revision_label": label,
                "topic": current_features["topic"],
                "period": current_features["period"],
                "current_start": float(current_features["start"]),
                "current_end": float(current_features["end"]),
                "prior_start": prior_start,
                "prior_end": prior_end,
                "current_text_snippet": str(current_features["text"])[:220],
                "prior_text_snippet": str(prior_text)[:220],
                "current_numbers": ";".join(f"{value:.6g}" for value in current_features["numbers"]),
                "prior_numbers": ";".join(f"{value:.6g}" for value in prior_numbers),
                "current_midpoint": (
                    float(current_features["midpoint"])
                    if current_features["midpoint"] is not None
                    else float("nan")
                ),
                "prior_midpoint": (
                    float(prior_midpoint)
                    if prior_midpoint is not None
                    else float("nan")
                ),
                "diff": (float(diff) if diff is not None else float("nan")),
                "overlap_score": float(match_row.get("overlap_score", 0.0)),
                "current_match_key": str(match_row.get("current_match_key", "")),
                "prior_match_key": prior_match_key,
            }
        )

    revision_df = pd.DataFrame(rows, columns=columns)
    revision_df.to_csv(output_path, index=False)
    return revision_df


def _compute_tone_changes_df(
    chunks_scored: pd.DataFrame, *, threshold: float
) -> pd.DataFrame:
    columns = [
        "start",
        "end",
        "sentiment_score",
        "rolling_mean_5",
        "rolling_std_5",
        "tone_change_z",
        "is_change",
        "text",
    ]
    if chunks_scored.empty:
        return pd.DataFrame(columns=columns)

    data = chunks_scored.copy().sort_values("start").reset_index(drop=True)
    data["sentiment_score"] = pd.to_numeric(data["signed_score"], errors="coerce").fillna(0.0)
    data["rolling_mean_5"] = data["sentiment_score"].rolling(window=5, min_periods=1).mean()
    data["rolling_std_5"] = (
        data["sentiment_score"].rolling(window=5, min_periods=1).std(ddof=0).fillna(0.0)
    )
    denom = data["rolling_std_5"].where(data["rolling_std_5"] > 1e-9, other=1.0)
    data["tone_change_z"] = (data["sentiment_score"] - data["rolling_mean_5"]) / denom
    data.loc[data["rolling_std_5"] <= 1e-9, "tone_change_z"] = 0.0
    data["is_change"] = data["tone_change_z"].abs() >= abs(float(threshold))
    keep = [col for col in columns if col in data.columns]
    return data[keep].copy()


def _build_metrics_payload(
    *,
    chunks_scored: pd.DataFrame,
    guidance_df: pd.DataFrame,
    guidance_revision_df: pd.DataFrame,
    tone_changes_df: pd.DataFrame,
    behavioral_summary: dict[str, Any] | None = None,
    prior_guidance_path: str | None,
    sentiment_model: str,
    sentiment_revision: str,
) -> dict[str, Any]:
    sentiment_series = pd.to_numeric(
        chunks_scored.get("signed_score", pd.Series([], dtype="float64")),
        errors="coerce",
    ).dropna()
    mean_sentiment = float(sentiment_series.mean()) if not sentiment_series.empty else 0.0
    std_sentiment = float(sentiment_series.std(ddof=0)) if len(sentiment_series) > 1 else 0.0

    guidance_strength_series = pd.to_numeric(
        guidance_df.get("guidance_strength", pd.Series([], dtype="float64")),
        errors="coerce",
    ).dropna()
    if behavioral_summary is None:
        behavioral_summary = {
            "uncertainty_score_overall": {"score": 0, "level": "low"},
            "reassurance_score_management": {"score": 0, "level": "low"},
            "analyst_skepticism_score": {"score": 0, "level": "low"},
        }
    tone_change_count = (
        int(tone_changes_df["is_change"].astype(bool).sum())
        if "is_change" in tone_changes_df.columns
        else 0
    )

    payload: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "git_commit": _resolve_git_commit(),
        "package_version": _resolve_package_version(),
        "generated_at": datetime.now(UTC).isoformat(),
        "models": {
            "sentiment": {
                "model": str(sentiment_model),
                "revision": str(sentiment_revision),
            }
        },
        "num_chunks_scored": int(len(chunks_scored)),
        "sentiment_mean": round(mean_sentiment, 6),
        "sentiment_std": round(std_sentiment, 6),
        "guidance": {
            "row_count": int(len(guidance_df)),
            "mean_strength": (
                round(float(guidance_strength_series.mean()), 6)
                if not guidance_strength_series.empty
                else None
            ),
        },
        "tone_changes": {
            "row_count": int(len(tone_changes_df)),
            "change_count": tone_change_count,
        },
        "behavioral_signals": {
            "uncertainty": behavioral_summary.get("uncertainty_score_overall", {}),
            "reassurance": behavioral_summary.get("reassurance_score_management", {}),
            "analyst_skepticism": behavioral_summary.get("analyst_skepticism_score", {}),
        },
        "guidance_revision": {
            "prior_guidance_path": (str(prior_guidance_path) if prior_guidance_path else None),
            "matched_count": 0,
            "raised_count": 0,
            "lowered_count": 0,
            "reaffirmed_count": 0,
            "unclear_count": 0,
            "mixed_count": 0,
            "top_revisions": [],
        },
    }

    if not guidance_revision_df.empty:
        labels = guidance_revision_df.get("revision_label", pd.Series([], dtype="object")).astype(str)
        matched_mask = (
            guidance_revision_df.get("is_matched", pd.Series([], dtype="bool"))
            .astype(bool)
            .reindex(guidance_revision_df.index, fill_value=False)
        )
        diff_numeric = pd.to_numeric(
            guidance_revision_df.get("diff", pd.Series([], dtype="float64")),
            errors="coerce",
        )
        top_df = guidance_revision_df.copy()
        top_df["abs_diff"] = diff_numeric.abs()
        top_df = top_df[top_df["abs_diff"].notna()].sort_values("abs_diff", ascending=False)
        top_revisions = []
        for _, row in top_df.head(5).iterrows():
            top_revisions.append(
                {
                    "start": round(float(row.get("current_start", 0.0)), 3),
                    "end": round(float(row.get("current_end", 0.0)), 3),
                    "label": str(row.get("revision_label", "unclear")),
                    "diff": round(float(row.get("diff", 0.0)), 6),
                    "snippet": str(row.get("current_text_snippet", "")),
                }
            )

        payload["guidance_revision"] = {
            "prior_guidance_path": (str(prior_guidance_path) if prior_guidance_path else None),
            "matched_count": int(matched_mask.sum()),
            "raised_count": int((labels == "raised").sum()),
            "lowered_count": int((labels == "lowered").sum()),
            "reaffirmed_count": int((labels == "reaffirmed").sum()),
            "unclear_count": int((labels == "unclear").sum()),
            "mixed_count": int((labels == "mixed").sum()),
            "top_revisions": top_revisions,
        }
    return payload


def _resolve_visual_source_path(
    *,
    youtube_url: str | None,
    audio_path: str | Path | None,
    cache_dir: Path,
    verbose: bool = False,
) -> tuple[Path | None, str | None]:
    if audio_path is not None:
        candidate = Path(audio_path).expanduser().resolve()
        if is_video_path(candidate):
            return candidate, None
        return None, None
    if youtube_url:
        try:
            video_path = Path(download_video(youtube_url, cache_dir))
            return video_path, None
        except Exception as exc:
            _log(verbose, f"visual video download unavailable: {exc}")
            return None, f"Video download unavailable: {exc}"
    return None, None


def _write_report_markdown(
    *,
    output_path: Path,
    metrics_payload: dict[str, Any],
    guidance_df: pd.DataFrame,
    guidance_revision_df: pd.DataFrame,
    behavioral_summary: dict[str, Any],
    qa_shift_summary: dict[str, Any],
    visual_summary: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Earnings Call Sentiment Report",
        "",
        "## Summary",
        f"- Chunks scored: {metrics_payload.get('num_chunks_scored')}",
        f"- Sentiment mean: {metrics_payload.get('sentiment_mean')}",
        f"- Sentiment std: {metrics_payload.get('sentiment_std')}",
        "",
        "## Guidance",
        f"- Guidance rows: {metrics_payload.get('guidance', {}).get('row_count')}",
        f"- Mean guidance strength: {metrics_payload.get('guidance', {}).get('mean_strength')}",
        "",
    ]

    if not guidance_df.empty:
        lines.extend(
            [
                "| start | end | guidance_strength | text |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in guidance_df.head(5).iterrows():
            text = str(row.get("text", "")).replace("\n", " ").strip()
            if len(text) > 120:
                text = f"{text[:117]}..."
            lines.append(
                f"| {float(row.get('start', 0.0)):.2f} | {float(row.get('end', 0.0)):.2f} | "
                f"{float(row.get('guidance_strength', 0.0)):.4f} | {text} |"
            )
        lines.append("")

    guidance_revision = metrics_payload.get("guidance_revision")
    if isinstance(guidance_revision, dict):
        lines.extend(
            [
                "## Guidance Revisions (vs prior)",
                f"- Prior guidance: {guidance_revision.get('prior_guidance_path')}",
                f"- Matched: {guidance_revision.get('matched_count')}",
                f"- Raised: {guidance_revision.get('raised_count')}",
                f"- Lowered: {guidance_revision.get('lowered_count')}",
                f"- Reaffirmed: {guidance_revision.get('reaffirmed_count')}",
                f"- Unclear: {guidance_revision.get('unclear_count')}",
                f"- Mixed: {guidance_revision.get('mixed_count')}",
                "",
            ]
        )
        top_revisions = guidance_revision.get("top_revisions", [])
        if isinstance(top_revisions, list) and top_revisions:
            lines.extend(
                [
                    "| start | end | label | diff | snippet |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for row in top_revisions[:5]:
                snippet = str(row.get("snippet", "")).replace("\n", " ").strip()
                if len(snippet) > 120:
                    snippet = f"{snippet[:117]}..."
                lines.append(
                    f"| {float(row.get('start', 0.0)):.2f} | "
                    f"{float(row.get('end', 0.0)):.2f} | "
                    f"{str(row.get('label', 'unclear'))} | "
                    f"{float(row.get('diff', 0.0)):.4f} | {snippet} |"
                )
        else:
            lines.append("_none_")
        lines.append("")

    lines.extend(
        [
            "## Tone & Behavioral Signals",
            f"- uncertainty: {behavioral_summary.get('uncertainty_score_overall', {}).get('level', 'low')}",
            f"- reassurance: {behavioral_summary.get('reassurance_score_management', {}).get('level', 'low')}",
            f"- analyst skepticism: {behavioral_summary.get('analyst_skepticism_score', {}).get('level', 'low')}",
            "",
        ]
    )
    strongest = behavioral_summary.get("strongest_evidence", {})
    if isinstance(strongest, dict):
        for key, label in (
            ("uncertainty", "Uncertainty"),
            ("reassurance", "Management reassurance"),
            ("analyst_skepticism", "Analyst skepticism"),
        ):
            rows = strongest.get(key, [])
            lines.append(f"### {label} evidence")
            if isinstance(rows, list) and rows:
                for item in rows[:2]:
                    snippet = str(item.get("text", "")).replace("\n", " ").strip()
                    if len(snippet) > 140:
                        snippet = f"{snippet[:137]}..."
                    lines.append(
                        f"- [{item.get('matched_phrase')}] strength={item.get('strength')}: {snippet}"
                    )
            else:
                lines.append("- _none_")
            lines.append("")

    qa_strongest = qa_shift_summary.get("strongest_evidence", {}) if isinstance(qa_shift_summary, dict) else {}
    lines.extend(
        [
            "## Q&A Shift",
            (
                "- prepared remarks vs Q&A: "
                f"{qa_shift_summary.get('prepared_remarks_vs_q_and_a', {}).get('label', 'mixed')}"
            ),
            (
                "- analyst skepticism: "
                f"{qa_shift_summary.get('analyst_skepticism', {}).get('level', 'low')}"
            ),
            (
                "- management answers vs prepared remarks uncertainty: "
                f"{qa_shift_summary.get('management_answers_vs_prepared_uncertainty', {}).get('label', 'mixed')}"
            ),
            (
                "- early vs late Q&A: "
                f"{qa_shift_summary.get('early_vs_late_q_and_a', {}).get('label', 'mixed')}"
            ),
            "",
            "### Q&A examples",
        ]
    )
    pair_rows = qa_strongest.get("qa_pairs", []) if isinstance(qa_strongest, dict) else []
    if isinstance(pair_rows, list) and pair_rows:
        for item in pair_rows[:2]:
            question = str(item.get("question_text", "")).replace("\n", " ").strip()
            answer = str(item.get("answer_text", "")).replace("\n", " ").strip()
            if len(question) > 120:
                question = f"{question[:117]}..."
            if len(answer) > 120:
                answer = f"{answer[:117]}..."
            lines.append(
                f"- delta={float(item.get('answer_minus_question', 0.0)):+.4f} | "
                f"Q: {question} | A: {answer}"
            )
    else:
        lines.append("- _none_")
    lines.append("")

    if visual_summary is not None:
        face_visibility = visual_summary.get("face_visibility_overall", {}) if isinstance(visual_summary, dict) else {}
        prepared_stability = (
            visual_summary.get("prepared_baseline_visual_stability", {})
            if isinstance(visual_summary, dict)
            else {}
        )
        qa_visual_shift = visual_summary.get("qa_visual_shift_score", {}) if isinstance(visual_summary, dict) else {}
        lines.extend(
            [
                "## Visual Behavior Signals",
                f"- face visibility: {face_visibility.get('level', 'low')}",
                f"- prepared remarks visual stability: {prepared_stability.get('level', 'low')}",
                f"- Q&A visual shift: {qa_visual_shift.get('level', 'low')}",
                "",
            ]
        )
        if visual_summary.get("visual_features_available"):
            notable = visual_summary.get("most_visually_changed_segments", [])
            if isinstance(notable, list) and notable:
                top_item = notable[0]
                lines.append(
                    "- notable segment: "
                    f"{top_item.get('section', 'segment')} "
                    f"{float(top_item.get('start_time_s', 0.0)):.1f}s-"
                    f"{float(top_item.get('end_time_s', 0.0)):.1f}s | "
                    f"change={float(top_item.get('visual_change_score', 0.0)):.4f}"
                )
            low_conf = visual_summary.get("notable_low_confidence_segments", [])
            if isinstance(low_conf, list) and low_conf:
                lines.append(
                    "- caution: "
                    f"{str(low_conf[0].get('confidence_note', 'low face visibility reduces confidence'))}"
                )
        else:
            limitations = visual_summary.get("limitations", [])
            note = limitations[0] if isinstance(limitations, list) and limitations else "visual data unavailable"
            lines.append(f"- caution: {note}")
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "- guidance.csv",
            "- guidance_revision.csv",
            "- uncertainty_signals.csv",
            "- reassurance_signals.csv",
            "- analyst_skepticism.csv",
            "- behavioral_summary.json",
            "- qa_shift_segments.csv",
            "- qa_shift_summary.json",
        ]
    )
    if visual_summary is not None:
        lines.extend(
            [
                "- visual_behavior_frames.csv",
                "- visual_behavior_segments.csv",
                "- visual_behavior_summary.json",
            ]
        )
    lines.extend(
        [
            "- metrics.json",
            "- report.md",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_postscore_stages(
    *,
    chunks_scored_df: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
    video_path: Path | None = None,
    enable_visual: bool = False,
    visual_note: str | None = None,
) -> dict[str, Path]:
    resume = bool(args.resume)
    force = bool(args.force)
    guidance_path = out_dir / "guidance.csv"
    guidance_revision_path = out_dir / "guidance_revision.csv"
    tone_changes_path = out_dir / "tone_changes.csv"
    uncertainty_path = out_dir / "uncertainty_signals.csv"
    reassurance_path = out_dir / "reassurance_signals.csv"
    skepticism_path = out_dir / "analyst_skepticism.csv"
    behavioral_summary_path = out_dir / "behavioral_summary.json"
    qa_shift_segments_path = out_dir / "qa_shift_segments.csv"
    qa_shift_summary_path = out_dir / "qa_shift_summary.json"
    visual_frames_path = out_dir / "visual_behavior_frames.csv"
    visual_segments_path = out_dir / "visual_behavior_segments.csv"
    visual_summary_path = out_dir / "visual_behavior_summary.json"
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"

    if _stage_should_run("guidance", [guidance_path], resume=resume, force=force):
        guidance_df = _extract_guidance_df(chunks_scored_df)
        guidance_df.to_csv(guidance_path, index=False)
    else:
        guidance_df = _read_csv_or_empty(guidance_path)

    if _stage_should_run(
        "guidance_revision", [guidance_revision_path], resume=resume, force=force
    ):
        prior_guidance_path = (
            Path(args.prior_guidance).expanduser().resolve()
            if args.prior_guidance
            else None
        )
        guidance_revision_df = _compute_guidance_revision(
            guidance_df,
            prior_guidance_path=prior_guidance_path,
            output_path=guidance_revision_path,
        )
    else:
        guidance_revision_df = _read_csv_or_empty(guidance_revision_path)

    if _stage_should_run(
        "tone_changes", [tone_changes_path], resume=resume, force=force
    ):
        tone_changes_df = _compute_tone_changes_df(
            chunks_scored_df,
            threshold=float(args.tone_change_threshold),
        )
        tone_changes_df.to_csv(tone_changes_path, index=False)
    else:
        tone_changes_df = _read_csv_or_empty(tone_changes_path)

    if _stage_should_run(
        "behavioral_signals",
        [uncertainty_path, reassurance_path, skepticism_path, behavioral_summary_path],
        resume=resume,
        force=force,
    ):
        behavioral_outputs = write_behavioral_outputs(chunks_scored_df, out_dir)
        behavioral_summary = behavioral_outputs["summary"]
    else:
        behavioral_summary = json.loads(behavioral_summary_path.read_text(encoding="utf-8"))

    if _stage_should_run(
        "qa_shift",
        [qa_shift_segments_path, qa_shift_summary_path],
        resume=resume,
        force=force,
    ):
        qa_shift_outputs = write_qa_shift_outputs(chunks_scored_df, out_dir)
        qa_shift_summary = qa_shift_outputs["summary"]
    else:
        qa_shift_summary = json.loads(qa_shift_summary_path.read_text(encoding="utf-8"))
    qa_shift_segments_df = _read_csv_or_empty(qa_shift_segments_path)

    visual_summary: dict[str, Any] | None = None
    if enable_visual:
        if _stage_should_run(
            "visual_behavior",
            [visual_frames_path, visual_segments_path, visual_summary_path],
            resume=resume,
            force=force,
        ):
            visual_outputs = write_visual_behavior_outputs(
                video_path,
                qa_shift_segments_df,
                out_dir,
                sample_fps=1.0,
            )
            visual_summary = visual_outputs["summary"]
            if visual_note:
                limits = visual_summary.setdefault("limitations", [])
                if visual_note not in limits:
                    limits.append(visual_note)
                visual_summary_path.write_text(json.dumps(visual_summary, indent=2), encoding="utf-8")
        else:
            visual_summary = json.loads(visual_summary_path.read_text(encoding="utf-8"))

    if _stage_should_run("metrics", [metrics_path], resume=resume, force=force):
        metrics_payload = _build_metrics_payload(
            chunks_scored=chunks_scored_df,
            guidance_df=guidance_df,
            guidance_revision_df=guidance_revision_df,
            tone_changes_df=tone_changes_df,
            behavioral_summary=behavioral_summary,
            prior_guidance_path=args.prior_guidance,
            sentiment_model=str(args.sentiment_model),
            sentiment_revision=str(args.sentiment_revision),
        )
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    else:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    if _stage_should_run("report", [report_path], resume=resume, force=force):
        _write_report_markdown(
            output_path=report_path,
            metrics_payload=metrics_payload,
            guidance_df=guidance_df,
            guidance_revision_df=guidance_revision_df,
            behavioral_summary=behavioral_summary,
            qa_shift_summary=qa_shift_summary,
            visual_summary=visual_summary,
        )

    return {
        "guidance_csv": guidance_path,
        "guidance_revision_csv": guidance_revision_path,
        "tone_changes_csv": tone_changes_path,
        "uncertainty_signals_csv": uncertainty_path,
        "reassurance_signals_csv": reassurance_path,
        "analyst_skepticism_csv": skepticism_path,
        "behavioral_summary_json": behavioral_summary_path,
        "qa_shift_segments_csv": qa_shift_segments_path,
        "qa_shift_summary_json": qa_shift_summary_path,
        "metrics_json": metrics_path,
        "report_md": report_path,
        **(
            {
                "visual_behavior_frames_csv": visual_frames_path,
                "visual_behavior_segments_csv": visual_segments_path,
                "visual_behavior_summary_json": visual_summary_path,
            }
            if enable_visual
            else {}
        ),
    }


def _print_outputs(console: Console, title: str, rows: list[tuple[str, str]]) -> None:
    console.print(f"[bold green]{title}[/bold green]")
    console.print()
    console.print("[bold]Output Files[/bold]")
    for label, value in rows:
        console.print(f"[bold]{label}:[/bold] {value}")


def _normalize_event_dt(value: str | None) -> tuple[str, bool]:
    if value is None or not str(value).strip():
        return datetime.now().astimezone().isoformat(), True

    normalized = str(value).strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "--event-dt must be ISO8601, e.g. 2024-08-01T16:00:00 or 2024-08-01 16:00"
        ) from exc

    if parsed.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        if local_tz is None:
            local_tz = UTC
        parsed = parsed.replace(tzinfo=local_tz)

    return parsed.isoformat(), False


def _resolve_version_identifier() -> str:
    git_commit = _resolve_git_commit()
    if git_commit:
        return f"git:{git_commit}"
    return __version__


def _resolve_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    sha = (proc.stdout or "").strip()
    if proc.returncode == 0 and sha:
        return sha
    return None


def _resolve_package_version() -> str | None:
    try:
        return importlib_metadata.version("earnings-call-sentiment")
    except importlib_metadata.PackageNotFoundError:
        return None


def _write_run_meta(
    *,
    out_dir: Path,
    symbol: str,
    event_dt: str,
    source_url: str | None,
) -> Path:
    normalized_symbol = str(symbol or "").strip().upper() or "UNKNOWN"
    generated_at = datetime.now(UTC).isoformat()
    event_token = re.sub(r"[^0-9]", "", event_dt)[:14] or "event"
    run_token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{normalized_symbol}_{event_token}_{run_token}"
    payload = {
        "symbol": normalized_symbol,
        "event_dt": str(event_dt),
        "source_url": str(source_url or ""),
        "run_id": run_id,
        "generated_at": generated_at,
        "version": _resolve_version_identifier(),
    }
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _resolve_summary_config(args: argparse.Namespace) -> SummaryConfig:
    return load_summary_config(
        enabled=bool(args.llm_summary),
        provider=args.summary_provider,
        model=args.summary_model,
        base_url=args.summary_base_url,
        api_key_env=args.summary_api_key_env,
        timeout_s=float(args.summary_timeout_s),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="earnings-call-sentiment",
        description=(
            "Analyze earnings call sentiment from transcripts and optional video input."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--youtube-url",
        default=None,
        help="YouTube URL to process (required unless --audio-path is provided)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Local audio file path to process (skips YouTube download)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Ticker symbol used for run metadata and backtesting (default: UNKNOWN).",
    )
    parser.add_argument(
        "--event-dt",
        default=None,
        help="Event timestamp (ISO8601), e.g. 2024-08-01T16:00:00 or 2024-08-01 16:00.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory for downloaded/intermediate audio and model cache",
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs",
        help="Directory where transcript and output artifacts will be written",
    )
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=("wav", "mp3", "m4a"),
        help="Audio format for YouTube extraction (default: wav)",
    )
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument(
        "--device",
        default="auto",
        help="Whisper device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Whisper compute type (e.g. int8, float16)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Transcription chunk length in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate and print planned steps without running the pipeline",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep intermediate download/WAV artifacts.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        default=False,
        help="Download/extract audio then exit (no transcription).",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        default=False,
        help="Stop after transcript.json/transcript.txt are generated.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help="Generate sentiment/risk artifacts from transcript segments.",
    )
    parser.add_argument(
        "--question-shifts",
        action="store_true",
        default=False,
        help="Detect question-related sentiment shifts and write CSV/PNG outputs.",
    )
    parser.add_argument(
        "--pre-window-s",
        type=float,
        default=60.0,
        help="Seconds before each question used for baseline sentiment.",
    )
    parser.add_argument(
        "--post-window-s",
        type=float,
        default=120.0,
        help="Seconds after each question used for post-question sentiment.",
    )
    parser.add_argument(
        "--min-gap-s",
        type=float,
        default=30.0,
        help="Minimum seconds between detected questions.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=15,
        help="Minimum question-text length for question-shift detection.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="Enable VAD filtering during transcription.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force post-score stages to rerun even when outputs already exist.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume post-score stages from existing non-empty artifacts (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and rerun post-score stages.",
    )
    parser.add_argument(
        "--tone-change-threshold",
        type=float,
        default=2.0,
        help="Absolute z-score threshold used to flag tone changes.",
    )
    parser.add_argument(
        "--prior-guidance",
        default=None,
        help="Path to a prior guidance.csv for revision comparison.",
    )
    parser.add_argument(
        "--sentiment-model",
        default=DEFAULT_SENTIMENT_MODEL_NAME,
        help="Pinned HuggingFace model id for sentiment scoring.",
    )
    parser.add_argument(
        "--sentiment-revision",
        default=DEFAULT_SENTIMENT_MODEL_REVISION,
        help="Pinned HuggingFace model revision for sentiment scoring.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Enforce strict output artifact contract and exit code 2 on violations.",
    )
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        default=False,
        help="Run optional post-pipeline narrative summary stage.",
    )
    parser.add_argument(
        "--summary-provider",
        choices=("none", "openai_compatible"),
        default=None,
        help="Summary provider for optional narrative stage.",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model identifier used by optional summary provider.",
    )
    parser.add_argument(
        "--summary-base-url",
        default=None,
        help="Base URL for openai-compatible summary provider.",
    )
    parser.add_argument(
        "--summary-api-key-env",
        default=None,
        help="Environment variable name holding summary provider API key.",
    )
    parser.add_argument(
        "--summary-timeout-s",
        type=float,
        default=30.0,
        help="Timeout in seconds for optional summary provider call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    mode_flags = [args.download_only, args.transcribe_only, args.score_only]
    if sum(bool(flag) for flag in mode_flags) > 1:
        parser.error(
            "Use at most one of --download-only, --transcribe-only, --score-only"
        )

    if not args.youtube_url and not args.audio_path:
        parser.error("--youtube-url is required when --audio-path is not provided")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    console = Console()
    symbol = str(args.symbol or "").strip().upper() or "UNKNOWN"
    try:
        event_dt_iso, defaulted_event_dt = _normalize_event_dt(args.event_dt)
    except ValueError as exc:
        parser.error(str(exc))

    if args.verbose:
        _log(args.verbose, f"args={args}")

    if args.dry_run:
        summary_config = _resolve_summary_config(args)
        summary_ok, summary_message = run_summary_preflight(summary_config)
        print("Dry run enabled; skipping execution.")
        print(f"youtube_url={args.youtube_url}")
        print(f"audio_path={args.audio_path}")
        print(f"cache_dir={cache_dir}")
        print(f"out_dir={out_dir}")
        print(f"resume={args.resume}")
        print(f"force={args.force}")
        print(f"prior_guidance={args.prior_guidance}")
        print(f"sentiment_model={args.sentiment_model}")
        print(f"sentiment_revision={args.sentiment_revision}")
        print(f"symbol={symbol}")
        print(f"event_dt={event_dt_iso}")
        print(f"summary_enabled={summary_config.enabled}")
        print(f"summary_provider={summary_config.provider}")
        print(f"summary_model={summary_config.model}")
        print(f"summary_base_url={summary_config.base_url}")
        print(f"summary_api_key_env={summary_config.api_key_env}")
        print(f"summary_timeout_s={summary_config.timeout_s}")
        print(f"summary_preflight_ok={summary_ok}")
        print(f"summary_preflight_message={summary_message}")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_enabled = bool(args.youtube_url or (args.audio_path and is_video_path(args.audio_path)))
    visual_source_path: Path | None = None
    visual_note: str | None = None
    if visual_enabled:
        visual_source_path, visual_note = _resolve_visual_source_path(
            youtube_url=args.youtube_url,
            audio_path=args.audio_path,
            cache_dir=cache_dir,
            verbose=bool(args.verbose),
        )
    if defaulted_event_dt:
        console.print(
            "[yellow]Warning:[/yellow] --event-dt not provided; defaulting to current "
            "local time. Backtesting requires the true event timestamp."
        )

    if args.download_only:
        source_audio = _resolve_source_audio(args, cache_dir)
        print(f"Download complete: {source_audio}")
        return 0

    if args.transcribe_only:
        source_audio = _resolve_source_audio(args, cache_dir)
        _log(args.verbose, f"source_audio={source_audio}")
        normalized_wav = cache_dir / "audio_normalized.wav"
        normalize_audio_to_wav(source_audio, normalized_wav, verbose=args.verbose)
        segments = transcribe_audio(
            str(normalized_wav),
            verbose=args.verbose,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            chunk_seconds=float(args.chunk_seconds),
            vad=bool(args.vad),
        )
        transcript_json, transcript_txt = write_transcript_artifacts(segments, out_dir)
        _print_outputs(
            console,
            "Transcription Complete",
            [
                ("Audio", str(normalized_wav)),
                ("Transcript JSON", str(transcript_json)),
                ("Transcript Text", str(transcript_txt)),
            ],
        )
        if args.question_shifts:
            console.print(
                "[yellow]Skipping --question-shifts in --transcribe-only mode "
                "(no sentiment scoring step).[/yellow]"
            )
        run_meta_path = _write_run_meta(
            out_dir=out_dir,
            symbol=symbol,
            event_dt=event_dt_iso,
            source_url=args.youtube_url,
        )
        console.print(f"[bold]Run Metadata:[/bold] {run_meta_path}")
        return 0

    if args.score_only:
        transcript_json = out_dir / "transcript.json"
        transcript_txt = out_dir / "transcript.txt"
        if transcript_json.exists() and transcript_json.is_file():
            segments = load_transcript_segments(transcript_json)
        else:
            source_audio = _resolve_source_audio(args, cache_dir)
            _log(args.verbose, f"source_audio={source_audio}")
            normalized_wav = cache_dir / "audio_normalized.wav"
            normalize_audio_to_wav(source_audio, normalized_wav, verbose=args.verbose)
            segments = transcribe_audio(
                str(normalized_wav),
                verbose=args.verbose,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_seconds=float(args.chunk_seconds),
                vad=bool(args.vad),
            )
            transcript_json, transcript_txt = write_transcript_artifacts(
                segments, out_dir
            )
        if not transcript_txt.exists():
            transcript_txt.write_text(
                "\n".join(item.get("text", "") for item in segments if item.get("text")),
                encoding="utf-8",
            )

        artifacts = write_sentiment_artifacts(
            segments=segments,
            output_path=out_dir,
            sentiment_model=str(args.sentiment_model),
            sentiment_revision=str(args.sentiment_revision),
        )
        sentiment_segments = artifacts["sentiment_segments"]
        chunks_scored_df = _build_chunks_scored_df(sentiment_segments)
        chunks_scored_csv = out_dir / "chunks_scored.csv"
        chunks_scored_df.to_csv(chunks_scored_csv, index=False)
        chunks_scored_jsonl = _write_chunks_scored_jsonl(chunks_scored_df, out_dir)

        post_paths = _run_postscore_stages(
            chunks_scored_df=chunks_scored_df,
            out_dir=out_dir,
            args=args,
            video_path=visual_source_path,
            enable_visual=visual_enabled,
            visual_note=visual_note,
        )
        output_rows = [
            ("Transcript JSON", str(transcript_json)),
            ("Sentiment Segments", str(artifacts["sentiment_segments_csv"])),
            ("Sentiment Timeline", str(artifacts["sentiment_timeline_png"])),
            ("Risk Metrics", str(artifacts["risk_metrics_json"])),
            ("Guidance", str(post_paths["guidance_csv"])),
            ("Guidance Revision", str(post_paths["guidance_revision_csv"])),
            ("Uncertainty Signals", str(post_paths["uncertainty_signals_csv"])),
            ("Reassurance Signals", str(post_paths["reassurance_signals_csv"])),
            ("Analyst Skepticism", str(post_paths["analyst_skepticism_csv"])),
            ("Behavior Summary", str(post_paths["behavioral_summary_json"])),
        ]
        if "visual_behavior_summary_json" in post_paths:
            output_rows.extend(
                [
                    ("Visual Frames", str(post_paths["visual_behavior_frames_csv"])),
                    ("Visual Segments", str(post_paths["visual_behavior_segments_csv"])),
                    ("Visual Summary", str(post_paths["visual_behavior_summary_json"])),
                ]
            )
        output_rows.extend(
            [
                ("Metrics", str(post_paths["metrics_json"])),
                ("Report", str(post_paths["report_md"])),
            ]
        )
        _print_outputs(console, "Scoring Complete", output_rows)
        if args.question_shifts:
            _run_question_shift_analysis(
                segments=segments,
                chunks_scored_jsonl=chunks_scored_jsonl,
                out_dir=out_dir,
                args=args,
                console=console,
            )
        run_meta_path = _write_run_meta(
            out_dir=out_dir,
            symbol=symbol,
            event_dt=event_dt_iso,
            source_url=args.youtube_url,
        )
        console.print(f"[bold]Run Metadata:[/bold] {run_meta_path}")
        if args.llm_summary:
            summary_config = _resolve_summary_config(args)
            try:
                summary_path = generate_optional_summary(
                    out_dir=out_dir,
                    config=summary_config,
                    verbose=bool(args.verbose),
                )
                if summary_path is not None:
                    console.print(f"[bold]Optional Summary:[/bold] {summary_path}")
            except RuntimeError as exc:
                console.print(
                    "[yellow]Optional summary stage failed:[/yellow] "
                    f"{exc}"
                )
        if args.strict and not _enforce_strict_outputs(out_dir, console):
            return 2
        return 0

    resolved_audio_path = None
    resolved_youtube_url = args.youtube_url
    if args.audio_path:
        resolved_audio = Path(args.audio_path).expanduser().resolve()
        if not resolved_audio.exists() or not resolved_audio.is_file():
            parser.error(f"--audio-path not found: {resolved_audio}")
        resolved_audio_path = str(resolved_audio)
        resolved_youtube_url = None

    result = run_pipeline(
        youtube_url=resolved_youtube_url,
        audio_path=resolved_audio_path,
        cache_dir=str(cache_dir),
        out_dir=str(out_dir),
        verbose=bool(args.verbose),
        audio_format=args.audio_format,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        chunk_seconds=float(args.chunk_seconds),
        vad=bool(args.vad),
        sentiment_model=str(args.sentiment_model),
        sentiment_revision=str(args.sentiment_revision),
    )

    sentiment_segments_path = Path(str(result["sentiment_segments_csv"]))
    sentiment_df = pd.read_csv(sentiment_segments_path)
    sentiment_records = sentiment_df.to_dict(orient="records")
    chunks_scored_df = _build_chunks_scored_df(sentiment_records)
    chunks_scored_csv = out_dir / "chunks_scored.csv"
    chunks_scored_df.to_csv(chunks_scored_csv, index=False)
    chunks_scored_jsonl = _write_chunks_scored_jsonl(chunks_scored_df, out_dir)

    post_paths = _run_postscore_stages(
        chunks_scored_df=chunks_scored_df,
        out_dir=out_dir,
        args=args,
        video_path=visual_source_path,
        enable_visual=visual_enabled,
        visual_note=visual_note,
    )

    output_rows = [
        ("Audio", str(result["audio"])),
        ("Transcript JSON", str(result["transcript_json"])),
        ("Transcript Text", str(result["transcript_txt"])),
        ("Sentiment Segments", str(result["sentiment_segments_csv"])),
        ("Sentiment Timeline", str(result["sentiment_timeline_png"])),
        ("Risk Metrics", str(result["risk_metrics_json"])),
        ("Guidance", str(post_paths["guidance_csv"])),
        ("Guidance Revision", str(post_paths["guidance_revision_csv"])),
        ("Uncertainty Signals", str(post_paths["uncertainty_signals_csv"])),
        ("Reassurance Signals", str(post_paths["reassurance_signals_csv"])),
        ("Analyst Skepticism", str(post_paths["analyst_skepticism_csv"])),
        ("Behavior Summary", str(post_paths["behavioral_summary_json"])),
    ]
    if "visual_behavior_summary_json" in post_paths:
        output_rows.extend(
            [
                ("Visual Frames", str(post_paths["visual_behavior_frames_csv"])),
                ("Visual Segments", str(post_paths["visual_behavior_segments_csv"])),
                ("Visual Summary", str(post_paths["visual_behavior_summary_json"])),
            ]
        )
    output_rows.extend(
        [
            ("Metrics", str(post_paths["metrics_json"])),
            ("Report", str(post_paths["report_md"])),
        ]
    )
    _print_outputs(console, "Earnings Call Analysis Complete", output_rows)

    if args.question_shifts:
        segments = load_transcript_segments(Path(str(result["transcript_json"])))
        _run_question_shift_analysis(
            segments=segments,
            chunks_scored_jsonl=chunks_scored_jsonl,
            out_dir=out_dir,
            args=args,
            console=console,
        )
    run_meta_path = _write_run_meta(
        out_dir=out_dir,
        symbol=symbol,
        event_dt=event_dt_iso,
        source_url=args.youtube_url,
    )
    console.print(f"[bold]Run Metadata:[/bold] {run_meta_path}")
    if args.llm_summary:
        summary_config = _resolve_summary_config(args)
        try:
            summary_path = generate_optional_summary(
                out_dir=out_dir,
                config=summary_config,
                verbose=bool(args.verbose),
            )
            if summary_path is not None:
                console.print(f"[bold]Optional Summary:[/bold] {summary_path}")
        except RuntimeError as exc:
            console.print(
                "[yellow]Optional summary stage failed:[/yellow] "
                f"{exc}"
            )
    if args.strict and not _enforce_strict_outputs(out_dir, console):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
