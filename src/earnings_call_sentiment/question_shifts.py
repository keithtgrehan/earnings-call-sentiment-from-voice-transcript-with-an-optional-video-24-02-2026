"""Question-shift analysis helpers.

Detects analyst-style questions and measures sentiment shifts around them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

QUESTION_PHRASES = (
    "question",
    "ask",
    "could you",
    "can you",
    "what about",
)


def _is_question_text(text: str, min_chars: int = 15) -> bool:
    normalized = text.strip().lower()
    if len(normalized) <= max(0, min_chars):
        return False
    if "?" in normalized:
        return True
    return any(phrase in normalized for phrase in QUESTION_PHRASES)


def _resolve_signed_column(chunks_scored: pd.DataFrame) -> str:
    candidates = (
        "signed_score",
        "signed_sentiment",
        "signed",
        "sentiment_signed",
        "score",
    )
    for column in candidates:
        if column in chunks_scored.columns:
            return column
    raise ValueError(
        "chunks_scored must include a signed sentiment column. "
        f"Tried: {', '.join(candidates)}"
    )


def _window_mean(
    chunks_scored: pd.DataFrame,
    score_column: str,
    window_start: float,
    window_end: float,
) -> float:
    if window_end <= window_start:
        return float("nan")
    overlap_mask = (chunks_scored["end"] > window_start) & (
        chunks_scored["start"] < window_end
    )
    window = chunks_scored.loc[overlap_mask, score_column]
    if window.empty:
        return float("nan")
    return float(window.mean())


def detect_question_shifts(
    segments: list[dict[str, Any]],
    chunks_scored: pd.DataFrame,
    before_window: float = 60.0,
    after_window: float = 120.0,
    min_gap_s: float = 30.0,
    min_chars: int = 15,
) -> pd.DataFrame:
    """Build a dataframe of sentiment shifts around detected analyst questions."""
    if "start" not in chunks_scored.columns or "end" not in chunks_scored.columns:
        raise ValueError("chunks_scored must include 'start' and 'end' columns.")

    score_column = _resolve_signed_column(chunks_scored)
    rows: list[dict[str, Any]] = []
    last_question_time: float | None = None

    sorted_segments = sorted(segments, key=lambda item: float(item.get("start", 0.0)))
    for segment in sorted_segments:
        text = str(segment.get("text", "")).strip()
        if not _is_question_text(text, min_chars=min_chars):
            continue

        question_start = float(segment.get("start", 0.0))
        if last_question_time is not None and question_start - last_question_time < max(
            0.0, min_gap_s
        ):
            continue

        sentiment_before = _window_mean(
            chunks_scored=chunks_scored,
            score_column=score_column,
            window_start=question_start - before_window,
            window_end=question_start,
        )
        sentiment_after = _window_mean(
            chunks_scored=chunks_scored,
            score_column=score_column,
            window_start=question_start,
            window_end=question_start + after_window,
        )
        sentiment_shift = sentiment_after - sentiment_before

        rows.append(
            {
                "question_time": question_start,
                "question_text": text,
                "sentiment_before": sentiment_before,
                "sentiment_after": sentiment_after,
                "sentiment_shift": sentiment_shift,
            }
        )
        last_question_time = question_start

    if not rows:
        return pd.DataFrame(
            columns=[
                "question_time",
                "question_text",
                "sentiment_before",
                "sentiment_after",
                "sentiment_shift",
            ]
        )

    return pd.DataFrame(rows).sort_values("question_time").reset_index(drop=True)


def analyze_question_shifts(
    segments: list[dict[str, Any]],
    chunks_scored: pd.DataFrame,
    before_window: float = 60.0,
    after_window: float = 120.0,
    min_gap_s: float = 30.0,
    min_chars: int = 15,
) -> pd.DataFrame:
    """Compatibility alias for detect_question_shifts."""
    return detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=before_window,
        after_window=after_window,
        min_gap_s=min_gap_s,
        min_chars=min_chars,
    )


def plot_question_shifts(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    out_path: str | Path | None = None,
):
    """Create sentiment-shift-vs-time chart for detected questions."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No detected analyst questions",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(
            df["question_time"],
            df["sentiment_shift"],
            marker="o",
            linewidth=1.5,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Question Time (s)")
        ax.set_ylabel("Sentiment Shift (after - before)")
    ax.set_title("Question-Driven Sentiment Shifts")
    fig.tight_layout()
    target_path = out_path if out_path is not None else output_path
    if target_path is not None:
        fig.savefig(Path(target_path), dpi=150)
    return fig


def run(
    segments: list[dict[str, Any]],
    chunks_scored: pd.DataFrame,
    out_dir: str | Path,
    pre_window_s: float = 60.0,
    post_window_s: float = 120.0,
    min_gap_s: float = 30.0,
    min_chars: int = 15,
) -> pd.DataFrame:
    """Compute question shifts and write standard CSV/PNG artifacts."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=pre_window_s,
        after_window=post_window_s,
        min_gap_s=min_gap_s,
        min_chars=min_chars,
    )
    csv_path = output_dir / "question_sentiment_shifts.csv"
    plot_path = output_dir / "question_shifts.png"
    df.to_csv(csv_path, index=False)
    fig = plot_question_shifts(df, output_path=plot_path)
    fig.clear()
    return df


def compute(
    segments: list[dict[str, Any]],
    chunks_scored: pd.DataFrame,
    pre_window_s: float = 60.0,
    post_window_s: float = 120.0,
    min_gap_s: float = 30.0,
    min_chars: int = 15,
) -> pd.DataFrame:
    """Compute-only alias for question shift detection."""
    return detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=pre_window_s,
        after_window=post_window_s,
        min_gap_s=min_gap_s,
        min_chars=min_chars,
    )
