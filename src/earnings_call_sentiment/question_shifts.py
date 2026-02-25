from __future__ import annotations

import pandas as pd

ANALYST_CUES = (
    "question",
    "questions",
    "ask",
    "could you",
    "can you",
    "what about",
    "how do you",
    "guidance",
    "outlook",
    "margin",
    "demand",
)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either start/end or start_s/end_s etc."""
    out = df.copy()
    if "start_s" in out.columns and "start" not in out.columns:
        out = out.rename(columns={"start_s": "start"})
    if "end_s" in out.columns and "end" not in out.columns:
        out = out.rename(columns={"end_s": "end"})
    return out


def _signed_from_label_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'signed' column exists. If already present, leave it."""
    out = df.copy()

    if "signed" in out.columns:
        return out

    label_col = None
    score_col = None

    for c in ("sentiment_label", "label"):
        if c in out.columns:
            label_col = c
            break

    for c in ("sentiment_score", "score", "sentiment_confidence"):
        if c in out.columns:
            score_col = c
            break

    if label_col is None or score_col is None:
        raise ValueError(
            f"Need either signed column or label+score columns. Have: {list(out.columns)}"
        )

    def row_signed(r: pd.Series) -> float:
        lab = str(r[label_col]).upper()
        s = float(r[score_col])
        return s if lab == "POSITIVE" else -s

    out["signed"] = out.apply(row_signed, axis=1)
    return out


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 15:
        return False
    tl = t.lower()
    if "?" in t:
        return True
    return any(cue in tl for cue in ANALYST_CUES)


def _avg_sentiment_in_window(df: pd.DataFrame, t0: float, t1: float) -> float | None:
    """Average signed sentiment for rows overlapping [t0, t1]."""
    if t1 <= t0:
        return None
    window = df[(df["end"] >= t0) & (df["start"] <= t1)]
    if window.empty:
        return None
    return float(window["signed"].mean())


def detect_question_shifts(
    segments: list[dict],
    chunks_scored: pd.DataFrame,
    *,
    pre_window_s: float = 60.0,
    post_window_s: float = 120.0,
    min_gap_s: float = 30.0,
) -> pd.DataFrame:
    """
    Detect 'question-like' segments and compute sentiment shift around them.

    Returns columns:
    - question_time_s
    - question_text
    - sentiment_before
    - sentiment_after
    - sentiment_shift
    """
    df = _normalize_cols(chunks_scored)
    df = _signed_from_label_score(df)

    if "start" not in df.columns or "end" not in df.columns:
        raise ValueError(f"chunks_scored must have start/end columns. Have: {list(df.columns)}")

    questions: list[dict] = []
    last_q_time: float | None = None

    for seg in segments:
        t = float(seg["start"])
        text = str(seg.get("text", ""))

        if not _looks_like_question(text):
            continue

        # de-duplicate: ignore questions too close together
        if last_q_time is not None and (t - last_q_time) < min_gap_s:
            continue

        before = _avg_sentiment_in_window(df, t - pre_window_s, t)
        after = _avg_sentiment_in_window(df, t, t + post_window_s)

        shift = None
        if before is not None and after is not None:
            shift = float(after - before)

        questions.append(
            {
                "question_time_s": t,
                "question_text": text,
                "sentiment_before": before,
                "sentiment_after": after,
                "sentiment_shift": shift,
            }
        )
        last_q_time = t

    out = pd.DataFrame(questions)
    if not out.empty:
        out = out.sort_values("question_time_s").reset_index(drop=True)
    return out


def plot_question_shifts(df: pd.DataFrame, *, out_path: str = "outputs/question_shifts.png") -> str:
    """Plot sentiment_shift over time and save as PNG."""
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("No rows to plot.")

    x = df["question_time_s"] / 60.0
    y = df["sentiment_shift"]

    plt.figure()
    plt.scatter(x, y)
    plt.axhline(0.0)
    plt.xlabel("Question time (minutes)")
    plt.ylabel("Sentiment shift (after - before)")
    plt.title("Sentiment shifts around detected questions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    return out_path
