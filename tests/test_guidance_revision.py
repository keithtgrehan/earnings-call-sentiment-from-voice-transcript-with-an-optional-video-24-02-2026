from __future__ import annotations

from pathlib import Path

import pandas as pd

from earnings_call_sentiment import cli as cli_module


def _guidance_df(
    *,
    topic: str,
    period: str,
    text: str,
    numbers: str,
    midpoint: float | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": 10.0,
                "end": 20.0,
                "text": text,
                "topic": topic,
                "period": period,
                "numbers": numbers,
                "numeric_signature": "",
                "midpoint_hint": midpoint,
            }
        ]
    )


def _compute_revision(
    *, prior_df: pd.DataFrame, current_df: pd.DataFrame, tmp_path: Path
) -> pd.DataFrame:
    prior_path = tmp_path / "prior_guidance.csv"
    output_path = tmp_path / "guidance_revision.csv"
    prior_df.to_csv(prior_path, index=False)
    return cli_module._compute_guidance_revision(
        current_df,
        prior_guidance_path=prior_path,
        output_path=output_path,
    )


def test_guidance_revision_range_raise(tmp_path: Path) -> None:
    prior_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 10 to 12 billion.",
        numbers="10;12",
        midpoint=11.0,
    )
    current_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 12 to 14 billion.",
        numbers="12;14",
        midpoint=13.0,
    )
    revision_df = _compute_revision(
        prior_df=prior_df, current_df=current_df, tmp_path=tmp_path
    )
    assert not revision_df.empty
    row = revision_df.iloc[0]
    assert bool(row["is_matched"]) is True
    assert str(row["revision_label"]) == "raised"
    assert float(row["diff"]) > 0


def test_guidance_revision_range_lowered(tmp_path: Path) -> None:
    prior_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 12 to 14 billion.",
        numbers="12;14",
        midpoint=13.0,
    )
    current_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 10 to 12 billion.",
        numbers="10;12",
        midpoint=11.0,
    )
    revision_df = _compute_revision(
        prior_df=prior_df, current_df=current_df, tmp_path=tmp_path
    )
    row = revision_df.iloc[0]
    assert bool(row["is_matched"]) is True
    assert str(row["revision_label"]) == "lowered"
    assert float(row["diff"]) < 0


def test_guidance_revision_reaffirmed(tmp_path: Path) -> None:
    prior_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 10 to 12 billion.",
        numbers="10;12",
        midpoint=11.0,
    )
    current_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 10 to 12 billion.",
        numbers="10;12",
        midpoint=11.0,
    )
    revision_df = _compute_revision(
        prior_df=prior_df, current_df=current_df, tmp_path=tmp_path
    )
    row = revision_df.iloc[0]
    assert bool(row["is_matched"]) is True
    assert str(row["revision_label"]) == "reaffirmed"
    assert abs(float(row["diff"])) < 1e-9


def test_guidance_revision_unmatched_unclear(tmp_path: Path) -> None:
    prior_df = _guidance_df(
        topic="revenue",
        period="FY",
        text="FY guidance revenue 10 to 12 billion.",
        numbers="10;12",
        midpoint=11.0,
    )
    current_df = _guidance_df(
        topic="margin",
        period="Q2",
        text="Q2 gross margin outlook 45 to 47 percent.",
        numbers="45;47",
        midpoint=46.0,
    )
    revision_df = _compute_revision(
        prior_df=prior_df, current_df=current_df, tmp_path=tmp_path
    )
    row = revision_df.iloc[0]
    assert bool(row["is_matched"]) is False
    assert str(row["revision_label"]) == "unclear"


def test_guidance_revision_empty_current_writes_headers(tmp_path: Path) -> None:
    output_path = tmp_path / "guidance_revision.csv"
    revision_df = cli_module._compute_guidance_revision(
        current_guidance=pd.DataFrame(),
        prior_guidance_path=None,
        output_path=output_path,
    )
    assert revision_df.empty
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "revision_label" in text
