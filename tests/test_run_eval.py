from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def test_run_eval_none_mode_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "num_chunks_scored": 123,
        "sentiment_mean": -0.04,
        "sentiment_std": 0.17,
        "guidance": {"row_count": 8, "mean_strength": 0.52},
        "tone_changes": {"row_count": 15, "change_count": 4},
        "guidance_revision": {
            "matched_count": 5,
            "raised_count": 2,
            "lowered_count": 2,
            "reaffirmed_count": 1,
            "mixed_count": 0,
            "unclear_count": 0,
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload), encoding="utf-8"
    )
    (out_dir / "report.md").write_text("synthetic report", encoding="utf-8")

    guidance = pd.DataFrame(
        [
            {"topic": "revenue", "period": "FY", "revision_label": "raised", "diff": 0.2, "overlap_score": 0.9, "current_text_snippet": "raised revenue outlook"},
            {"topic": "margin", "period": "Q4", "revision_label": "lowered", "diff": -0.5, "overlap_score": 0.8, "current_text_snippet": "lower margin guidance"},
            {"topic": "eps", "period": "FY", "revision_label": "reaffirmed", "diff": 0.0, "overlap_score": 0.7, "current_text_snippet": "reaffirmed eps"},
            {"topic": "opex", "period": "FY", "revision_label": "lowered", "diff": -0.8, "overlap_score": 0.6, "current_text_snippet": "opex pressure"},
            {"topic": "capex", "period": "Q1", "revision_label": "raised", "diff": 0.1, "overlap_score": 0.5, "current_text_snippet": "capex increase"},
            {"topic": "other", "period": "Unknown", "revision_label": "unclear", "diff": 0.03, "overlap_score": 0.4, "current_text_snippet": "unclear context"},
        ]
    )
    guidance.to_csv(out_dir / "guidance_revision.csv", index=False)

    tone = pd.DataFrame(
        [
            {"start": 10.0, "end": 20.0, "tone_change_z": -1.0, "is_change": True, "text": "minor change"},
            {"start": 30.0, "end": 40.0, "tone_change_z": -3.4, "is_change": True, "text": "major negative shift"},
            {"start": 50.0, "end": 60.0, "tone_change_z": 2.8, "is_change": True, "text": "major positive shift"},
            {"start": 70.0, "end": 80.0, "tone_change_z": 0.5, "is_change": False, "text": "ignored if changed set exists"},
        ]
    )
    tone.to_csv(out_dir / "tone_changes.csv", index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--out-dir",
            str(out_dir),
            "--llm",
            "none",
            "--limit",
            "5",
            "--top-k",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    json_path = out_dir / "llm_eval.json"
    md_path = out_dir / "llm_eval.md"
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.stat().st_size > 0
    assert md_path.stat().st_size > 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["llm_mode"] == "none"
    assert payload["key_metrics"]["num_chunks_scored"] == 123
    assert len(payload["top_guidance_revisions"]) == 5
    assert len(payload["top_tone_changes"]) == 3
    assert payload["top_guidance_revisions"][0]["topic"] == "opex"
    assert payload["top_tone_changes"][0]["tone_change_z"] == -3.4


def test_run_eval_none_mode_without_optional_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"num_chunks_scored": 1, "sentiment_mean": 0.0, "sentiment_std": 0.0}),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--out-dir",
            str(out_dir),
            "--llm",
            "none",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((out_dir / "llm_eval.json").read_text(encoding="utf-8"))
    assert payload["top_guidance_revisions"] == []
    assert payload["top_tone_changes"] == []
    assert "narrative" in payload
