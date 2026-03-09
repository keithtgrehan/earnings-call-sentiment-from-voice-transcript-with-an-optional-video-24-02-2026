from __future__ import annotations

from pathlib import Path

import pandas as pd

from earnings_call_sentiment import cli as cli_module


def test_metrics_schema_version_keys_present() -> None:
    payload = cli_module._build_metrics_payload(
        chunks_scored=pd.DataFrame(),
        guidance_df=pd.DataFrame(),
        guidance_revision_df=pd.DataFrame(),
        tone_changes_df=pd.DataFrame(),
        prior_guidance_path=None,
        sentiment_model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        sentiment_revision="714eb0f",
    )
    assert payload["schema_version"] == "1.0.0"
    assert "git_commit" in payload
    assert "package_version" in payload
    assert isinstance(payload["generated_at"], str) and payload["generated_at"]
    assert payload["models"]["sentiment"]["model"] == (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    assert payload["models"]["sentiment"]["revision"] == "714eb0f"


def test_strict_mode_fails_on_missing_artifact(tmp_path: Path) -> None:
    # Create only one artifact; strict validation must fail with missing file list.
    (tmp_path / "transcript.json").write_text("{}", encoding="utf-8")
    errors = cli_module._validate_strict_outputs(tmp_path)
    assert errors
    assert any("sentiment_segments.csv" in item for item in errors)


def test_strict_mode_passes_when_artifacts_present(tmp_path: Path) -> None:
    required = cli_module._strict_required_artifacts(tmp_path)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".md":
            path.write_text("# report", encoding="utf-8")
        elif path.suffix == ".txt":
            path.write_text("text", encoding="utf-8")
        elif path.suffix in {".csv", ".jsonl"}:
            path.write_text("col\nvalue\n", encoding="utf-8")
        elif path.suffix == ".png":
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
        else:
            path.write_text("x", encoding="utf-8")

    errors = cli_module._validate_strict_outputs(tmp_path)
    assert errors == []
