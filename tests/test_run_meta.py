from __future__ import annotations

import json
from pathlib import Path

from earnings_call_sentiment import cli as cli_module


def test_write_run_meta_contains_required_schema(tmp_path: Path) -> None:
    path = cli_module._write_run_meta(
        out_dir=tmp_path,
        symbol="aapl",
        event_dt="2024-08-01T16:00:00-04:00",
        source_url="https://www.youtube.com/watch?v=test123",
    )
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "symbol",
        "event_dt",
        "source_url",
        "run_id",
        "generated_at",
        "version",
    }
    assert payload["symbol"] == "AAPL"
    assert payload["event_dt"] == "2024-08-01T16:00:00-04:00"
    assert payload["source_url"] == "https://www.youtube.com/watch?v=test123"
    assert isinstance(payload["run_id"], str) and payload["run_id"]
    assert isinstance(payload["generated_at"], str) and payload["generated_at"]
    assert isinstance(payload["version"], str) and payload["version"]


def test_normalize_event_dt_accepts_space_format() -> None:
    value, defaulted = cli_module._normalize_event_dt("2024-08-01 16:00")
    assert defaulted is False
    assert value.startswith("2024-08-01T16:00:00")


def test_normalize_event_dt_defaults_to_now() -> None:
    value, defaulted = cli_module._normalize_event_dt(None)
    assert defaulted is True
    assert "T" in value
