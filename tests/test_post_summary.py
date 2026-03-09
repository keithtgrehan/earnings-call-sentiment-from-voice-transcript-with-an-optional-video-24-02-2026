from __future__ import annotations

from pathlib import Path

import pytest

from earnings_call_sentiment.post_summary import generate_optional_summary
from earnings_call_sentiment.summary_config import SummaryConfig


def test_disabled_summary_mode_returns_none(tmp_path: Path) -> None:
    config = SummaryConfig(
        enabled=False,
        provider="none",
        model=None,
        base_url=None,
        api_key_env=None,
        timeout_s=30.0,
    )
    assert generate_optional_summary(tmp_path, config, verbose=False) is None


def test_enabled_summary_mode_missing_config_fails_fast(tmp_path: Path) -> None:
    config = SummaryConfig(
        enabled=True,
        provider="openai_compatible",
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key_env="MISSING_SUMMARY_KEY",
        timeout_s=5.0,
    )

    with pytest.raises(RuntimeError, match="MISSING_SUMMARY_KEY"):
        generate_optional_summary(tmp_path, config, verbose=False)
