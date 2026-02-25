from __future__ import annotations

from pathlib import Path

import pytest

from earnings_call_sentiment import cli as cli_module


def test_cli_dry_run_skips_download_and_creates_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        cli_module,
        "download_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("download_audio should not run in --dry-run mode")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_pipeline should not run in --dry-run mode")
        ),
    )

    exit_code = cli_module.main(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not list(cache_dir.glob("audio.*"))
