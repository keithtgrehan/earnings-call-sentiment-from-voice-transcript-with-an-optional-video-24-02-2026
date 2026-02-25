from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys


def test_dry_run_wins_over_download_only(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing}" if existing else str(src_path)
    )
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "earnings-call-sentiment"
    shim.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -m earnings_call_sentiment "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"

    proc = subprocess.run(
        [
            "earnings-call-sentiment",
            "--youtube-url",
            "https://example.com",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--download-only",
            "--dry-run",
        ],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
    assert not list(cache_dir.glob("audio.*"))
