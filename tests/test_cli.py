from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing}" if existing else str(src_path)
    )

    proc = subprocess.run(
        [sys.executable, "-m", "earnings_call_sentiment", "--help"],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
