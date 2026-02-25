import subprocess
import sys


def test_cli_help_works():
    r = subprocess.run(
        [sys.executable, "-m", "earnings_call_sentiment", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    # stable assertions
    assert "usage:" in out
    assert "earnings-call-sentiment" in out
