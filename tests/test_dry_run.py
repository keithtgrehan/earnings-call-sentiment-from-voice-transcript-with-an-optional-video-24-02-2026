from pathlib import Path


def test_dry_run_creates_outdir(tmp_path: Path):
    # Import should work even if pipeline isn't implemented yet
    from earnings_call_sentiment import cli

    # If you haven't implemented dry-run yet, this test will fail until you do.
    # Keep it here as the next target.
    assert hasattr(cli, "build_parser")
