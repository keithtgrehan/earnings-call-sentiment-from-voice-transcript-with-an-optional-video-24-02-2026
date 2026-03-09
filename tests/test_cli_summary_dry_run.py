from __future__ import annotations

from earnings_call_sentiment.cli import main


def test_dry_run_reports_optional_summary_intent(capsys) -> None:
    exit_code = main(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--dry-run",
            "--llm-summary",
        ]
    )
    captured = capsys.readouterr()
    output = (captured.out or "") + (captured.err or "")

    assert exit_code == 0
    assert "summary_enabled=True" in output
    assert "summary_preflight_ok=False" in output
