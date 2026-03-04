from __future__ import annotations

from earnings_call_sentiment.cli import build_parser


def test_parser_has_download_only_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--download-only",
        ]
    )
    assert hasattr(args, "download_only")
    assert args.download_only is True


def test_parser_accepts_question_shift_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--question-shifts",
            "--pre-window-s",
            "45",
            "--post-window-s",
            "90",
            "--min-gap-s",
            "20",
        ]
    )
    assert args.question_shifts is True
    assert args.pre_window_s == 45.0
    assert args.post_window_s == 90.0
    assert args.min_gap_s == 20.0


def test_parser_accepts_prior_guidance_and_resume_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=test123",
            "--symbol",
            "AAPL",
            "--event-dt",
            "2024-08-01T16:00:00",
            "--prior-guidance",
            "./outputs_prior/guidance.csv",
            "--force",
            "--no-resume",
            "--vad",
            "--tone-change-threshold",
            "1.5",
        ]
    )
    assert args.prior_guidance == "./outputs_prior/guidance.csv"
    assert args.force is True
    assert args.resume is False
    assert args.vad is True
    assert args.tone_change_threshold == 1.5
    assert args.symbol == "AAPL"
    assert args.event_dt == "2024-08-01T16:00:00"
