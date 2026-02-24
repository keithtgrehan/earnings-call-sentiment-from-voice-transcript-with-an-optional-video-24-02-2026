import argparse

def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="earnings-call-sentiment",
        description="Earnings call sentiment pipeline (YouTube URL input).",
    )

def main(argv=None) -> int:
    parser = build_parser()
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    _args = parser.parse_args(argv)
    return 0
