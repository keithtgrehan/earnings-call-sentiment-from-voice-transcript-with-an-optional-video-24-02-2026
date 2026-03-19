from __future__ import annotations

import argparse
import json

from earnings_call_sentiment.source_manifests import validate_source_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manually curated source/video/transcript pairs from the source manifest."
    )
    parser.add_argument(
        "--source-manifest",
        help="Optional source manifest override. Defaults to data/source_manifests/earnings_call_sources.csv.",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="Optionally make lightweight URL HEAD/GET checks for non-template rows.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for optional URL checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_source_pairs(
        sources_path=args.source_manifest,
        check_urls=args.check_urls,
        timeout_s=args.timeout,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
