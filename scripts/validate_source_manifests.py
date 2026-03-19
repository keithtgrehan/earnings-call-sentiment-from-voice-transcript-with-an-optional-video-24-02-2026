from __future__ import annotations

import json
import sys

from earnings_call_sentiment.source_manifests import validate_source_manifests


def main() -> int:
    report = validate_source_manifests()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
