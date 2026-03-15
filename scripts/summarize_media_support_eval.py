from __future__ import annotations

import json
from pathlib import Path

from earnings_call_sentiment.media_support_eval import repo_root, validate_media_support_eval


def main() -> None:
    root = repo_root()
    summary = validate_media_support_eval()
    output_dir = root / "outputs" / "media_support_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "media_support_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
