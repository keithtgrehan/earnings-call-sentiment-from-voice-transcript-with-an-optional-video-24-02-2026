#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify expected earnings-call-sentiment output artifacts."
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs",
        help="Directory containing pipeline outputs (default: ./outputs).",
    )
    parser.add_argument(
        "--require-run-meta",
        action="store_true",
        default=False,
        help="Also require out_dir/run_meta.json to exist and be non-empty.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    required = [
        out_dir / "transcript.json",
        out_dir / "transcript.txt",
        out_dir / "sentiment_segments.csv",
        out_dir / "chunks_scored.jsonl",
        out_dir / "guidance.csv",
        out_dir / "metrics.json",
        out_dir / "report.md",
    ]
    if args.require_run_meta:
        required.append(out_dir / "run_meta.json")

    failures: list[str] = []
    print(f"Verifying outputs in: {out_dir}")
    for path in required:
        ok = _nonempty(path)
        rel = path.relative_to(out_dir) if path.is_absolute() else path
        print(f"{'OK' if ok else 'MISSING'}  {rel}")
        if not ok:
            failures.append(str(path))

    if failures:
        print("\nVerification failed. Missing/empty artifacts:")
        for path in failures:
            print(f"- {path}")
        return 1

    print("\nAll required artifacts are present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
