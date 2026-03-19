#!/usr/bin/env python3
"""Summarize NLP sidecar disagreement against deterministic transcript artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from earnings_call_sentiment.nlp_sidecar import (
    default_nlp_dir,
    repo_root,
    write_nlp_disagreement_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize FinBERT and optional emotion sidecars against deterministic transcript artifacts."
    )
    parser.add_argument(
        "--source-id",
        required=True,
        help="Source id used to resolve default sidecar and deterministic output paths.",
    )
    parser.add_argument(
        "--nlp-scores",
        help="Optional path to nlp_segment_scores.csv. Defaults to the standard sidecar location.",
    )
    parser.add_argument(
        "--deterministic-out-dir",
        help="Optional output directory containing chunks_scored.csv, guidance_revision.csv, and qa_shift_summary.json.",
    )
    parser.add_argument(
        "--out-path",
        help="Optional JSON output path. Defaults next to the sidecar scores CSV.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def main() -> int:
    args = parse_args()
    default_scores_path = default_nlp_dir(source_id=args.source_id) / "nlp_segment_scores.csv"
    scores_path = Path(args.nlp_scores).expanduser().resolve() if args.nlp_scores else default_scores_path
    if not scores_path.exists():
        raise RuntimeError(f"NLP sidecar scores CSV not found: {scores_path}")

    deterministic_out_dir = (
        Path(args.deterministic_out_dir).expanduser().resolve()
        if args.deterministic_out_dir
        else (repo_root() / "outputs" / args.source_id).resolve()
    )
    out_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path
        else scores_path.parent / "nlp_disagreement_summary.json"
    )

    write_nlp_disagreement_summary(
        score_rows=_load_rows(scores_path),
        out_path=out_path,
        deterministic_out_dir=deterministic_out_dir if deterministic_out_dir.exists() else None,
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "nlp_scores": str(scores_path),
                "deterministic_out_dir": str(deterministic_out_dir),
                "summary_path": str(out_path),
                "notes": [
                    "This summary is for comparison only.",
                    "Deterministic outputs remain the source of truth.",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
