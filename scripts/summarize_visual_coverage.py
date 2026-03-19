from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from earnings_call_sentiment.openface_sidecar import build_visual_coverage_summary, repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize conservative OpenFace visual coverage across sidecar outputs."
    )
    parser.add_argument(
        "--visual-root",
        default="data/processed/multimodal/visual",
        help="Base directory that contains per-source segment_visual_features.csv files.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional JSON output path. Defaults to <visual-root>/visual_coverage_summary.json.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    visual_root = (repo_root() / Path(args.visual_root)).resolve()
    csv_paths = sorted(visual_root.glob("*/segment_visual_features.csv"))
    if not csv_paths:
        payload = {
            "status": "ok",
            "visual_root": str(visual_root),
            "files_found": 0,
            "segments_total": 0,
            "segments_attempted": 0,
            "segments_usable": 0,
            "segments_unusable": 0,
            "segments_skipped": 0,
            "extraction_success_rate": 0.0,
            "source_group_coverage": 0,
            "source_breakdown": {},
        }
    else:
        frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
        merged = pd.concat(frames, ignore_index=True)
        coverage = build_visual_coverage_summary(merged.to_dict(orient="records"))
        source_breakdown: dict[str, dict[str, int]] = {}
        for source_id, group in merged.groupby("source_id"):
            source_breakdown[str(source_id)] = {
                "segments_total": int(len(group)),
                "segments_attempted": int((group["attempted"].astype(str).str.lower() == "true").sum()),
                "segments_usable": int((group["segment_visual_usability"].astype(str) == "usable").sum()),
                "segments_unusable": int((group["segment_visual_usability"].astype(str) == "unusable").sum()),
            }
        payload = {
            "status": "ok",
            "visual_root": str(visual_root),
            "files_found": len(csv_paths),
            **coverage,
            "source_breakdown": source_breakdown,
        }

    out_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path
        else visual_root / "visual_coverage_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
