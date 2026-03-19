#!/usr/bin/env python3
"""Build a conservative multimodal coverage summary."""

from __future__ import annotations

import argparse
import json

from earnings_call_sentiment.multimodal_eval_summary import write_multimodal_eval_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multimodal coverage, usability, and sidecar evidence without claiming predictive advantage."
    )
    parser.add_argument(
        "--source-manifest",
        help="Optional source manifest override.",
    )
    parser.add_argument(
        "--segment-manifest",
        help="Optional segment manifest override.",
    )
    parser.add_argument(
        "--alignment-root",
        help="Optional alignment sidecar root override.",
    )
    parser.add_argument(
        "--visual-root",
        help="Optional visual sidecar root override.",
    )
    parser.add_argument(
        "--nlp-root",
        help="Optional NLP sidecar root override.",
    )
    parser.add_argument(
        "--out-dir",
        help="Optional output directory. Defaults to data/processed/multimodal/eval/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = write_multimodal_eval_summary(
        out_dir=args.out_dir,
        sources_path=args.source_manifest,
        segments_path=args.segment_manifest,
        alignment_root=args.alignment_root,
        visual_root=args.visual_root,
        nlp_root=args.nlp_root,
    )
    print(
        json.dumps(
            {
                "summary_path": str(paths["summary_path"]),
                "coverage_csv_path": str(paths["coverage_csv_path"]),
                "notes": [
                    "Transcript remains primary.",
                    "Audio, visual, and NLP layers are supporting instrumentation only.",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
