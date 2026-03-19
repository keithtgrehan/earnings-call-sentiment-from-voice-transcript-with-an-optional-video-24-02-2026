from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.openface_sidecar import (
    default_visual_dir,
    run_openface_feature_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optional OpenFace extraction and build conservative per-segment visual summaries."
    )
    parser.add_argument("--source-id", required=True, help="Source id used for manifest lookup and output naming.")
    parser.add_argument("--video-path", required=True, help="Local video file path.")
    parser.add_argument(
        "--openface-bin",
        default=None,
        help="Optional explicit path or executable name for OpenFace FeatureExtraction.",
    )
    parser.add_argument(
        "--source-manifest",
        default=None,
        help="Optional source manifest override. Defaults to data/source_manifests/earnings_call_sources.csv.",
    )
    parser.add_argument(
        "--segment-manifest",
        default=None,
        help="Optional segment manifest override. Defaults to data/source_manifests/earnings_call_segments.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional base output dir. Defaults to data/processed/multimodal/visual/<source_id>/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_openface_feature_sidecar(
        source_id=str(args.source_id).strip(),
        video_path=Path(args.video_path).expanduser().resolve(),
        out_dir=default_visual_dir(source_id=str(args.source_id).strip(), out_dir=args.out_dir),
        openface_bin=args.openface_bin,
        source_manifest_path=(Path(args.source_manifest).expanduser().resolve() if args.source_manifest else None),
        segment_manifest_path=(Path(args.segment_manifest).expanduser().resolve() if args.segment_manifest else None),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
