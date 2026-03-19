from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.alignment_sidecar import write_segment_candidates_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert alignment sidecar JSON into inspectable candidate segment rows."
    )
    parser.add_argument(
        "--alignment-json",
        required=True,
        help="Path to aligned_transcript.json emitted by run_whisperx_alignment.py.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output CSV path. Defaults to segment_candidates.csv next to the alignment JSON.",
    )
    parser.add_argument(
        "--speaker-role-hint",
        default="unknown",
        help="Optional role hint applied to all candidate rows.",
    )
    parser.add_argument(
        "--labeling-status",
        default="pending_review",
        help="Labeling status applied to generated candidate rows.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    out_path = write_segment_candidates_csv(
        alignment_json_path=Path(args.alignment_json).expanduser().resolve(),
        out_path=(Path(args.out_path).expanduser().resolve() if args.out_path else None),
        speaker_role_hint=str(args.speaker_role_hint).strip() or "unknown",
        labeling_status=str(args.labeling_status).strip() or "pending_review",
    )
    print(json.dumps({"segment_candidates_csv": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
