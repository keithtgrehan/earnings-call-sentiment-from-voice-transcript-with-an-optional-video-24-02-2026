#!/usr/bin/env python3
"""Link verified repo-local artifacts into the curated multimodal slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUTS_ROOT = Path("cache/curated_multimodal_slice")
PRIORITY_SOURCE_IDS = [
    "msft_fy26_q2_example",
    "bac_q4_2025_example",
    "dis_q1_fy26_example",
    "goog_q1_2025_example",
]

# Keep this mapping intentionally narrow. Only add entries when the repo-local
# outputs line up cleanly with the curated source manifest metadata.
VERIFIED_LOCAL_ARTIFACTS = {
    "msft_fy26_q2_example": {
        "transcript.json": Path("outputs/downstream_decision_eval/MSFT_2026_Q2_call05/transcript.json"),
        "transcript.txt": Path("outputs/downstream_decision_eval/MSFT_2026_Q2_call05/transcript.txt"),
        "chunks_scored.csv": Path("outputs/MSFT_2026_Q2_call05/chunks_scored.csv"),
        "chunks_scored.jsonl": Path("outputs/MSFT_2026_Q2_call05/chunks_scored.jsonl"),
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Symlink verified repo-local transcript and chunk artifacts into "
            "cache/curated_multimodal_slice/<source_id>/ without downloading anything."
        )
    )
    parser.add_argument(
        "--inputs-root",
        default=str(DEFAULT_INPUTS_ROOT),
        help="Curated slice root. Defaults to cache/curated_multimodal_slice.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=PRIORITY_SOURCE_IDS,
        help="Curated source_ids to check. Defaults to the four prioritized rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned links without creating them.",
    )
    return parser.parse_args()


def ensure_symlink(target: Path, link_path: Path, *, dry_run: bool) -> str:
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return "already_linked"
        if dry_run:
            return "would_relink"
        link_path.unlink()
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target.resolve())
        return "relinked"
    if link_path.exists():
        return "existing_conflict"
    if dry_run:
        return "would_link"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target.resolve())
    return "linked"


def main() -> int:
    args = parse_args()
    repo_dir = repo_root()
    inputs_root = (repo_dir / args.inputs_root).resolve()
    summary: list[dict[str, object]] = []

    for source_id in args.source_ids:
        artifact_map = VERIFIED_LOCAL_ARTIFACTS.get(source_id)
        if artifact_map is None:
            summary.append(
                {
                    "source_id": source_id,
                    "prepared": False,
                    "skip_reason": "no_clean_repo_local_mapping",
                    "linked_files": [],
                    "missing_files": ["audio.*", "video.*", "transcript/chunks clean match"],
                }
            )
            continue

        linked_files: list[dict[str, str]] = []
        missing_files: list[str] = []
        source_dir = inputs_root / source_id

        for dest_name, rel_target in artifact_map.items():
            target = (repo_dir / rel_target).resolve()
            if not target.exists():
                missing_files.append(str(rel_target))
                continue
            status = ensure_symlink(target=target, link_path=source_dir / dest_name, dry_run=args.dry_run)
            linked_files.append(
                {
                    "dest": str((source_dir / dest_name).relative_to(repo_dir)),
                    "src": str(rel_target),
                    "status": status,
                }
            )

        missing_files.extend(["audio.*", "video.*"])
        summary.append(
            {
                "source_id": source_id,
                "prepared": bool(linked_files),
                "skip_reason": "" if linked_files else "no_verified_local_artifacts",
                "linked_files": linked_files,
                "missing_files": missing_files,
            }
        )

    print(json.dumps({"dry_run": args.dry_run, "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
