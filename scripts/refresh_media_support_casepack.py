#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.media_support_casepack import (
    TASK_IMPACT_ASSIGNMENT_COLUMNS,
    TASK_IMPACT_RESULTS_COLUMNS,
    build_downstream_case_frame,
    build_task_impact_case_frame,
    ensure_case_outputs,
    repo_root,
    write_case_frame,
    write_header_only_csv,
)


DOWNSTREAM_CASES = Path("data/media_support_eval/downstream_decision_eval_cases.csv")
TASK_IMPACT_CASES = Path("data/media_support_eval/task_impact_eval_cases.csv")
TASK_IMPACT_RESULTS = Path("data/media_support_eval/task_impact_results_template.csv")
TASK_IMPACT_ASSIGNMENTS = Path("data/media_support_eval/task_impact_assignment_template.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the repo-native downstream/media-support case packages.")
    parser.add_argument(
        "--generate-missing-outputs",
        action="store_true",
        help="Generate missing transcript-review output bundles for labeled cases that do not already have metrics/report artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild generated transcript-review bundles even when output artifacts already exist.",
    )
    return parser.parse_args()


def _ensure_header_only(path: Path, columns: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    write_header_only_csv(path, columns)


def main() -> None:
    args = parse_args()
    root = repo_root()

    downstream = build_downstream_case_frame(root)
    task_impact = build_task_impact_case_frame(root)

    generated_dirs: list[str] = []
    if args.generate_missing_outputs:
        generated = ensure_case_outputs(task_impact, root=root, force=args.force)
        generated_dirs = [str(path) for path in generated]
        downstream = build_downstream_case_frame(root)
        task_impact = build_task_impact_case_frame(root)

    write_case_frame(downstream, root / DOWNSTREAM_CASES)
    write_case_frame(task_impact, root / TASK_IMPACT_CASES)
    _ensure_header_only(root / TASK_IMPACT_RESULTS, TASK_IMPACT_RESULTS_COLUMNS)
    _ensure_header_only(root / TASK_IMPACT_ASSIGNMENTS, TASK_IMPACT_ASSIGNMENT_COLUMNS)

    summary = {
        "downstream_case_rows": int(len(downstream)),
        "downstream_cases_with_support_targets": int(
            downstream["target_support_direction"].astype(str).str.strip().ne("").sum()
        ),
        "task_impact_case_rows": int(len(task_impact)),
        "generated_output_dirs": generated_dirs,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {root / DOWNSTREAM_CASES}")
    print(f"wrote {root / TASK_IMPACT_CASES}")
    print(f"wrote {root / TASK_IMPACT_RESULTS}")
    print(f"wrote {root / TASK_IMPACT_ASSIGNMENTS}")


if __name__ == "__main__":
    main()
