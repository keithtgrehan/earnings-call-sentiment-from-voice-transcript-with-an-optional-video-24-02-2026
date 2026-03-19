#!/usr/bin/env python3
"""Validate optional external dataset staging roots and placeholder loaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.datasets import (
    list_dataset_entries,
    normalize_dataset_records,
    validate_dataset_presence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate staged external dataset roots for MAEC, MELD, and RAVDESS."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "maec", "meld", "ravdess"),
        default="all",
        help="Dataset to validate. Defaults to all.",
    )
    parser.add_argument(
        "--maec-root",
        help="Optional MAEC root override.",
    )
    parser.add_argument(
        "--meld-root",
        help="Optional MELD root override.",
    )
    parser.add_argument(
        "--ravdess-root",
        help="Optional RAVDESS root override.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="Number of normalized sample rows to include when a dataset is present.",
    )
    return parser.parse_args()


def _root_override(args: argparse.Namespace, dataset_id: str) -> str | None:
    if dataset_id == "maec":
        return args.maec_root
    if dataset_id == "meld":
        return args.meld_root
    if dataset_id == "ravdess":
        return args.ravdess_root
    return None


def main() -> int:
    args = parse_args()
    dataset_ids = (
        [entry.dataset_id for entry in list_dataset_entries()]
        if args.dataset == "all"
        else [args.dataset]
    )

    reports = []
    for dataset_id in dataset_ids:
        report = validate_dataset_presence(
            dataset_id,
            root_override=_root_override(args, dataset_id),
        )
        if report.get("present") and report.get("structure_ok"):
            samples = normalize_dataset_records(
                dataset_id,
                root_override=_root_override(args, dataset_id),
                limit=args.sample_limit,
            )
        else:
            samples = []
        report["normalized_sample_records"] = samples
        report["normalized_sample_count"] = len(samples)
        reports.append(report)

    print(json.dumps({"datasets": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
