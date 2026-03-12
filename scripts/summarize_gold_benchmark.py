from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys


ALLOWED_LABELS = {"raised", "maintained", "lowered", "withdrawn", "unclear"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_labels(rows: list[dict[str, str]], labels_path: Path) -> None:
    if not rows:
        raise SystemExit(f"{labels_path} is empty.")

    counts = Counter(row["call_id"] for row in rows)
    duplicates = sorted(call_id for call_id, count in counts.items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate call_id rows in {labels_path}: {', '.join(duplicates)}")

    invalid = sorted(
        {
            row["guidance_change_label"]
            for row in rows
            if row["guidance_change_label"] not in ALLOWED_LABELS
        }
    )
    if invalid:
        raise SystemExit(
            "Invalid guidance_change_label values in "
            f"{labels_path}: {', '.join(invalid)}"
        )


def _enrich_rows(
    label_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    manifest_by_call = {row["call_id"]: row for row in manifest_rows}
    enriched: list[dict[str, str]] = []
    for row in label_rows:
        merged = dict(row)
        manifest = manifest_by_call.get(row["call_id"], {})
        merged["source_url"] = manifest.get("source_url", "")
        merged["manifest_notes"] = manifest.get("notes", "")
        enriched.append(merged)
    return enriched


def _print_distribution(rows: list[dict[str, str]]) -> Counter[str]:
    distribution = Counter(row["guidance_change_label"] for row in rows)
    print("Gold guidance benchmark")
    print(f"Rows: {len(rows)}")
    print()
    print("Label distribution")
    for label in ["raised", "maintained", "lowered", "withdrawn", "unclear"]:
        print(f"  {label:<10} {distribution.get(label, 0)}")
    return distribution


def _print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        ("call_id", "call_id"),
        ("ticker", "ticker"),
        ("company", "company"),
        ("quarter", "quarter"),
        ("event_date", "event_date"),
        ("guidance_change_label", "label"),
    ]
    widths: dict[str, int] = {}
    for key, header in columns:
        widths[key] = max(len(header), *(len(str(row.get(key, ""))) for row in rows))

    print()
    print("Calls")
    header = "  ".join(header.ljust(widths[key]) for key, header in columns)
    divider = "  ".join("-" * widths[key] for key, _ in columns)
    print(header)
    print(divider)
    for row in rows:
        print(
            "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns)
        )


def main() -> int:
    repo_root = _repo_root()
    benchmark_root = repo_root / "data" / "gold_guidance_calls"
    labels_path = benchmark_root / "labels.csv"
    manifest_path = benchmark_root / "call_manifest.csv"

    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path}")

    label_rows = _read_csv(labels_path)
    _validate_labels(label_rows, labels_path)

    manifest_rows: list[dict[str, str]] = []
    if manifest_path.exists():
        manifest_rows = _read_csv(manifest_path)

    rows = _enrich_rows(label_rows, manifest_rows)
    rows = sorted(rows, key=lambda row: row["call_id"])

    _print_distribution(rows)
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
