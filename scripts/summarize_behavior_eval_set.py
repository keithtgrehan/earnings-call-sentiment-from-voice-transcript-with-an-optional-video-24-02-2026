from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "data" / "behavior_signal_eval"
FILES = {
    "uncertainty": {
        "path": EVAL_ROOT / "uncertainty_labels.csv",
        "allowed": {"absent", "present", "strong"},
    },
    "reassurance": {
        "path": EVAL_ROOT / "reassurance_labels.csv",
        "allowed": {"absent", "present"},
    },
    "skepticism": {
        "path": EVAL_ROOT / "skepticism_labels.csv",
        "allowed": {"low", "medium", "high"},
    },
}


def _validate_file(name: str, path: Path, allowed: set[str]) -> tuple[int, Counter[str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")

    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{name} has no rows")

    seen_ids: set[str] = set()
    for row in rows:
        item_id = row["item_id"]
        if item_id in seen_ids:
            raise ValueError(f"duplicate item_id in {name}: {item_id}")
        seen_ids.add(item_id)

        label = row["label"]
        if label not in allowed:
            raise ValueError(f"invalid label in {name}: {label}")

        source_path = REPO_ROOT / row["source_path"]
        if not source_path.exists():
            raise FileNotFoundError(f"missing source file for {item_id}: {source_path}")

        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        start_char = int(row["start_char"])
        end_char = int(row["end_char"])
        if start_char < 0 or end_char <= start_char or end_char > len(source_text):
            raise ValueError(f"invalid offsets for {item_id}: {start_char}, {end_char}")

        span_text = source_text[start_char:end_char]
        if span_text != row["text"]:
            raise ValueError(f"text mismatch for {item_id}")

        counts[label] += 1

    return len(rows), counts


def main() -> int:
    total_rows = 0
    print("behavior_signal_eval")
    for name, config in FILES.items():
        row_count, counts = _validate_file(name, config["path"], config["allowed"])
        total_rows += row_count
        ordered = ", ".join(f"{label}={counts.get(label, 0)}" for label in sorted(config["allowed"]))
        print(f"- {name}: rows={row_count} | {ordered}")
    print(f"total_rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
