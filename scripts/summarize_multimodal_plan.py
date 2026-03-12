from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'multimodal_signal_eval' / 'acquisition_manifest.csv'
REQUIRED_COLUMNS = {
    'candidate_id',
    'ticker',
    'company',
    'year',
    'quarter',
    'event_date',
    'source_package',
    'source_type',
    'source_url',
    'modality_priority',
    'expected_value',
    'current_status',
    'notes',
}


def main() -> int:
    if not MANIFEST.exists():
        print(f'missing manifest: {MANIFEST}', file=sys.stderr)
        return 1
    with MANIFEST.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            print(f'manifest missing columns: {sorted(missing)}', file=sys.stderr)
            return 1
        rows = list(reader)
    if not rows:
        print('manifest has no rows', file=sys.stderr)
        return 1

    by_priority = Counter(row['modality_priority'] for row in rows)
    by_status = Counter(row['current_status'] for row in rows)
    by_source_type = Counter(row['source_type'] for row in rows)

    print('Multimodal acquisition manifest summary')
    print(f'rows: {len(rows)}')
    print('')
    print('By modality_priority:')
    for key, value in sorted(by_priority.items()):
        print(f'- {key}: {value}')
    print('')
    print('By current_status:')
    for key, value in sorted(by_status.items()):
        print(f'- {key}: {value}')
    print('')
    print('By source_type:')
    for key, value in sorted(by_source_type.items()):
        print(f'- {key}: {value}')
    print('')
    print('Top shortlist:')
    for row in rows[:5]:
        print(
            f"- {row['candidate_id']} {row['ticker']} {row['quarter']} | "
            f"{row['modality_priority']} | {row['current_status']} | {row['expected_value']}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
