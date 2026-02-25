#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running CLI dry-run smoke check..."
python -m earnings_call_sentiment \
  --youtube-url "https://example.com/watch?v=dryrun" \
  --cache-dir "./_smoke_cache" \
  --out-dir "./_smoke_out" \
  --question-shifts \
  --dry-run \
  --verbose

echo "Smoke check completed."
echo "SMOKE PASS"
