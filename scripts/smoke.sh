#!/usr/bin/env bash
set -euo pipefail

echo "== python =="
python -V

echo "== venv =="
python -c "import sys; print(sys.executable)"

echo "== imports =="
python - <<'PY'
import importlib
mods = [
    "earnings_call_sentiment",
    "earnings_call_sentiment.cli",
    "earnings_call_sentiment.downloader",
    "earnings_call_sentiment.transcriber",
    "earnings_call_sentiment.sentiment",
]
for m in mods:
    importlib.import_module(m)
print("OK imports")
PY

echo "== CLI help =="
python -m earnings_call_sentiment --help >/dev/null
echo "OK --help"

echo "== CLI dry-run =="
earnings-call-sentiment --youtube-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --dry-run >/dev/null
echo "OK --dry-run"

echo "SMOKE PASS"
