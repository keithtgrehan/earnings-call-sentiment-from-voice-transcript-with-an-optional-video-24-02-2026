#!/usr/bin/env bash
set -euo pipefail

echo "== ruff =="
ruff check . --fix
ruff format .

echo "== pyright =="
pyright

echo "== pytest =="
pytest -q

echo "== smoke =="
./scripts/smoke.sh

echo "CHECK PASS"
