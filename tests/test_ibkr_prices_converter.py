from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ibkr_prices_to_prices_csv.py"
    )


def test_ibkr_converter_default_columns(tmp_path: Path) -> None:
    source = tmp_path / "ibkr.csv"
    output = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"Date/Time": "2024-08-01 16:00", "Symbol": "aapl", "Close": "210.5"},
            {"Date/Time": "2024-08-01 17:00", "Symbol": "aapl", "Close": "211.0"},
            {"Date/Time": "bad-date", "Symbol": "aapl", "Close": "200.0"},
        ]
    ).to_csv(source, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--in",
            str(source),
            "--out",
            str(output),
            "--timezone",
            "US/Eastern",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.exists()

    data = pd.read_csv(output)
    assert list(data.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert len(data) == 2
    assert set(data["symbol"].tolist()) == {"AAPL"}
    assert data["open"].tolist() == data["close"].tolist()
    assert data["high"].tolist() == data["close"].tolist()
    assert data["low"].tolist() == data["close"].tolist()
    assert data["volume"].tolist() == [0.0, 0.0]
    assert data["timestamp"].str.endswith("Z").all()


def test_ibkr_converter_override_columns_and_symbol(tmp_path: Path) -> None:
    source = tmp_path / "ibkr_alt.csv"
    output = tmp_path / "prices_alt.csv"
    pd.DataFrame(
        [
            {"Time": "2024-08-01T20:00:00Z", "Last Price": "99.1", "Qty": "150"},
            {"Time": "2024-08-01T21:00:00Z", "Last Price": "98.6", "Qty": "200"},
        ]
    ).to_csv(source, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--in",
            str(source),
            "--out",
            str(output),
            "--symbol",
            "msft",
            "--timestamp-col",
            "Time",
            "--close-col",
            "Last Price",
            "--volume-col",
            "Qty",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = pd.read_csv(output)
    assert len(data) == 2
    assert set(data["symbol"].tolist()) == {"MSFT"}
    assert data["close"].tolist() == [99.1, 98.6]
    assert data["volume"].tolist() == [150.0, 200.0]
