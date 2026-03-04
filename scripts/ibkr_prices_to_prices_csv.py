#!/usr/bin/env python3
"""Convert IBKR-exported CSV data into canonical prices.csv format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

TIMESTAMP_CANDIDATES = ["Date/Time", "DateTime", "Datetime", "Time", "Date"]
CLOSE_CANDIDATES = ["Close", "Price", "Last", "Last Price", "TradePrice"]
SYMBOL_CANDIDATES = ["Symbol", "Ticker", "Underlying", "Underlying Symbol"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IBKR export CSV into timestamp,symbol,open,high,low,close,volume format."
    )
    parser.add_argument("--in", dest="input_csv", required=True, help="Input IBKR CSV.")
    parser.add_argument(
        "--out", dest="output_csv", default="prices.csv", help="Output prices CSV path."
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol override for all rows (uppercased).",
    )
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help='Timestamp column name (default tries: "Date/Time","DateTime","Datetime","Time","Date").',
    )
    parser.add_argument(
        "--close-col",
        default=None,
        help='Close/price column name (default tries: "Close","Price","Last","Last Price","TradePrice").',
    )
    parser.add_argument("--open-col", default=None, help="Optional open column name.")
    parser.add_argument("--high-col", default=None, help="Optional high column name.")
    parser.add_argument("--low-col", default=None, help="Optional low column name.")
    parser.add_argument("--volume-col", default=None, help="Optional volume column name.")
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone used for naive timestamps before conversion to UTC (default: UTC).",
    )
    return parser.parse_args(argv)


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: list[str],
    *,
    required: bool,
    field_name: str,
) -> str | None:
    columns = frame.columns.tolist()
    if explicit:
        if explicit in columns:
            return explicit
        raise RuntimeError(f"Column not found for {field_name}: {explicit}")

    lower_map = {col.strip().lower(): col for col in columns}
    for candidate in candidates:
        direct = next((col for col in columns if col == candidate), None)
        if direct is not None:
            return direct
        mapped = lower_map.get(candidate.strip().lower())
        if mapped is not None:
            return mapped

    if required:
        raise RuntimeError(
            f"Could not find required {field_name} column. "
            f"Tried: {candidates}. Available: {columns}"
        )
    return None


def _parse_single_timestamp(value: Any, timezone_name: str, tz: ZoneInfo) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is None:
        try:
            timestamp = timestamp.tz_localize(tz)
        except Exception:
            # In ambiguous/nonexistent DST edge cases, mark invalid and drop.
            return None
    try:
        return timestamp.tz_convert("UTC")
    except Exception as exc:
        raise RuntimeError(
            f"Failed converting timestamp to UTC for timezone {timezone_name}: {value}"
        ) from exc


def convert_ibkr_prices(
    *,
    input_csv: Path,
    output_csv: Path,
    symbol_override: str | None,
    timestamp_col: str | None,
    close_col: str | None,
    open_col: str | None,
    high_col: str | None,
    low_col: str | None,
    volume_col: str | None,
    timezone_name: str,
) -> pd.DataFrame:
    try:
        tz = ZoneInfo(timezone_name)
    except Exception as exc:  # pragma: no cover - platform dependent msg
        raise RuntimeError(f"Invalid timezone: {timezone_name}") from exc

    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise RuntimeError(f"Input CSV is empty: {input_csv}")

    ts_col = _resolve_column(
        frame,
        timestamp_col,
        TIMESTAMP_CANDIDATES,
        required=True,
        field_name="timestamp",
    )
    cls_col = _resolve_column(
        frame,
        close_col,
        CLOSE_CANDIDATES,
        required=True,
        field_name="close",
    )
    sym_col = _resolve_column(
        frame,
        None,
        SYMBOL_CANDIDATES,
        required=False,
        field_name="symbol",
    )
    opn_col = _resolve_column(
        frame,
        open_col,
        [],
        required=False,
        field_name="open",
    )
    hgh_col = _resolve_column(
        frame,
        high_col,
        [],
        required=False,
        field_name="high",
    )
    low_col_resolved = _resolve_column(
        frame,
        low_col,
        [],
        required=False,
        field_name="low",
    )
    vol_col = _resolve_column(
        frame,
        volume_col,
        [],
        required=False,
        field_name="volume",
    )

    timestamps = frame[ts_col].map(lambda item: _parse_single_timestamp(item, timezone_name, tz))
    close_values = pd.to_numeric(frame[cls_col], errors="coerce")
    open_values = (
        pd.to_numeric(frame[opn_col], errors="coerce")
        if opn_col is not None
        else close_values.copy()
    )
    high_values = (
        pd.to_numeric(frame[hgh_col], errors="coerce")
        if hgh_col is not None
        else close_values.copy()
    )
    low_values = (
        pd.to_numeric(frame[low_col_resolved], errors="coerce")
        if low_col_resolved is not None
        else close_values.copy()
    )
    volume_values = (
        pd.to_numeric(frame[vol_col], errors="coerce").fillna(0)
        if vol_col is not None
        else pd.Series([0] * len(frame), index=frame.index, dtype="float64")
    )

    if symbol_override is not None and str(symbol_override).strip():
        symbols = pd.Series(
            [str(symbol_override).strip().upper()] * len(frame), index=frame.index
        )
    elif sym_col is not None:
        symbols = frame[sym_col].astype(str).str.strip().str.upper()
    else:
        symbols = pd.Series(["UNKNOWN"] * len(frame), index=frame.index)

    output = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "open": open_values,
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values,
        }
    )
    output = output.dropna(subset=["timestamp", "close"]).copy()
    output["open"] = pd.to_numeric(output["open"], errors="coerce").fillna(output["close"])
    output["high"] = pd.to_numeric(output["high"], errors="coerce").fillna(output["close"])
    output["low"] = pd.to_numeric(output["low"], errors="coerce").fillna(output["close"])
    output["volume"] = pd.to_numeric(output["volume"], errors="coerce").fillna(0)

    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    output = output.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    output["timestamp"] = output["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output = output[
        ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    ].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = convert_ibkr_prices(
        input_csv=Path(args.input_csv).expanduser().resolve(),
        output_csv=Path(args.output_csv).expanduser().resolve(),
        symbol_override=args.symbol,
        timestamp_col=args.timestamp_col,
        close_col=args.close_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        volume_col=args.volume_col,
        timezone_name=str(args.timezone),
    )
    print(f"Wrote {len(output)} rows to {Path(args.output_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
