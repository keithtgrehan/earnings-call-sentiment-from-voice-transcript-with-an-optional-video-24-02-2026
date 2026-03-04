#!/usr/bin/env python3
"""Deterministic backtest harness for earnings-call sentiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_WINDOWS = "0h:1h,0h:1d,close:close"
FEATURE_COLUMNS = [
    "management_confidence_mean",
    "analyst_pressure_mean",
    "tone_change_count",
    "guidance_strength_mean",
    "guidance_revision_matched_count",
    "guidance_revision_raised_count",
    "guidance_revision_lowered_count",
    "guidance_revision_reaffirmed_count",
    "guidance_revision_unclear_count",
    "guidance_revision_mixed_count",
]


@dataclass
class RunMeta:
    run_id: str
    run_dir: Path
    outputs_dir: Path
    symbol: str
    event_dt: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest signal artifacts against post-earnings returns."
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing run subfolders (each with outputs/ artifacts).",
    )
    parser.add_argument(
        "--prices-csv",
        required=True,
        help="CSV containing timestamp,symbol,open,high,low,close,volume columns.",
    )
    parser.add_argument(
        "--event-window",
        default=DEFAULT_WINDOWS,
        help='Comma-separated windows, e.g. "0h:1h,0h:1d,close:close".',
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for backtest_results.csv, backtest_summary.json, and report.",
    )
    return parser.parse_args()


def _parse_event_dt(value: str) -> pd.Timestamp | None:
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("_", "T")
    normalized = re.sub(r"T(\d{2})-(\d{2})-(\d{2})$", r"T\1:\2:\3", normalized)
    normalized = re.sub(r"T(\d{2})-(\d{2})$", r"T\1:\2", normalized)
    parsed = pd.to_datetime(normalized, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _try_load_metadata(run_dir: Path, outputs_dir: Path) -> tuple[str, pd.Timestamp] | None:
    metadata_candidates = [
        run_dir / "event.json",
        run_dir / "metadata.json",
        outputs_dir / "event.json",
        outputs_dir / "metadata.json",
        outputs_dir / "run_meta.json",
    ]
    for path in metadata_candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        symbol = str(payload.get("symbol", "")).strip().upper()
        event_dt = _parse_event_dt(str(payload.get("event_dt", "")))
        if symbol and event_dt is not None:
            return symbol, event_dt
    return None


def _parse_name_metadata(name: str) -> tuple[str, pd.Timestamp] | None:
    # Accept patterns like AAPL_2024-08-01T16:00:00 or AAPL-2024-08-01.
    match = re.match(
        r"^(?P<symbol>[A-Za-z][A-Za-z0-9._-]{0,15})[_-](?P<dt>\d{4}-\d{2}-\d{2}(?:[T_]\d{2}[-:]?\d{2}(?:[-:]?\d{2})?)?)$",
        name,
    )
    if match is None:
        return None
    symbol = match.group("symbol").upper()
    event_dt = _parse_event_dt(match.group("dt"))
    if event_dt is None:
        return None
    return symbol, event_dt


def discover_runs(runs_dir: Path) -> list[RunMeta]:
    runs: list[RunMeta] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        outputs_dir = child / "outputs"
        if not outputs_dir.is_dir():
            # Permit a flat run directory containing artifacts directly.
            if (child / "metrics.json").exists() and (child / "report.md").exists():
                outputs_dir = child
            else:
                continue

        meta = _try_load_metadata(child, outputs_dir)
        if meta is None:
            meta = _parse_name_metadata(child.name)
        if meta is None:
            print(f"[backtest] skipping {child.name}: missing symbol/event_dt metadata")
            continue
        symbol, event_dt = meta
        runs.append(
            RunMeta(
                run_id=child.name,
                run_dir=child,
                outputs_dir=outputs_dir,
                symbol=symbol,
                event_dt=event_dt,
            )
        )
    return runs


def parse_event_windows(spec: str) -> list[str]:
    windows = [token.strip() for token in spec.split(",") if token.strip()]
    if not windows:
        raise ValueError("No event windows provided")
    for token in windows:
        if token == "close:close":
            continue
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?[hd]:[+-]?\d+(?:\.\d+)?[hd]", token):
            continue
        raise ValueError(f"Unsupported event window: {token}")
    return windows


def _parse_offset(token: str) -> pd.Timedelta:
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([hd])", token)
    if match is None:
        raise ValueError(f"Invalid offset: {token}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "h":
        return pd.Timedelta(hours=value)
    return pd.Timedelta(days=value)


def load_prices(prices_csv: Path) -> pd.DataFrame:
    prices = pd.read_csv(prices_csv)
    required = {"timestamp", "symbol", "close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"prices csv missing required columns: {missing}")
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "symbol", "close"]).sort_values(
        ["symbol", "timestamp"]
    )
    if prices.empty:
        raise ValueError("prices csv has no valid rows after parsing")
    return prices


def _price_at_or_after(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    rows = df[df["timestamp"] >= ts]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return pd.Timestamp(row["timestamp"]), float(row["close"])


def _price_at_or_before(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    rows = df[df["timestamp"] <= ts]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return pd.Timestamp(row["timestamp"]), float(row["close"])


def compute_window_return(
    symbol_prices: pd.DataFrame, event_dt: pd.Timestamp, window: str
) -> tuple[float | None, pd.Timestamp | None, pd.Timestamp | None]:
    if window == "close:close":
        start_data = _price_at_or_before(symbol_prices, event_dt)
        if start_data is None:
            start_data = _price_at_or_after(symbol_prices, event_dt)
        if start_data is None:
            return None, None, None
        start_ts, start_price = start_data
        next_rows = symbol_prices[symbol_prices["timestamp"] > start_ts]
        if next_rows.empty or start_price == 0:
            return None, None, None
        end_ts = pd.Timestamp(next_rows.iloc[0]["timestamp"])
        end_price = float(next_rows.iloc[0]["close"])
        return (end_price / start_price) - 1.0, start_ts, end_ts

    start_token, end_token = window.split(":")
    start_dt = event_dt + _parse_offset(start_token)
    end_dt = event_dt + _parse_offset(end_token)
    start_data = _price_at_or_after(symbol_prices, start_dt)
    end_data = _price_at_or_after(symbol_prices, end_dt)
    if start_data is None or end_data is None:
        return None, None, None
    start_ts, start_price = start_data
    end_ts, end_price = end_data
    if start_price == 0:
        return None, None, None
    return (end_price / start_price) - 1.0, start_ts, end_ts


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def extract_signal_features(outputs_dir: Path) -> dict[str, float]:
    management = _read_csv(outputs_dir / "management_confidence.csv")
    analyst = _read_csv(outputs_dir / "analyst_pressure.csv")
    tone = _read_csv(outputs_dir / "tone_changes.csv")
    guidance = _read_csv(outputs_dir / "guidance.csv")
    guidance_revision = _read_csv(outputs_dir / "guidance_revision.csv")

    tone_change_count = float("nan")
    if not tone.empty:
        if "is_change" in tone.columns:
            tone_change_count = float(
                tone["is_change"].astype(str).str.lower().isin({"1", "true", "t", "yes"}).sum()
            )
        else:
            tone_change_count = float(len(tone))

    revision_counts: dict[str, float] = {
        "guidance_revision_matched_count": float("nan"),
        "guidance_revision_raised_count": float("nan"),
        "guidance_revision_lowered_count": float("nan"),
        "guidance_revision_reaffirmed_count": float("nan"),
        "guidance_revision_unclear_count": float("nan"),
        "guidance_revision_mixed_count": float("nan"),
    }
    if not guidance_revision.empty:
        labels = guidance_revision.get("revision_label", pd.Series([], dtype="object")).astype(
            str
        )
        matched = guidance_revision.get("is_matched", pd.Series([], dtype="object"))
        matched_mask = matched.astype(str).str.lower().isin({"1", "true", "t", "yes"})
        revision_counts = {
            "guidance_revision_matched_count": float(matched_mask.sum()),
            "guidance_revision_raised_count": float((labels == "raised").sum()),
            "guidance_revision_lowered_count": float((labels == "lowered").sum()),
            "guidance_revision_reaffirmed_count": float((labels == "reaffirmed").sum()),
            "guidance_revision_unclear_count": float((labels == "unclear").sum()),
            "guidance_revision_mixed_count": float((labels == "mixed").sum()),
        }

    features = {
        "management_confidence_mean": _safe_mean(management, "confidence_score"),
        "analyst_pressure_mean": _safe_mean(analyst, "pressure_score"),
        "tone_change_count": tone_change_count,
        "guidance_strength_mean": _safe_mean(guidance, "guidance_strength"),
        **revision_counts,
    }
    return features


def build_backtest_rows(
    runs: list[RunMeta], prices: pd.DataFrame, windows: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        symbol_prices = prices[prices["symbol"] == run.symbol].sort_values("timestamp")
        if symbol_prices.empty:
            print(f"[backtest] skipping {run.run_id}: no prices for symbol {run.symbol}")
            continue
        features = extract_signal_features(run.outputs_dir)
        for window in windows:
            realized, start_ts, end_ts = compute_window_return(
                symbol_prices, run.event_dt, window
            )
            row = {
                "run_id": run.run_id,
                "symbol": run.symbol,
                "event_dt": run.event_dt.isoformat(),
                "window": window,
                "window_start_ts": start_ts.isoformat() if start_ts is not None else None,
                "window_end_ts": end_ts.isoformat() if end_ts is not None else None,
                "return": realized,
                **features,
            }
            row["signal_features"] = json.dumps(
                {key: row.get(key) for key in FEATURE_COLUMNS}, sort_keys=True
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "symbol",
                "event_dt",
                "window",
                "window_start_ts",
                "window_end_ts",
                "return",
                *FEATURE_COLUMNS,
                "signal_features",
            ]
        )
    return pd.DataFrame(rows)


def bootstrap_mean_diff_ci(
    high_values: np.ndarray,
    low_values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float] | None:
    if len(high_values) < 2 or len(low_values) < 2:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        high_sample = rng.choice(high_values, size=len(high_values), replace=True)
        low_sample = rng.choice(low_values, size=len(low_values), replace=True)
        diffs[idx] = float(np.mean(high_sample) - np.mean(low_sample))
    lower = float(np.quantile(diffs, alpha / 2))
    upper = float(np.quantile(diffs, 1 - alpha / 2))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": lower,
        "ci_high": upper,
    }


def high_low_ttest(
    actual_returns: np.ndarray, scores: np.ndarray, *, quantile: float = 0.3
) -> dict[str, float] | None:
    if len(actual_returns) < 4:
        return None
    low_cut = float(np.quantile(scores, quantile))
    high_cut = float(np.quantile(scores, 1 - quantile))
    low_bucket = actual_returns[scores <= low_cut]
    high_bucket = actual_returns[scores >= high_cut]
    if len(low_bucket) < 2 or len(high_bucket) < 2:
        return None
    stat, pvalue = stats.ttest_ind(high_bucket, low_bucket, equal_var=False)
    return {
        "high_mean": float(np.mean(high_bucket)),
        "low_mean": float(np.mean(low_bucket)),
        "mean_diff": float(np.mean(high_bucket) - np.mean(low_bucket)),
        "t_stat": float(stat),
        "p_value": float(pvalue),
        "n_high": int(len(high_bucket)),
        "n_low": int(len(low_bucket)),
    }


def evaluate_window(df: pd.DataFrame) -> dict[str, Any]:
    clean = df.copy()
    clean["event_dt"] = pd.to_datetime(clean["event_dt"], errors="coerce")
    clean["return"] = pd.to_numeric(clean["return"], errors="coerce")
    clean = clean.dropna(subset=["event_dt", "return"]).sort_values("event_dt")
    if clean.empty:
        return {
            "n_rows": 0,
            "baseline": None,
            "model": None,
            "correlations": {},
            "regression_coefficients": {},
            "high_vs_low_ttest": None,
            "bootstrap_ci": None,
        }

    for col in FEATURE_COLUMNS:
        clean[col] = pd.to_numeric(clean.get(col, np.nan), errors="coerce")

    split_index = int(math.floor(len(clean) * 0.7))
    split_index = max(1, min(split_index, len(clean) - 1))
    train_df = clean.iloc[:split_index].copy()
    test_df = clean.iloc[split_index:].copy()

    fill_values = train_df[FEATURE_COLUMNS].mean(numeric_only=True).fillna(0.0)
    train_x = train_df[FEATURE_COLUMNS].fillna(fill_values).to_numpy(dtype=float)
    test_x = test_df[FEATURE_COLUMNS].fillna(fill_values).to_numpy(dtype=float)
    train_y = train_df["return"].to_numpy(dtype=float)
    test_y = test_df["return"].to_numpy(dtype=float)

    baseline_pred = np.full_like(test_y, fill_value=float(np.mean(train_y)))
    x_design = np.column_stack([np.ones(len(train_x)), train_x])
    coeffs, *_ = np.linalg.lstsq(x_design, train_y, rcond=None)
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    model_pred = test_design @ coeffs

    baseline_mse = float(np.mean((test_y - baseline_pred) ** 2))
    model_mse = float(np.mean((test_y - model_pred) ** 2))

    model_corr = None
    if len(test_y) > 1 and float(np.std(model_pred)) > 0 and float(np.std(test_y)) > 0:
        model_corr = float(np.corrcoef(test_y, model_pred)[0, 1])

    correlations: dict[str, dict[str, float | None]] = {}
    for feature in FEATURE_COLUMNS:
        series = clean[feature]
        valid = pd.DataFrame({"x": series, "y": clean["return"]}).dropna()
        if len(valid) < 3 or valid["x"].std(ddof=0) == 0 or valid["y"].std(ddof=0) == 0:
            correlations[feature] = {"corr": None, "p_value": None}
            continue
        corr, p_value = stats.pearsonr(valid["x"], valid["y"])
        correlations[feature] = {"corr": float(corr), "p_value": float(p_value)}

    ttest = high_low_ttest(test_y, model_pred, quantile=0.3)
    bootstrap = None
    if ttest is not None:
        low_cut = float(np.quantile(model_pred, 0.3))
        high_cut = float(np.quantile(model_pred, 0.7))
        low_vals = test_y[model_pred <= low_cut]
        high_vals = test_y[model_pred >= high_cut]
        bootstrap = bootstrap_mean_diff_ci(high_vals, low_vals, seed=42)

    return {
        "n_rows": int(len(clean)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "baseline": {
            "predictor": "train_mean_return",
            "test_mse": baseline_mse,
            "test_mae": float(np.mean(np.abs(test_y - baseline_pred))),
            "train_mean_return": float(np.mean(train_y)),
        },
        "model": {
            "predictor": "linear_regression_signals",
            "test_mse": model_mse,
            "test_mae": float(np.mean(np.abs(test_y - model_pred))),
            "test_corr": model_corr,
            "mse_improvement_vs_baseline": float(baseline_mse - model_mse),
        },
        "regression_coefficients": {
            "intercept": float(coeffs[0]),
            **{feature: float(value) for feature, value in zip(FEATURE_COLUMNS, coeffs[1:])},
        },
        "correlations": correlations,
        "high_vs_low_ttest": ttest,
        "bootstrap_ci": bootstrap,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Backtest Signals Report",
        "",
        f"- Runs analyzed: {summary.get('runs_analyzed', 0)}",
        f"- Event rows: {summary.get('event_rows', 0)}",
        "",
    ]
    window_summaries = summary.get("windows", {})
    if not isinstance(window_summaries, dict) or not window_summaries:
        lines.append("_No window summaries available._")
        return "\n".join(lines)

    for window, payload in window_summaries.items():
        lines.extend([f"## Window {window}", ""])
        lines.append(f"- Rows: {payload.get('n_rows', 0)}")
        model = payload.get("model")
        baseline = payload.get("baseline")
        if isinstance(model, dict) and isinstance(baseline, dict):
            lines.extend(
                [
                    f"- Baseline test MSE: {baseline.get('test_mse')}",
                    f"- Model test MSE: {model.get('test_mse')}",
                    f"- MSE improvement: {model.get('mse_improvement_vs_baseline')}",
                    f"- Model test correlation: {model.get('test_corr')}",
                ]
            )
        ttest = payload.get("high_vs_low_ttest")
        if isinstance(ttest, dict):
            lines.append(
                f"- High-vs-low t-test p-value: {ttest.get('p_value')} "
                f"(mean diff: {ttest.get('mean_diff')})"
            )
        lines.append("")
        lines.extend(
            [
                "| feature | corr | p_value |",
                "| --- | ---: | ---: |",
            ]
        )
        correlations = payload.get("correlations", {})
        if isinstance(correlations, dict) and correlations:
            for feature, vals in correlations.items():
                corr = vals.get("corr") if isinstance(vals, dict) else None
                pvalue = vals.get("p_value") if isinstance(vals, dict) else None
                lines.append(f"| {feature} | {corr} | {pvalue} |")
        else:
            lines.append("| _none_ |  |  |")
        lines.append("")
    return "\n".join(lines)


def run_backtest(
    *, runs_dir: Path, prices_csv: Path, event_windows: str, out_dir: Path
) -> tuple[Path, Path, Path]:
    windows = parse_event_windows(event_windows)
    prices = load_prices(prices_csv)
    runs = discover_runs(runs_dir)
    results_df = build_backtest_rows(runs, prices, windows)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "backtest_results.csv"
    summary_path = out_dir / "backtest_summary.json"
    report_path = out_dir / "backtest_report.md"

    results_df.to_csv(results_path, index=False)

    summary: dict[str, Any] = {
        "runs_analyzed": int(len(runs)),
        "event_rows": int(len(results_df)),
        "windows": {},
    }
    if not results_df.empty:
        for window, group in results_df.groupby("window"):
            summary["windows"][str(window)] = evaluate_window(group)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    return results_path, summary_path, report_path


def main() -> int:
    args = parse_args()
    results_path, summary_path, report_path = run_backtest(
        runs_dir=Path(args.runs_dir).expanduser().resolve(),
        prices_csv=Path(args.prices_csv).expanduser().resolve(),
        event_windows=str(args.event_window),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print(f"Wrote: {results_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
