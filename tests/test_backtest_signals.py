from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _load_backtest_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "backtest_signals.py"
    spec = importlib.util.spec_from_file_location("backtest_signals", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(
    runs_dir: Path,
    *,
    run_name: str,
    symbol: str,
    event_dt: str,
    confidence: float,
    pressure: float,
    tone_count: int,
    guidance_strength: float,
    revision_label: str,
) -> None:
    run_dir = runs_dir / run_name
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    (run_dir / "event.json").write_text(
        json.dumps({"symbol": symbol, "event_dt": event_dt}),
        encoding="utf-8",
    )
    pd.DataFrame([{"confidence_score": confidence}]).to_csv(
        outputs / "management_confidence.csv", index=False
    )
    pd.DataFrame([{"pressure_score": pressure}]).to_csv(
        outputs / "analyst_pressure.csv", index=False
    )
    tone_rows = [{"is_change": True}] * max(0, tone_count)
    if not tone_rows:
        tone_rows = [{"is_change": False}]
    pd.DataFrame(tone_rows).to_csv(outputs / "tone_changes.csv", index=False)
    pd.DataFrame([{"guidance_strength": guidance_strength}]).to_csv(
        outputs / "guidance.csv", index=False
    )
    pd.DataFrame(
        [{"is_matched": True, "revision_label": revision_label}]
    ).to_csv(outputs / "guidance_revision.csv", index=False)
    (outputs / "metrics.json").write_text("{}", encoding="utf-8")
    (outputs / "report.md").write_text("report", encoding="utf-8")


def test_backtest_stats_helpers() -> None:
    module = _load_backtest_module()

    # Strongly separated buckets should show a statistically significant difference.
    low = np.array([-0.08, -0.07, -0.06, -0.09, -0.05, -0.07], dtype=float)
    high = np.array([0.07, 0.09, 0.1, 0.08, 0.06, 0.11], dtype=float)
    actual = np.concatenate([low, high])
    scores = np.concatenate([np.zeros_like(low), np.ones_like(high)])
    ttest = module.high_low_ttest(actual, scores, quantile=0.5)
    assert ttest is not None
    assert float(ttest["mean_diff"]) > 0
    assert float(ttest["p_value"]) < 0.05

    ci = module.bootstrap_mean_diff_ci(high, low, n_boot=300, seed=42)
    assert ci is not None
    assert float(ci["ci_low"]) < float(ci["mean_diff"]) < float(ci["ci_high"])


def test_backtest_script_generates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "backtest_signals.py"

    runs_dir = tmp_path / "runs"
    prices_csv = tmp_path / "prices.csv"
    out_dir = tmp_path / "backtest_out"
    base_dt = pd.Timestamp("2024-01-01T09:00:00")

    price_rows: list[dict[str, object]] = []
    for idx in range(8):
        event_dt = base_dt + pd.Timedelta(days=idx)
        confidence = 0.2 + (idx * 0.08)
        pressure = 1.0 - confidence
        guidance_strength = confidence
        one_hour_return = -0.04 + (idx * 0.015)

        _write_run(
            runs_dir,
            run_name=f"AAA_{event_dt.strftime('%Y-%m-%dT%H-%M-%S')}",
            symbol="AAA",
            event_dt=event_dt.isoformat(),
            confidence=confidence,
            pressure=pressure,
            tone_count=idx % 3,
            guidance_strength=guidance_strength,
            revision_label="raised" if idx >= 4 else "lowered",
        )

        start_close = 100.0
        end_close = start_close * (1 + one_hour_return)
        price_rows.append(
            {
                "timestamp": event_dt.isoformat(),
                "symbol": "AAA",
                "open": start_close,
                "high": start_close,
                "low": start_close,
                "close": start_close,
                "volume": 1_000_000,
            }
        )
        price_rows.append(
            {
                "timestamp": (event_dt + pd.Timedelta(hours=1)).isoformat(),
                "symbol": "AAA",
                "open": end_close,
                "high": end_close,
                "low": end_close,
                "close": end_close,
                "volume": 1_000_000,
            }
        )

    pd.DataFrame(price_rows).to_csv(prices_csv, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--runs-dir",
            str(runs_dir),
            "--prices-csv",
            str(prices_csv),
            "--event-window",
            "0h:1h",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    results_path = out_dir / "backtest_results.csv"
    summary_path = out_dir / "backtest_summary.json"
    report_path = out_dir / "backtest_report.md"
    assert results_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert results_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    assert report_path.stat().st_size > 0

    results_df = pd.read_csv(results_path)
    assert len(results_df) == 8
    assert {"symbol", "event_dt", "window", "return", "signal_features"}.issubset(
        results_df.columns
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "windows" in summary
    assert "0h:1h" in summary["windows"]
