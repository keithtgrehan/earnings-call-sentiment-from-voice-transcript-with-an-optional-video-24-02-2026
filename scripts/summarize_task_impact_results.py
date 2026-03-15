from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from earnings_call_sentiment.media_support_comparison import summarize_task_impact_results
from earnings_call_sentiment.media_support_eval import repo_root


CASES_FILE = Path("data/media_support_eval/task_impact_eval_cases.csv")
RESULTS_FILE = Path("data/media_support_eval/task_impact_results_template.csv")


def main() -> None:
    root = repo_root()
    cases = pd.read_csv(root / CASES_FILE, dtype=str).fillna("")
    results_path = root / RESULTS_FILE
    results = pd.read_csv(results_path, dtype=str).fillna("") if results_path.exists() else pd.DataFrame()
    summary = summarize_task_impact_results(cases, results)

    output_dir = root / "outputs" / "media_support_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "task_impact_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
