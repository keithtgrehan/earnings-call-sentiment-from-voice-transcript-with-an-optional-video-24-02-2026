from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from earnings_call_sentiment.media_support_comparison import evaluate_downstream_decision_cases
from earnings_call_sentiment.media_support_eval import repo_root


CASES_FILE = Path("data/media_support_eval/downstream_decision_eval_cases.csv")


def main() -> None:
    root = repo_root()
    cases_path = root / CASES_FILE
    cases = pd.read_csv(cases_path, dtype=str).fillna("")
    result_frame, summary = evaluate_downstream_decision_cases(cases)

    output_dir = root / "outputs" / "media_support_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "downstream_decision_comparison.json"
    rows_path = output_dir / "downstream_decision_comparison_rows.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result_frame.to_csv(rows_path, index=False)

    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
