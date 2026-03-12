from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.signals.behavior import compute_behavioral_outputs
from summarize_behavior_eval_set import EVAL_ROOT, FILES, REPO_ROOT, _validate_file


OUTPUT_ROOT = REPO_ROOT / "outputs" / "behavior_eval"


def _single_chunk_frame(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": 0.0,
                "end": 5.0,
                "text": text,
                "sentiment": "NEUTRAL",
                "score": 0.0,
                "signed_score": 0.0,
            }
        ]
    )


def _predict_uncertainty(payload: dict[str, Any]) -> tuple[str, str, str]:
    df = payload["uncertainty_df"]
    if df.empty:
        return "absent", "", "no uncertainty signal matched"
    level = str(payload["summary"]["uncertainty_score_overall"]["level"])
    top_row = df.sort_values(["strength", "start_char"], ascending=[False, True]).iloc[0]
    label = "strong" if level == "high" else "present"
    return label, str(top_row["matched_phrase"]), f"summary_level={level}"


def _predict_reassurance(payload: dict[str, Any]) -> tuple[str, str, str]:
    df = payload["reassurance_df"]
    if df.empty:
        return "absent", "", "no reassurance signal matched"
    top_row = df.sort_values(["strength", "start_char"], ascending=[False, True]).iloc[0]
    return "present", str(top_row["matched_phrase"]), "reassurance signal matched"


def _predict_skepticism(payload: dict[str, Any]) -> tuple[str, str, str]:
    df = payload["skepticism_df"]
    if df.empty:
        return "none", "", "no skepticism signal matched"
    top_row = df.sort_values(["strength"], ascending=[False]).iloc[0]
    return str(top_row["skepticism_label"]), str(top_row["matched_phrase"]), "skepticism label from deterministic cues"


def _predict_label(family: str, text: str) -> tuple[str, str, str]:
    payload = compute_behavioral_outputs(_single_chunk_frame(text))
    if family == "uncertainty":
        return _predict_uncertainty(payload)
    if family == "reassurance":
        return _predict_reassurance(payload)
    if family == "skepticism":
        return _predict_skepticism(payload)
    raise ValueError(f"unsupported family: {family}")


def _family_output_path(family: str) -> Path:
    return EVAL_ROOT / f"{family}_labels.csv"


def _evaluate_family(family: str) -> list[dict[str, Any]]:
    path = _family_output_path(family)
    _validate_file(family, path, FILES[family]["allowed"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    predictions: list[dict[str, Any]] = []
    for row in rows:
        predicted_label, matched_phrase, prediction_note = _predict_label(family, row["text"])
        predictions.append(
            {
                "item_id": row["item_id"],
                "signal_family": family,
                "source_package": row["source_package"],
                "source_call_id": row["source_call_id"],
                "ticker": row["ticker"],
                "company": row["company"],
                "gold_label": row["label"],
                "predicted_label": predicted_label,
                "match_bool": row["label"] == predicted_label,
                "matched_phrase": matched_phrase,
                "text": row["text"],
                "source_path": row["source_path"],
                "notes": row["notes"],
                "prediction_note": prediction_note,
            }
        )
    return predictions


def _summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_family[row["signal_family"]].append(row)

    family_summaries: dict[str, Any] = {}
    total_rows = len(predictions)
    total_matches = 0
    mismatch_rows: list[dict[str, Any]] = []
    for family, rows in by_family.items():
        gold_counts = Counter(row["gold_label"] for row in rows)
        pred_counts = Counter(row["predicted_label"] for row in rows)
        matches = sum(1 for row in rows if row["match_bool"])
        total_matches += matches
        mismatch_rows.extend([row for row in rows if not row["match_bool"]])
        family_summaries[family] = {
            "row_count": len(rows),
            "matches": matches,
            "agreement": matches / len(rows) if rows else 0.0,
            "gold_distribution": dict(sorted(gold_counts.items())),
            "predicted_distribution": dict(sorted(pred_counts.items())),
        }

    return {
        "row_count": total_rows,
        "matches": total_matches,
        "overall_agreement": total_matches / total_rows if total_rows else 0.0,
        "family_summaries": family_summaries,
        "mismatch_rows": mismatch_rows,
        "notes": [
            "This is a tiny internal eval set for deterministic Phase 1 behavior signals.",
            "Predictions are made by running the current deterministic behavior rules on each gold span in isolation.",
            "Uncertainty mapping is explicit: no match -> absent, low/medium -> present, high -> strong.",
        ],
    }


def _write_predictions(predictions: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "behavior_eval_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    print(f"wrote {out_path}")


def _write_summary_json(summary: dict[str, Any]) -> None:
    out_path = OUTPUT_ROOT / "behavior_eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def _write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Behavior Eval Summary",
        "",
        f"- rows: {summary['row_count']}",
        f"- matches: {summary['matches']}",
        f"- overall agreement: {summary['overall_agreement']:.1%}",
        "",
        "## Family Agreement",
    ]
    for family in ("uncertainty", "reassurance", "skepticism"):
        family_summary = summary["family_summaries"].get(family, {})
        lines.extend(
            [
                f"### {family}",
                f"- rows: {family_summary.get('row_count', 0)}",
                f"- agreement: {family_summary.get('agreement', 0.0):.1%}",
                f"- gold distribution: {family_summary.get('gold_distribution', {})}",
                f"- predicted distribution: {family_summary.get('predicted_distribution', {})}",
                "",
            ]
        )

    lines.extend(["## Mismatches", ""])
    mismatches = summary["mismatch_rows"]
    if not mismatches:
        lines.append("- none")
    else:
        for row in mismatches:
            lines.append(
                f"- {row['item_id']} ({row['signal_family']}): gold={row['gold_label']} predicted={row['predicted_label']} | {row['matched_phrase'] or 'no match'}"
            )
            lines.append(f"  - text: {row['text']}")

    lines.extend(["", "## Notes"])
    for note in summary["notes"]:
        lines.append(f"- {note}")

    out_path = OUTPUT_ROOT / "behavior_eval_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> int:
    predictions: list[dict[str, Any]] = []
    for family in ("uncertainty", "reassurance", "skepticism"):
        predictions.extend(_evaluate_family(family))

    summary = _summarize(predictions)
    _write_predictions(predictions)
    _write_summary_json(summary)
    _write_summary_md(summary)
    print(f"rows={summary['row_count']}")
    print(f"matches={summary['matches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
