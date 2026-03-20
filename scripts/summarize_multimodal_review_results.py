from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = REPO_ROOT / "data" / "media_support_eval" / "task_impact_eval_cases.csv"
RESULTS_FILE = REPO_ROOT / "data" / "media_support_eval" / "multimodal_review_results_codex_proto.csv"


def _split_citations(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(";") if part.strip()]


def _has_sidecar_citation(text: str) -> bool:
    citations = _split_citations(text)
    return any("/multimodal/" in item or "multimodal/" in item for item in citations)


def _summarize_case_agreement(group: pd.DataFrame) -> dict[str, float | int]:
    comparable_case_count = 0
    modal_rates: list[float] = []
    for _, case_group in group.groupby("case_id"):
        labels = case_group["predicted_guidance_label"].astype(str).str.strip()
        labels = labels[labels != ""]
        if len(labels) < 2:
            continue
        comparable_case_count += 1
        modal_rate = float(labels.value_counts(normalize=True).iloc[0])
        modal_rates.append(modal_rate)
    return {
        "comparable_case_count": comparable_case_count,
        "mean_case_modal_agreement": round(sum(modal_rates) / len(modal_rates), 4) if modal_rates else 0.0,
    }


def summarize_results(cases: pd.DataFrame, results: pd.DataFrame) -> dict[str, object]:
    if results.empty:
        return {
            "submission_count": 0,
            "conditions": {},
            "notes": [
                "No reviewer submissions were recorded yet.",
                "This scaffold is prototype-level and descriptive only.",
            ],
        }

    merged = results.merge(
        cases[["case_id", "gold_guidance_label"]],
        on="case_id",
        how="left",
    )
    merged["label_match"] = (
        merged["predicted_guidance_label"].astype(str).str.strip()
        == merged["gold_guidance_label"].astype(str).str.strip()
    )
    merged["completion_seconds_num"] = pd.to_numeric(merged["completion_seconds"], errors="coerce")
    merged["clarity_rating_num"] = pd.to_numeric(merged["clarity_rating"], errors="coerce")
    merged["evidence_traceability_rating_num"] = pd.to_numeric(
        merged["evidence_traceability_rating"], errors="coerce"
    )
    merged["artifact_citation_count"] = merged["cited_artifact_paths"].astype(str).apply(
        lambda value: len(_split_citations(value))
    )
    merged["has_sidecar_citation"] = merged["cited_artifact_paths"].astype(str).apply(_has_sidecar_citation)

    conditions: dict[str, object] = {}
    for condition, group in merged.groupby("condition"):
        agreement = _summarize_case_agreement(group)
        conditions[str(condition)] = {
            "submission_count": int(len(group)),
            "mean_completion_seconds": round(float(group["completion_seconds_num"].mean()), 2),
            "label_accuracy": round(float(group["label_match"].mean()), 4),
            "mean_clarity_rating": round(float(group["clarity_rating_num"].mean()), 2),
            "mean_evidence_traceability_rating": round(
                float(group["evidence_traceability_rating_num"].mean()),
                2,
            ),
            "mean_artifact_citation_count": round(float(group["artifact_citation_count"].mean()), 2),
            "sidecar_citation_rate": round(float(group["has_sidecar_citation"].mean()), 4),
            **agreement,
        }

    return {
        "submission_count": int(len(merged)),
        "conditions": conditions,
        "notes": [
            "This summary is descriptive only until enough counterbalanced reviewer observations exist.",
            "Use it to inspect speed, consistency, and evidence-traceability patterns rather than to claim statistical significance.",
        ],
    }


def main() -> None:
    cases = pd.read_csv(CASES_FILE, dtype=str).fillna("")
    results = pd.read_csv(RESULTS_FILE, dtype=str).fillna("") if RESULTS_FILE.exists() else pd.DataFrame()
    summary = summarize_results(cases, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
