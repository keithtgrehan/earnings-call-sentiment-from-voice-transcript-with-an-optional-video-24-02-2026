from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from earnings_call_sentiment.media_support_eval import (
    build_visual_trainability_report,
    load_runtime_smoke_manifest,
    repo_root,
)


TASK_IMPACT_CASES = Path("data/media_support_eval/task_impact_eval_cases.csv")
TASK_IMPACT_RESULTS = Path("data/media_support_eval/task_impact_results_template.csv")
DOWNSTREAM_CASES = Path("data/media_support_eval/downstream_decision_eval_cases.csv")


def _runtime_breadth_summary() -> dict[str, object]:
    runtime_smoke = load_runtime_smoke_manifest()
    if runtime_smoke.empty:
        return {
            "runtime_smoke_rows": 0,
            "successful_runs": 0,
            "video_quality_ok_rows": 0,
            "face_visibility_breakdown": {},
        }

    return {
        "runtime_smoke_rows": int(len(runtime_smoke)),
        "successful_runs": int((runtime_smoke["runtime_success"].astype(str).str.lower() == "true").sum()),
        "audio_quality_ok_rows": int((runtime_smoke["audio_quality_ok"].astype(str).str.lower() == "true").sum()),
        "video_quality_ok_rows": int((runtime_smoke["video_quality_ok"].astype(str).str.lower() == "true").sum()),
        "face_visibility_breakdown": {
            str(key): int(value)
            for key, value in runtime_smoke["face_visibility_outcome"].astype(str).value_counts().to_dict().items()
        },
    }


def _task_impact_readiness(root: Path) -> dict[str, object]:
    cases_path = root / TASK_IMPACT_CASES
    results_path = root / TASK_IMPACT_RESULTS
    if not cases_path.exists():
        return {
            "case_rows": 0,
            "results_template_present": results_path.exists(),
            "guidance_label_distribution": {},
        }

    cases = pd.read_csv(cases_path, dtype=str).fillna("")
    return {
        "case_rows": int(len(cases)),
        "results_template_present": results_path.exists(),
        "guidance_label_distribution": {
            str(key): int(value)
            for key, value in cases["gold_guidance_label"].astype(str).value_counts().to_dict().items()
        },
    }


def _downstream_case_summary(root: Path) -> dict[str, object]:
    cases_path = root / DOWNSTREAM_CASES
    if not cases_path.exists():
        return {
            "case_rows": 0,
            "support_target_rows": 0,
        }
    cases = pd.read_csv(cases_path, dtype=str).fillna("")
    support_target_rows = int(cases["target_support_direction"].astype(str).str.strip().ne("").sum())
    return {
        "case_rows": int(len(cases)),
        "support_target_rows": support_target_rows,
    }


def _significance_readiness(task_case_rows: int, downstream_case_rows: int, visual_groups: int) -> dict[str, object]:
    return {
        "supportable_now": False,
        "current_independent_downstream_cases": downstream_case_rows,
        "current_gold_aligned_task_cases": task_case_rows,
        "current_visual_training_groups": visual_groups,
        "recommended_tests_if_data_expands": {
            "downstream_support_direction": "Exact sign test or McNemar-style paired test on independent source-call cases.",
            "human_task_time": "Wilcoxon signed-rank or paired t-test only after a counterbalanced within-subject setup.",
            "human_task_label_accuracy": "McNemar test only after each participant sees matched baseline/treatment cases.",
        },
        "minimum_next_data": {
            "downstream_support_direction_cases": "At least 20 independent source-call cases with a fixed paired baseline vs assisted comparison.",
            "human_task_pilot": "At least 12 participants with 4-5 counterbalanced cases each, plus pre-registered scoring rules.",
            "visual_model_grouped_eval": "At least 3 source groups with nonblank visual_tension labels; 10+ groups is a more credible next capstone target.",
        },
    }


def main() -> None:
    root = repo_root()
    visual = build_visual_trainability_report()
    runtime = _runtime_breadth_summary()
    task_impact = _task_impact_readiness(root)
    downstream = _downstream_case_summary(root)
    summary = {
        "visual_trainability": visual,
        "runtime_breadth": runtime,
        "downstream_cases": downstream,
        "task_impact_readiness": task_impact,
        "significance_readiness": _significance_readiness(
            task_case_rows=int(task_impact["case_rows"]),
            downstream_case_rows=int(downstream["case_rows"]),
            visual_groups=int(visual["source_groups_with_visual_tension_labels"]),
        ),
    }

    output_dir = root / "outputs" / "media_support_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "media_support_readiness.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
