from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from earnings_call_sentiment.media_support_eval import load_labeled_feature_frame, repo_root
from earnings_call_sentiment.media_support_models import MODEL_ROOT, TASK_CONFIG


def _estimator(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ]
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _evaluate_task(frame: pd.DataFrame, *, feature_columns: list[str], label_column: str) -> dict[str, Any]:
    labels = frame[label_column].astype(str)
    groups = frame["source_call_id"].astype(str)
    unique_groups = sorted(groups.unique().tolist())
    class_counts = {str(key): int(value) for key, value in labels.value_counts().to_dict().items()}

    if len(unique_groups) < 2 or len(class_counts) < 2:
        return {
            "status": "skipped",
            "reason": "insufficient_unique_groups_or_classes",
            "row_count": int(len(frame)),
            "group_count": int(len(unique_groups)),
            "class_counts": class_counts,
        }

    work = frame.copy()
    for column in feature_columns:
        if column not in work.columns:
            work[column] = 0.0
    X = work[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = labels
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, str]] = []
    for train_idx, test_idx in logo.split(X, y, groups):
        y_train = y.iloc[train_idx]
        if len(set(y_train.tolist())) < len(class_counts):
            continue
        estimator = _estimator(feature_columns)
        estimator.fit(X.iloc[train_idx], y_train)
        predictions = estimator.predict(X.iloc[test_idx])
        for truth, predicted, source_call_id in zip(y.iloc[test_idx], predictions, groups.iloc[test_idx]):
            fold_rows.append(
                {
                    "y_true": str(truth),
                    "y_pred": str(predicted),
                    "source_call_id": str(source_call_id),
                }
            )

    if not fold_rows:
        return {
            "status": "skipped",
            "reason": "group_holdout_failed_due_to_class_sparsity",
            "row_count": int(len(frame)),
            "group_count": int(len(unique_groups)),
            "class_counts": class_counts,
        }

    eval_df = pd.DataFrame(fold_rows)
    metrics = {
        "accuracy": round(float(accuracy_score(eval_df["y_true"], eval_df["y_pred"])), 4),
        "macro_f1": round(float(f1_score(eval_df["y_true"], eval_df["y_pred"], average="macro")), 4),
        "macro_precision": round(float(precision_score(eval_df["y_true"], eval_df["y_pred"], average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(eval_df["y_true"], eval_df["y_pred"], average="macro", zero_division=0)), 4),
    }

    calibration_mode = "not_available"
    final_estimator: Any = _estimator(feature_columns)
    if len(unique_groups) >= 3 and min(class_counts.values()) >= 3:
        calibration_mode = "sigmoid_cv3"
        final_estimator = CalibratedClassifierCV(
            estimator=_estimator(feature_columns),
            cv=3,
            method="sigmoid",
        )
    final_estimator.fit(X, y)

    reliability_weight = round(max(0.25, min(0.75, metrics["macro_f1"])), 4)
    return {
        "status": "ok",
        "row_count": int(len(frame)),
        "group_count": int(len(unique_groups)),
        "class_counts": class_counts,
        "metrics": metrics,
        "calibration_mode": calibration_mode,
        "reliability_weight": reliability_weight,
        "model": final_estimator,
        "evaluation_rows": eval_df.to_dict(orient="records"),
    }


def _write_task_outputs(task_name: str, result: dict[str, Any], *, feature_columns: list[str]) -> dict[str, Any]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = repo_root() / "outputs" / "media_model_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        key: value
        for key, value in result.items()
        if key not in {"model"}
    }
    summary_payload["feature_columns"] = feature_columns

    summary_path = output_dir / f"{task_name}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if result.get("status") == "ok":
        model_path = MODEL_ROOT / f"{task_name}.joblib"
        metadata_path = MODEL_ROOT / f"{task_name}_metadata.json"
        joblib.dump(result["model"], model_path)
        metadata_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        summary_payload["model_path"] = str(model_path)
        summary_payload["metadata_path"] = str(metadata_path)
    return summary_payload


def main() -> None:
    overall_summary: dict[str, Any] = {"tasks": {}}
    for task_name, config in TASK_CONFIG.items():
        frame = load_labeled_feature_frame(
            feature_modality=str(config["modality"]),
            label_column=str(config["label_column"]),
        )
        if frame.empty:
            overall_summary["tasks"][task_name] = {
                "status": "skipped",
                "reason": "no_labeled_rows",
            }
            continue
        feature_columns = list(config["feature_columns"])
        result = _evaluate_task(
            frame,
            feature_columns=feature_columns,
            label_column=str(config["label_column"]),
        )
        overall_summary["tasks"][task_name] = _write_task_outputs(
            task_name,
            result,
            feature_columns=feature_columns,
        )

    output_dir = repo_root() / "outputs" / "media_model_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = output_dir / "training_summary.json"
    overall_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(json.dumps(overall_summary, indent=2))
    print(f"wrote {overall_path}")


if __name__ == "__main__":
    main()
