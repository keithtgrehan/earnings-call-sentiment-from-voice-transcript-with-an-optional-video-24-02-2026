from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.media_support_comparison import aggregate_support_target
from earnings_call_sentiment.media_support_eval import load_media_manifest, load_segment_labels, repo_root


GUIDANCE_LABEL_PACKAGES: tuple[tuple[str, Path], ...] = (
    ("gold_guidance_calls", Path("data/gold_guidance_calls/labels.csv")),
    ("gold_guidance_calls_holdout", Path("data/gold_guidance_calls_holdout/labels.csv")),
    ("gold_guidance_calls_holdout_watchlist", Path("data/gold_guidance_calls_holdout_watchlist/labels.csv")),
)

DOWNSTREAM_CASE_COLUMNS = [
    "case_id",
    "source_id",
    "call_id",
    "source_package",
    "source_call_id",
    "ticker",
    "company",
    "quarter",
    "event_date",
    "gold_guidance_label",
    "source_path",
    "metrics_path",
    "qa_shift_path",
    "audio_summary_path",
    "visual_summary_path",
    "saved_multimodal_summary_path",
    "target_support_direction",
    "target_support_signed_mean",
    "label_supportive_count",
    "label_cautionary_count",
    "label_neutral_count",
    "label_unavailable_count",
    "notes",
]

TASK_IMPACT_CASE_COLUMNS = [
    "case_id",
    "call_id",
    "source_call_id",
    "ticker",
    "company",
    "gold_guidance_label",
    "baseline_transcript_path",
    "treatment_report_path",
    "treatment_metrics_path",
    "treatment_transcript_path",
    "notes",
]

TASK_IMPACT_RESULTS_COLUMNS = [
    "participant_id",
    "case_id",
    "condition",
    "completion_seconds",
    "predicted_guidance_label",
    "evidence_text",
    "summary_text",
    "clarity_rating",
    "evidence_quality_score",
    "notes",
]

TASK_IMPACT_ASSIGNMENT_COLUMNS = [
    "participant_id",
    "case_id",
    "condition",
    "order_index",
    "status",
    "notes",
]

DEFAULT_SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SENTIMENT_MODEL_REVISION = "714eb0f"
PLACEHOLDER_NOTE = (
    "Lightweight transcript-only placeholder bundle generated because the full review workflow could not be rerun "
    "from the currently available local assets in this checkout."
)


@dataclass(frozen=True)
class CaseOutputPaths:
    output_dir: Path
    cache_dir: Path
    metrics_path: Path
    qa_shift_path: Path
    report_path: Path
    transcript_path: Path
    audio_summary_path: Path
    visual_summary_path: Path
    multimodal_summary_path: Path


def _root(root: Path | None = None) -> Path:
    return repo_root() if root is None else Path(root).expanduser().resolve()


def _repo_relative_path(value: str | Path, *, root: Path | None = None) -> str:
    resolved_root = _root(root)
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(resolved_root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(resolved_root.resolve()).as_posix()
        except ValueError:
            return str(path)


def derive_source_call_id(source_path: str) -> str:
    return Path(str(source_path)).stem


def load_guidance_case_rows(root: Path | None = None) -> pd.DataFrame:
    resolved_root = _root(root)
    rows: list[dict[str, str]] = []
    for source_package, rel_path in GUIDANCE_LABEL_PACKAGES:
        frame = pd.read_csv(resolved_root / rel_path, dtype=str).fillna("")
        frame["source_package"] = source_package
        frame["source_call_id"] = frame["source_path"].map(derive_source_call_id)
        rows.extend(frame.to_dict(orient="records"))
    result = pd.DataFrame(rows)
    return result.sort_values(["event_date", "ticker", "call_id"]).reset_index(drop=True)


def build_support_target_table(root: Path | None = None) -> dict[str, dict[str, Any]]:
    labels = load_segment_labels()
    by_call: dict[str, dict[str, Any]] = {}
    for source_call_id, group in labels.groupby("source_call_id"):
        aggregate = aggregate_support_target(group)
        counts = aggregate["counts"]
        by_call[str(source_call_id)] = {
            "target_support_direction": aggregate["target_support_direction"],
            "target_support_signed_mean": aggregate["target_support_signed_mean"],
            "label_supportive_count": int(counts.get("supportive", 0)),
            "label_cautionary_count": int(counts.get("cautionary", 0)),
            "label_neutral_count": int(counts.get("neutral", 0)),
            "label_unavailable_count": int(counts.get("unavailable", 0)),
        }
    return by_call


def _existing_source_id_map() -> dict[str, str]:
    manifest = load_media_manifest()
    mapping: dict[str, str] = {}
    for _, row in manifest.iterrows():
        source_call_id = str(row.get("source_call_id", "")).strip()
        source_id = str(row.get("source_id", "")).strip()
        if source_call_id and source_id and source_call_id not in mapping:
            mapping[source_call_id] = source_id
    return mapping


def case_output_paths(
    source_call_id: str,
    *,
    root: Path | None = None,
) -> CaseOutputPaths:
    resolved_root = _root(root)
    direct_dir = resolved_root / "outputs" / source_call_id
    output_dir = direct_dir if (direct_dir / "metrics.json").exists() else resolved_root / "outputs" / "downstream_decision_eval" / source_call_id
    cache_dir = resolved_root / "cache" / "downstream_decision_eval" / source_call_id
    return CaseOutputPaths(
        output_dir=output_dir,
        cache_dir=cache_dir,
        metrics_path=output_dir / "metrics.json",
        qa_shift_path=output_dir / "qa_shift_summary.json",
        report_path=output_dir / "report.md",
        transcript_path=output_dir / "transcript.txt",
        audio_summary_path=output_dir / "audio_behavior_summary.json",
        visual_summary_path=output_dir / "visual_behavior_summary.json",
        multimodal_summary_path=output_dir / "multimodal_support_summary.json",
    )


def build_downstream_case_frame(root: Path | None = None) -> pd.DataFrame:
    cases = load_guidance_case_rows(root)
    source_id_map = _existing_source_id_map()
    support_targets = build_support_target_table(root)
    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        source_call_id = str(case["source_call_id"])
        output_paths = case_output_paths(source_call_id, root=root)
        support_target = support_targets.get(source_call_id, {})
        has_support_target = bool(support_target)
        uses_existing_run = "outputs" in str(output_paths.output_dir) and output_paths.output_dir.parent.name != "downstream_decision_eval"
        notes = [
            f"{case['source_package']} labeled guidance case.",
            "Reuses an existing repo-local artifact bundle."
            if uses_existing_run
            else "Stable document-review bundle can be regenerated from the committed raw transcript/excerpt.",
        ]
        if has_support_target:
            notes.append("Source-level support target is derived from segment_labels.csv for this call.")
        else:
            notes.append("No source-level media-support target labels exist yet, so support-direction scoring stays unscored for this row.")
        rows.append(
            {
                "case_id": str(case["call_id"]),
                "source_id": source_id_map.get(source_call_id, ""),
                "call_id": str(case["call_id"]),
                "source_package": str(case["source_package"]),
                "source_call_id": source_call_id,
                "ticker": str(case["ticker"]),
                "company": str(case["company"]),
                "quarter": str(case["quarter"]),
                "event_date": str(case["event_date"]),
                "gold_guidance_label": str(case["guidance_change_label"]),
                "source_path": _repo_relative_path(str(case["source_path"]), root=root),
                "metrics_path": _repo_relative_path(output_paths.metrics_path, root=root),
                "qa_shift_path": _repo_relative_path(output_paths.qa_shift_path, root=root),
                "audio_summary_path": _repo_relative_path(output_paths.audio_summary_path, root=root)
                if output_paths.audio_summary_path.exists()
                else "",
                "visual_summary_path": _repo_relative_path(output_paths.visual_summary_path, root=root)
                if output_paths.visual_summary_path.exists()
                else "",
                "saved_multimodal_summary_path": _repo_relative_path(output_paths.multimodal_summary_path, root=root)
                if output_paths.multimodal_summary_path.exists()
                else "",
                "target_support_direction": support_target.get("target_support_direction", ""),
                "target_support_signed_mean": support_target.get("target_support_signed_mean", ""),
                "label_supportive_count": support_target.get("label_supportive_count", ""),
                "label_cautionary_count": support_target.get("label_cautionary_count", ""),
                "label_neutral_count": support_target.get("label_neutral_count", ""),
                "label_unavailable_count": support_target.get("label_unavailable_count", ""),
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=DOWNSTREAM_CASE_COLUMNS)


def build_task_impact_case_frame(root: Path | None = None) -> pd.DataFrame:
    cases = load_guidance_case_rows(root)
    rows: list[dict[str, str]] = []
    for _, case in cases.iterrows():
        source_call_id = str(case["source_call_id"])
        output_paths = case_output_paths(source_call_id, root=root)
        notes = (
            f"{case['source_package']} labeled guidance case. "
            "Control uses the committed transcript/excerpt. Treatment uses the repo-local deterministic report bundle "
            "when available, otherwise a verified transcript-only placeholder bundle."
        )
        rows.append(
            {
                "case_id": str(case["call_id"]),
                "call_id": str(case["call_id"]),
                "source_call_id": source_call_id,
                "ticker": str(case["ticker"]),
                "company": str(case["company"]),
                "gold_guidance_label": str(case["guidance_change_label"]),
                "baseline_transcript_path": _repo_relative_path(str(case["source_path"]), root=root),
                "treatment_report_path": _repo_relative_path(output_paths.report_path, root=root),
                "treatment_metrics_path": _repo_relative_path(output_paths.metrics_path, root=root),
                "treatment_transcript_path": _repo_relative_path(output_paths.transcript_path, root=root),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=TASK_IMPACT_CASE_COLUMNS)


def write_case_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_header_only_csv(path: Path, columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
    return path


def _event_dt_iso(event_date: str) -> str:
    text = str(event_date).strip()
    return f"{text}T00:00:00+00:00" if text else "1970-01-01T00:00:00+00:00"


def _placeholder_transcript_segments(text: str) -> list[dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": float(max(5, min(len(text.split()), 120))),
            "text": text.strip(),
        }
    ]


def _write_placeholder_bundle(case: pd.Series, output_paths: CaseOutputPaths, text: str) -> None:
    event_date = str(case.get("event_date", "")).strip()
    source_package = str(case.get("source_package", "task_impact_casepack")).strip() or "task_impact_casepack"
    source_path_value = str(case.get("source_path", "") or case.get("baseline_transcript_path", "")).strip()
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    (output_paths.output_dir / "inputs").mkdir(parents=True, exist_ok=True)

    transcript_segments = _placeholder_transcript_segments(text)
    transcript_json_path = output_paths.output_dir / "transcript.json"
    transcript_json_path.write_text(json.dumps(transcript_segments, indent=2), encoding="utf-8")
    output_paths.transcript_path.write_text(text.strip() + "\n", encoding="utf-8")

    metrics_payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "placeholder_bundle": True,
        "placeholder_reason": PLACEHOLDER_NOTE,
        "overall_review_signal": "amber",
        "review_confidence_pct": 35,
        "guidance": {"row_count": 0, "mean_strength": 0.0},
        "tone_changes": {"row_count": 0, "change_count": 0},
        "behavioral_signals": {
            "uncertainty": {"score": 0, "level": "low"},
            "reassurance": {"score": 0, "level": "low"},
            "analyst_skepticism": {"score": 0, "level": "low"},
        },
        "guidance_revision": {
            "prior_guidance_path": None,
            "matched_count": 0,
            "raised_count": 0,
            "lowered_count": 0,
            "reaffirmed_count": 0,
            "unclear_count": 0,
            "mixed_count": 0,
            "top_revisions": [],
        },
    }
    output_paths.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    qa_shift_summary = {
        "placeholder_bundle": True,
        "notes": [PLACEHOLDER_NOTE],
        "counts": {},
        "prepared_remarks_vs_q_and_a": {"label": "mixed"},
        "analyst_skepticism": {"level": "low"},
        "management_answers_vs_prepared_uncertainty": {"label": "mixed"},
        "early_vs_late_q_and_a": {"label": "mixed"},
        "strongest_evidence": {},
    }
    output_paths.qa_shift_path.write_text(json.dumps(qa_shift_summary, indent=2), encoding="utf-8")

    media_quality_path = output_paths.output_dir / "media_quality.json"
    media_quality_path.write_text(
        json.dumps(
            {
                "audio_quality_ok": False,
                "video_quality_ok": False,
                "quality_notes": [PLACEHOLDER_NOTE],
                "suppression_flags": {
                    "audio_support_suppressed": True,
                    "video_support_suppressed": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths.multimodal_summary_path.write_text(
        json.dumps(
            {
                "transcript_primary_assessment": "amber",
                "audio_support_direction": "unavailable",
                "video_support_direction": "unavailable",
                "fusion_mode": "heuristic_fallback",
                "calibrated_support_score": 0.0,
                "multimodal_alignment": "low",
                "multimodal_confidence_adjustment": 0,
                "notes": [PLACEHOLDER_NOTE],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_meta_path = output_paths.output_dir / "run_meta.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "symbol": str(case["ticker"]),
                "event_dt": _event_dt_iso(event_date),
                "source_url": "",
                "run_id": str(case["source_call_id"]),
                "generated_at": datetime.now(UTC).isoformat(),
                "version": "placeholder-casepack",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# Placeholder Review Bundle",
        "",
        PLACEHOLDER_NOTE,
        "",
        f"- case id: {case['call_id']}",
        f"- source package: {source_package}",
        f"- source call id: {case['source_call_id']}",
        f"- source path: {source_path_value}",
        "- media support: unavailable",
        "",
        "## Transcript Excerpt",
        "",
        text.strip(),
    ]
    output_paths.report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    timing_note_path = output_paths.output_dir / "document_timing_note.txt"
    timing_note_path.write_text(
        "Placeholder bundle timing is relative only and is not tied to real media timestamps.",
        encoding="utf-8",
    )


def ensure_case_outputs(
    cases: pd.DataFrame,
    *,
    root: Path | None = None,
    force: bool = False,
) -> list[Path]:
    resolved_root = _root(root)
    generated: list[Path] = []
    for _, case in cases.iterrows():
        output_paths = case_output_paths(str(case["source_call_id"]), root=resolved_root)
        if (
            not force
            and output_paths.metrics_path.exists()
            and output_paths.qa_shift_path.exists()
            and output_paths.report_path.exists()
            and output_paths.transcript_path.exists()
        ):
            continue

        source_path_value = str(case.get("source_path", "") or case.get("baseline_transcript_path", ""))
        source_path = resolved_root / source_path_value
        text = source_path.read_text(encoding="utf-8")
        _write_placeholder_bundle(case, output_paths, text)
        generated.append(output_paths.output_dir)
    return generated
