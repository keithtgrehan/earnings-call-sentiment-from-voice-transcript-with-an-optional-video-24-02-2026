"""Conservative multimodal coverage and instrumentation reporting."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from earnings_call_sentiment.source_manifests import load_segments_manifest, load_sources_manifest

DEFAULT_ALIGNMENT_ROOT = Path("data/processed/multimodal/alignment")
DEFAULT_VISUAL_ROOT = Path("data/processed/multimodal/visual")
DEFAULT_NLP_ROOT = Path("data/processed/multimodal/nlp")
DEFAULT_EVAL_ROOT = Path("data/processed/multimodal/eval")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_eval_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        base = Path(out_dir)
    else:
        base = repo_root() / DEFAULT_EVAL_ROOT
    return base.expanduser().resolve()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _as_int(value: Any) -> int:
    try:
        if value is None or _clean_text(value) == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame[column].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _nonblank_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series([False] * len(frame), index=frame.index)
    return frame[column].astype(str).str.strip().ne("")


def _value_counts(frame: pd.DataFrame, column: str, *, mask: pd.Series | None = None) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    work = frame[mask].copy() if mask is not None else frame.copy()
    if work.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in work[column].astype(str).value_counts().to_dict().items()
    }


def _list_layers(*, has_alignment: bool, has_visual: bool, has_nlp: bool) -> str:
    layers = []
    if has_alignment:
        layers.append("alignment")
    if has_visual:
        layers.append("visual")
    if has_nlp:
        layers.append("nlp")
    return "+".join(layers) if layers else "transcript_only"


def _alignment_metrics(alignment_root: Path, source_id: str) -> dict[str, Any]:
    source_dir = alignment_root / source_id
    summary = _read_json(source_dir / "alignment_summary.json")
    aligned_segments = _read_csv(source_dir / "aligned_segments.csv")
    diarization_segments = _read_csv(source_dir / "diarization_segments.csv")

    has_artifacts = bool(summary) or not aligned_segments.empty
    return {
        "has_alignment": has_artifacts,
        "audio_aligned_segments": _as_int(summary.get("segment_count")) or int(len(aligned_segments)),
        "alignment_word_count": _as_int(summary.get("word_count")),
        "diarization_segment_count": _as_int(summary.get("diarization_segment_count")) or int(len(diarization_segments)),
        "diarization_applied": bool(summary.get("diarization_applied")),
    }


def _visual_metrics(visual_root: Path, source_id: str) -> tuple[dict[str, Any], Counter[str]]:
    source_dir = visual_root / source_id
    frame = _read_csv(source_dir / "segment_visual_features.csv")
    attempted_mask = _bool_mask(frame, "attempted")
    succeeded_mask = _bool_mask(frame, "extraction_succeeded")
    missing_tool_mask = _bool_mask(frame, "missing_tool_flag")
    usable_mask = frame["segment_visual_usability"].astype(str).eq("usable") if "segment_visual_usability" in frame.columns else pd.Series([False] * len(frame), index=frame.index)
    unusable_mask = frame["segment_visual_usability"].astype(str).eq("unusable") if "segment_visual_usability" in frame.columns else pd.Series([False] * len(frame), index=frame.index)
    limited_mask = frame["segment_visual_usability"].astype(str).eq("limited") if "segment_visual_usability" in frame.columns else pd.Series([False] * len(frame), index=frame.index)
    extraction_error_mask = _nonblank_mask(frame, "extraction_errors")

    unusable_reasons = Counter(
        frame.loc[unusable_mask, "usability_reason"].astype(str).tolist()
    ) if not frame.empty and "usability_reason" in frame.columns else Counter()

    return {
        "has_visual": not frame.empty,
        "visual_rows_total": int(len(frame)),
        "visually_usable_segments": int(usable_mask.sum()),
        "visually_limited_segments": int(limited_mask.sum()),
        "visually_unusable_segments": int(unusable_mask.sum()),
        "visual_attempted_segments": int(attempted_mask.sum()),
        "visual_failed_segments": int((attempted_mask & ~succeeded_mask).sum()),
        "visual_missing_tool_segments": int(missing_tool_mask.sum()),
        "visual_extraction_error_rows": int(extraction_error_mask.sum()),
    }, unusable_reasons


def _nlp_metrics(nlp_root: Path, source_id: str) -> dict[str, Any]:
    source_dir = nlp_root / source_id
    frame = _read_csv(source_dir / "nlp_segment_scores.csv")
    return {
        "has_nlp": not frame.empty,
        "nlp_rows_total": int(len(frame)),
        "segments_with_nlp_support": int(frame["segment_id"].astype(str).nunique()) if "segment_id" in frame.columns and not frame.empty else 0,
        "nlp_model_count": int(frame["model_name"].astype(str).nunique()) if "model_name" in frame.columns and not frame.empty else 0,
        "nlp_model_roles": sorted(frame["model_role"].astype(str).unique().tolist()) if "model_role" in frame.columns and not frame.empty else [],
    }


def build_multimodal_eval_summary(
    *,
    sources_path: str | Path | None = None,
    segments_path: str | Path | None = None,
    alignment_root: str | Path | None = None,
    visual_root: str | Path | None = None,
    nlp_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = load_sources_manifest(sources_path)
    segments = load_segments_manifest(segments_path)
    alignment_root_path = Path(alignment_root) if alignment_root is not None else (repo_root() / DEFAULT_ALIGNMENT_ROOT)
    visual_root_path = Path(visual_root) if visual_root is not None else (repo_root() / DEFAULT_VISUAL_ROOT)
    nlp_root_path = Path(nlp_root) if nlp_root is not None else (repo_root() / DEFAULT_NLP_ROOT)

    company_layout_rows: list[dict[str, Any]] = []
    unusable_reason_counts: Counter[str] = Counter()

    for _, source_row in sources.iterrows():
        source_id = _clean_text(source_row.get("source_id"))
        source_segments = segments[segments["source_id"].astype(str) == source_id].copy()
        face_expected_count = int(
            source_segments["face_expected"].astype(str).str.strip().str.lower().eq("true").sum()
        ) if not source_segments.empty and "face_expected" in source_segments.columns else 0

        alignment_metrics = _alignment_metrics(alignment_root_path, source_id)
        visual_metrics, reason_counts = _visual_metrics(visual_root_path, source_id)
        nlp_metrics = _nlp_metrics(nlp_root_path, source_id)
        unusable_reason_counts.update(reason_counts)

        company_layout_rows.append(
            {
                "source_id": source_id,
                "company": _clean_text(source_row.get("company")),
                "ticker": _clean_text(source_row.get("ticker")),
                "source_family": _clean_text(source_row.get("source_family")),
                "layout_type": _clean_text(source_row.get("layout_type")),
                "status": _clean_text(source_row.get("status")),
                "manifest_segment_count": int(len(source_segments)),
                "face_expected_segment_count": face_expected_count,
                "has_alignment": alignment_metrics["has_alignment"],
                "audio_aligned_segments": alignment_metrics["audio_aligned_segments"],
                "diarization_segment_count": alignment_metrics["diarization_segment_count"],
                "has_visual": visual_metrics["has_visual"],
                "visual_rows_total": visual_metrics["visual_rows_total"],
                "visually_usable_segments": visual_metrics["visually_usable_segments"],
                "visually_limited_segments": visual_metrics["visually_limited_segments"],
                "visually_unusable_segments": visual_metrics["visually_unusable_segments"],
                "visual_failed_segments": visual_metrics["visual_failed_segments"],
                "visual_missing_tool_segments": visual_metrics["visual_missing_tool_segments"],
                "has_nlp": nlp_metrics["has_nlp"],
                "segments_with_nlp_support": nlp_metrics["segments_with_nlp_support"],
                "nlp_model_count": nlp_metrics["nlp_model_count"],
                "evidence_layers": _list_layers(
                    has_alignment=alignment_metrics["has_alignment"],
                    has_visual=visual_metrics["has_visual"],
                    has_nlp=nlp_metrics["has_nlp"],
                ),
            }
        )

    coverage_frame = pd.DataFrame(company_layout_rows)

    company_layout_coverage: list[dict[str, Any]] = []
    if not coverage_frame.empty:
        for (company, layout_type), group in coverage_frame.groupby(["company", "layout_type"], dropna=False):
            company_layout_coverage.append(
                {
                    "company": str(company),
                    "layout_type": str(layout_type),
                    "source_call_count": int(len(group)),
                    "sources_with_alignment": int(group["has_alignment"].astype(bool).sum()),
                    "sources_with_visual": int(group["has_visual"].astype(bool).sum()),
                    "sources_with_nlp": int(group["has_nlp"].astype(bool).sum()),
                    "manifest_segment_count": int(group["manifest_segment_count"].sum()),
                    "visually_usable_segments": int(group["visually_usable_segments"].sum()),
                    "audio_aligned_segments": int(group["audio_aligned_segments"].sum()),
                    "segments_with_nlp_support": int(group["segments_with_nlp_support"].sum()),
                }
            )

    source_layout_groups = {}
    if not sources.empty:
        source_layout_groups = {
            f"{row['source_family']}|{row['layout_type']}": int(count)
            for (row, count) in [
                (
                    {
                        "source_family": source_family,
                        "layout_type": layout_type,
                    },
                    count,
                )
                for (source_family, layout_type), count in sources.groupby(["source_family", "layout_type"]).size().to_dict().items()
            ]
        }

    summary = {
        "status": "ok",
        "paths": {
            "sources_manifest": str(Path(sources_path).resolve()) if sources_path is not None else str((repo_root() / "data/source_manifests/earnings_call_sources.csv").resolve()),
            "segments_manifest": str(Path(segments_path).resolve()) if segments_path is not None else str((repo_root() / "data/source_manifests/earnings_call_segments.csv").resolve()),
            "alignment_root": str(alignment_root_path.resolve()),
            "visual_root": str(visual_root_path.resolve()),
            "nlp_root": str(nlp_root_path.resolve()),
        },
        "headline_counts": {
            "source_calls": int(len(sources)),
            "independent_source_layout_groups": int(sources[["source_family", "layout_type"]].drop_duplicates().shape[0]) if not sources.empty else 0,
            "segments": int(len(segments)),
            "visually_usable_segments": int(coverage_frame["visually_usable_segments"].sum()) if not coverage_frame.empty else 0,
            "audio_aligned_segments": int(coverage_frame["audio_aligned_segments"].sum()) if not coverage_frame.empty else 0,
            "segments_with_nlp_support": int(coverage_frame["segments_with_nlp_support"].sum()) if not coverage_frame.empty else 0,
        },
        "evidence_existence": {
            "sources_with_alignment": int(coverage_frame["has_alignment"].astype(bool).sum()) if not coverage_frame.empty else 0,
            "sources_without_alignment": int((~coverage_frame["has_alignment"].astype(bool)).sum()) if not coverage_frame.empty else 0,
            "sources_with_visual": int(coverage_frame["has_visual"].astype(bool).sum()) if not coverage_frame.empty else 0,
            "sources_without_visual": int((~coverage_frame["has_visual"].astype(bool)).sum()) if not coverage_frame.empty else 0,
            "sources_with_nlp": int(coverage_frame["has_nlp"].astype(bool).sum()) if not coverage_frame.empty else 0,
            "sources_without_nlp": int((~coverage_frame["has_nlp"].astype(bool)).sum()) if not coverage_frame.empty else 0,
            "sources_with_any_multimodal_sidecar": int(
                coverage_frame[["has_alignment", "has_visual", "has_nlp"]].any(axis=1).sum()
            )
            if not coverage_frame.empty
            else 0,
            "sources_with_no_multimodal_sidecar": int(
                (~coverage_frame[["has_alignment", "has_visual", "has_nlp"]].any(axis=1)).sum()
            )
            if not coverage_frame.empty
            else 0,
        },
        "extraction_failure_counts": {
            "visual_failed_segments": int(coverage_frame["visual_failed_segments"].sum()) if not coverage_frame.empty else 0,
            "visual_missing_tool_segments": int(coverage_frame["visual_missing_tool_segments"].sum()) if not coverage_frame.empty else 0,
            "visual_sources_without_artifacts": int((~coverage_frame["has_visual"].astype(bool)).sum()) if not coverage_frame.empty else 0,
            "alignment_sources_without_artifacts": int((~coverage_frame["has_alignment"].astype(bool)).sum()) if not coverage_frame.empty else 0,
            "nlp_sources_without_artifacts": int((~coverage_frame["has_nlp"].astype(bool)).sum()) if not coverage_frame.empty else 0,
        },
        "visual_unusable_reason_counts": {
            str(key): int(value)
            for key, value in dict(unusable_reason_counts).items()
        },
        "source_layout_group_counts": source_layout_groups,
        "source_coverage_by_company_layout": company_layout_coverage,
        "notes": [
            "Transcript artifacts remain primary; audio, visual, and NLP layers are supporting evidence only.",
            "This summary reports coverage, usability, and instrumentation breadth. It does not claim model quality or predictive improvement.",
            "Missing sidecar artifacts indicate absent support evidence in this checkout, not a fallback change to deterministic transcript outputs.",
        ],
    }
    return summary, company_layout_rows


def write_multimodal_eval_summary(
    *,
    out_dir: str | Path | None = None,
    sources_path: str | Path | None = None,
    segments_path: str | Path | None = None,
    alignment_root: str | Path | None = None,
    visual_root: str | Path | None = None,
    nlp_root: str | Path | None = None,
) -> dict[str, Path]:
    target_dir = default_eval_dir(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    summary, source_rows = build_multimodal_eval_summary(
        sources_path=sources_path,
        segments_path=segments_path,
        alignment_root=alignment_root,
        visual_root=visual_root,
        nlp_root=nlp_root,
    )

    summary_path = target_dir / "multimodal_eval_summary.json"
    coverage_csv_path = target_dir / "multimodal_source_coverage.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with coverage_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_id",
            "company",
            "ticker",
            "source_family",
            "layout_type",
            "status",
            "manifest_segment_count",
            "face_expected_segment_count",
            "has_alignment",
            "audio_aligned_segments",
            "diarization_segment_count",
            "has_visual",
            "visual_rows_total",
            "visually_usable_segments",
            "visually_limited_segments",
            "visually_unusable_segments",
            "visual_failed_segments",
            "visual_missing_tool_segments",
            "has_nlp",
            "segments_with_nlp_support",
            "nlp_model_count",
            "evidence_layers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    return {
        "summary_path": summary_path,
        "coverage_csv_path": coverage_csv_path,
    }
