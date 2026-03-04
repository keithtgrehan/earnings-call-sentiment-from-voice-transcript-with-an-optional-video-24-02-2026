#!/usr/bin/env python3
"""Optional narrative evaluation layer over deterministic output artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd


LLM_RESPONSE_SCHEMA: dict[str, type] = {
    "takeaways": list,
    "guidance_changes": dict,
    "risk_points": list,
    "uncertainty_language": list,
    "confidence": str,
}

GUIDANCE_LABELS = ("raised", "lowered", "reaffirmed", "mixed", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic/LLM narrative evaluation from output artifacts."
    )
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--llm", choices=("ollama", "none"), default="none")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=300)
    return parser.parse_args()


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Missing required artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse metrics JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"metrics.json must contain an object: {path}")
    return payload


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _clip_text(value: Any, max_len: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _summarize_guidance_revisions(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["diff_num"] = pd.to_numeric(work.get("diff", pd.Series([], dtype="float64")), errors="coerce")
    work["abs_diff"] = work["diff_num"].abs()
    work = work.sort_values("abs_diff", ascending=False)

    rows: list[dict[str, Any]] = []
    for _, row in work.head(max(1, limit)).iterrows():
        rows.append(
            {
                "topic": str(row.get("topic", "")),
                "period": str(row.get("period", "")),
                "revision_label": str(row.get("revision_label", "unclear")),
                "diff": _coerce_float(row.get("diff_num")),
                "current_midpoint": _coerce_float(row.get("current_midpoint")),
                "prior_midpoint": _coerce_float(row.get("prior_midpoint")),
                "overlap_score": _coerce_float(row.get("overlap_score")),
                "current_start": _coerce_float(row.get("current_start")),
                "current_end": _coerce_float(row.get("current_end")),
                "current_text": _clip_text(row.get("current_text_snippet", "")),
                "prior_text": _clip_text(row.get("prior_text_snippet", "")),
            }
        )
    return rows


def _summarize_tone_changes(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    if "tone_change_z" in work.columns:
        work["score"] = pd.to_numeric(work["tone_change_z"], errors="coerce")
    else:
        work["score"] = pd.to_numeric(
            work.get("sentiment_score", pd.Series([], dtype="float64")),
            errors="coerce",
        )
    work["abs_score"] = work["score"].abs()
    if "is_change" in work.columns:
        mask = (
            work["is_change"]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "t", "yes"})
        )
        filtered = work[mask]
        if not filtered.empty:
            work = filtered
    work = work.sort_values("abs_score", ascending=False)

    rows: list[dict[str, Any]] = []
    for _, row in work.head(max(1, limit)).iterrows():
        rows.append(
            {
                "start": _coerce_float(row.get("start")),
                "end": _coerce_float(row.get("end")),
                "tone_change_z": _coerce_float(row.get("tone_change_z")),
                "sentiment_score": _coerce_float(row.get("sentiment_score")),
                "is_change": str(row.get("is_change", "")),
                "text": _clip_text(row.get("text", "")),
            }
        )
    return rows


def _extract_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "num_chunks_scored": metrics.get("num_chunks_scored"),
        "sentiment_mean": metrics.get("sentiment_mean"),
        "sentiment_std": metrics.get("sentiment_std"),
    }
    guidance = metrics.get("guidance", {})
    if isinstance(guidance, dict):
        summary["guidance_row_count"] = guidance.get("row_count")
        summary["guidance_strength_mean"] = guidance.get("mean_strength")
    tone_changes = metrics.get("tone_changes", {})
    if isinstance(tone_changes, dict):
        summary["tone_change_row_count"] = tone_changes.get("row_count")
        summary["tone_change_count"] = tone_changes.get("change_count")
    guidance_revision = metrics.get("guidance_revision", {})
    if isinstance(guidance_revision, dict):
        summary["guidance_revision"] = {
            key: guidance_revision.get(key)
            for key in (
                "matched_count",
                "raised_count",
                "lowered_count",
                "reaffirmed_count",
                "mixed_count",
                "unclear_count",
            )
        }
    return summary


def _deterministic_narrative(
    key_metrics: dict[str, Any],
    revisions: list[dict[str, Any]],
    tone_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = key_metrics.get("guidance_revision")
    if isinstance(counts, dict):
        guidance_counts = {
            label: int(counts.get(f"{label}_count", 0) or 0) for label in GUIDANCE_LABELS
        }
    else:
        guidance_counts = {label: 0 for label in GUIDANCE_LABELS}

    takeaways = [
        f"Chunks scored: {key_metrics.get('num_chunks_scored')}",
        (
            "Guidance revisions: "
            f"raised={guidance_counts['raised']}, lowered={guidance_counts['lowered']}, "
            f"reaffirmed={guidance_counts['reaffirmed']}, mixed={guidance_counts['mixed']}, "
            f"unclear={guidance_counts['unclear']}"
        ),
        f"Tone change events considered: {len(tone_changes)}",
    ]

    risk_points: list[str] = []
    for revision in revisions[:3]:
        label = str(revision.get("revision_label", "unclear"))
        topic = str(revision.get("topic", ""))
        diff = revision.get("diff")
        if label == "lowered":
            risk_points.append(f"Lowered {topic} guidance (diff={diff}).")
    if not risk_points and tone_changes:
        top = tone_changes[0]
        risk_points.append(
            "Largest tone shift at "
            f"{top.get('start')}s->{top.get('end')}s with z={top.get('tone_change_z')}."
        )
    if not risk_points:
        risk_points.append("No explicit risk signals detected in available deterministic artifacts.")

    uncertainty_language = []
    for revision in revisions[:3]:
        prior_text = str(revision.get("prior_text", ""))
        current_text = str(revision.get("current_text", ""))
        combined = f"{prior_text} {current_text}".lower()
        if any(token in combined for token in ("uncertain", "challenging", "depends", "maybe")):
            uncertainty_language.append(_clip_text(current_text or prior_text, max_len=120))
    if not uncertainty_language:
        uncertainty_language.append("No explicit uncertainty phrase matched in top guidance revisions.")

    return {
        "takeaways": takeaways,
        "guidance_changes": guidance_counts,
        "risk_points": risk_points,
        "uncertainty_language": uncertainty_language,
        "confidence": "deterministic",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or first >= last:
        raise RuntimeError("LLM output did not contain a JSON object.")
    candidate = stripped[first : last + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM output JSON parse failed.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM output must be a JSON object.")
    return payload


def _validate_llm_response(payload: dict[str, Any]) -> None:
    missing = sorted(set(LLM_RESPONSE_SCHEMA) - set(payload))
    if missing:
        raise RuntimeError(f"LLM output missing required keys: {missing}")
    extra = sorted(set(payload) - set(LLM_RESPONSE_SCHEMA))
    if extra:
        raise RuntimeError(f"LLM output contains unexpected keys: {extra}")

    for key, expected_type in LLM_RESPONSE_SCHEMA.items():
        if not isinstance(payload[key], expected_type):
            raise RuntimeError(f"LLM output key '{key}' must be {expected_type.__name__}.")

    guidance = payload["guidance_changes"]
    required_guidance_keys = set(GUIDANCE_LABELS)
    if set(guidance) != required_guidance_keys:
        raise RuntimeError(
            "LLM output guidance_changes must contain exactly: "
            f"{sorted(required_guidance_keys)}"
        )
    for label in GUIDANCE_LABELS:
        value = guidance[label]
        if not isinstance(value, int):
            raise RuntimeError(f"guidance_changes['{label}'] must be int")

    for key in ("takeaways", "risk_points", "uncertainty_language"):
        values = payload[key]
        if not values:
            raise RuntimeError(f"LLM output list '{key}' must not be empty")
        if not all(isinstance(item, str) and item.strip() for item in values):
            raise RuntimeError(f"LLM output list '{key}' must contain non-empty strings")

    confidence = payload["confidence"].strip().lower()
    if confidence not in {"low", "medium", "high"}:
        raise RuntimeError("LLM output confidence must be one of: low, medium, high")


def _build_ollama_prompt(
    *,
    key_metrics: dict[str, Any],
    revisions: list[dict[str, Any]],
    tone_changes: list[dict[str, Any]],
    report_text: str,
    max_tokens: int,
) -> str:
    schema = {
        "takeaways": ["string", "string"],
        "guidance_changes": {label: "int" for label in GUIDANCE_LABELS},
        "risk_points": ["string"],
        "uncertainty_language": ["string"],
        "confidence": "low|medium|high",
    }
    prompt_payload = {
        "key_metrics": key_metrics,
        "top_guidance_revisions": revisions,
        "top_tone_changes": tone_changes,
        "report_excerpt": _clip_text(report_text, max_len=1200),
    }
    return (
        "You are evaluating deterministic earnings-call artifacts.\n"
        "Tasks:\n"
        "1) summarize key call takeaways\n"
        "2) highlight guidance changes (raise/lower/reaffirm)\n"
        "3) flag risk points and uncertainty language\n"
        "Return STRICT JSON only, no markdown, no comments.\n"
        f"Target at most {max_tokens} tokens.\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Data: {json.dumps(prompt_payload, ensure_ascii=False)}"
    )


def _call_ollama_json(
    *,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_predict": int(max_tokens), "temperature": 0},
    }
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Failed to call local Ollama at http://127.0.0.1:11434. "
            "Ensure Ollama is running."
        ) from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama API response was not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Ollama API returned non-object payload.")

    model_text = str(decoded.get("response", "")).strip()
    if not model_text:
        raise RuntimeError("Ollama API returned empty response text.")

    llm_payload = _extract_json_object(model_text)
    _validate_llm_response(llm_payload)
    return llm_payload


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LLM Evaluation",
        "",
        f"- mode: {summary.get('llm_mode')}",
        f"- model: {summary.get('model')}",
        f"- generated_at: {summary.get('generated_at')}",
        "",
        "## Key Metrics",
        "```json",
        json.dumps(summary.get("key_metrics", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Guidance Revisions",
        "",
        "| topic | period | label | diff | overlap | current_text |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    revisions = summary.get("top_guidance_revisions", [])
    if isinstance(revisions, list) and revisions:
        for item in revisions:
            lines.append(
                "| {topic} | {period} | {label} | {diff} | {overlap} | {text} |".format(
                    topic=item.get("topic", ""),
                    period=item.get("period", ""),
                    label=item.get("revision_label", ""),
                    diff=item.get("diff", ""),
                    overlap=item.get("overlap_score", ""),
                    text=_clip_text(item.get("current_text", ""), max_len=100).replace("|", "/"),
                )
            )
    else:
        lines.append("| _none_ |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Tone Changes",
            "",
            "| start | end | z | text |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    tone = summary.get("top_tone_changes", [])
    if isinstance(tone, list) and tone:
        for item in tone:
            lines.append(
                "| {start} | {end} | {z} | {text} |".format(
                    start=item.get("start", ""),
                    end=item.get("end", ""),
                    z=item.get("tone_change_z", ""),
                    text=_clip_text(item.get("text", ""), max_len=100).replace("|", "/"),
                )
            )
    else:
        lines.append("| _none_ |  |  |  |")

    lines.extend(["", "## Narrative", ""])
    narrative = summary.get("narrative", {})
    if isinstance(narrative, dict):
        for key in ("takeaways", "risk_points", "uncertainty_language"):
            values = narrative.get(key, [])
            lines.append(f"### {key.replace('_', ' ').title()}")
            if isinstance(values, list) and values:
                for value in values:
                    lines.append(f"- {value}")
            else:
                lines.append("- _none_")
            lines.append("")
        lines.append(f"Confidence: {narrative.get('confidence')}")
    else:
        lines.append("_No narrative available._")
    return "\n".join(lines)


def run_eval(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"
    guidance_path = out_dir / "guidance_revision.csv"
    tone_path = out_dir / "tone_changes.csv"

    metrics = _load_metrics(metrics_path)
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    guidance_df = _load_optional_csv(guidance_path)
    tone_df = _load_optional_csv(tone_path)

    limit = max(1, int(args.limit))
    top_k = max(limit, int(args.top_k))
    revisions = _summarize_guidance_revisions(guidance_df, top_k)[:limit]
    tone_changes = _summarize_tone_changes(tone_df, top_k)[:limit]
    key_metrics = _extract_key_metrics(metrics)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "llm_mode": args.llm,
        "model": args.model,
        "inputs": {
            "metrics_json": str(metrics_path),
            "report_md": str(report_path) if report_path.exists() else None,
            "guidance_revision_csv": str(guidance_path) if guidance_path.exists() else None,
            "tone_changes_csv": str(tone_path) if tone_path.exists() else None,
        },
        "key_metrics": key_metrics,
        "top_guidance_revisions": revisions,
        "top_tone_changes": tone_changes,
    }

    if args.llm == "none":
        summary["narrative"] = _deterministic_narrative(key_metrics, revisions, tone_changes)
    else:
        prompt = _build_ollama_prompt(
            key_metrics=key_metrics,
            revisions=revisions,
            tone_changes=tone_changes,
            report_text=report_text,
            max_tokens=int(args.max_tokens),
        )
        summary["narrative"] = _call_ollama_json(
            model=args.model,
            prompt=prompt,
            timeout=int(args.timeout),
            max_tokens=int(args.max_tokens),
        )

    json_path = out_dir / "llm_eval.json"
    md_path = out_dir / "llm_eval.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_build_markdown(summary), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()
    json_path, md_path = run_eval(args)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
