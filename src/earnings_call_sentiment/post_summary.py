"""Optional post-pipeline summary generation."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

from earnings_call_sentiment.schema_utils import normalize_summary_payload, safe_json_loads
from earnings_call_sentiment.summary_config import SummaryConfig, run_summary_preflight

SUMMARY_SCHEMA_VERSION = "1.0.0"


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[verbose] {message}")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _read_csv_records(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.head(limit).to_dict(orient="records")


def build_summary_input(out_dir: Path) -> dict[str, Any]:
    """Build provider input from deterministic artifacts only."""
    out_path = Path(out_dir).expanduser().resolve()

    artifact_paths = {
        "metrics": out_path / "metrics.json",
        "risk_metrics": out_path / "risk_metrics.json",
        "guidance": out_path / "guidance.csv",
        "guidance_revision": out_path / "guidance_revision.csv",
        "tone_changes": out_path / "tone_changes.csv",
        "report": out_path / "report.md",
        "run_meta": out_path / "run_meta.json",
    }

    source_artifacts = [
        name
        for name, path in artifact_paths.items()
        if path.exists() and path.is_file() and path.stat().st_size > 0
    ]

    report_text = ""
    report_path = artifact_paths["report"]
    if report_path.exists() and report_path.is_file() and report_path.stat().st_size > 0:
        report_text = report_path.read_text(encoding="utf-8")[:8000]

    return {
        "metrics": _read_json(artifact_paths["metrics"]) or {},
        "risk_metrics": _read_json(artifact_paths["risk_metrics"]) or {},
        "guidance": _read_csv_records(artifact_paths["guidance"]),
        "guidance_revision": _read_csv_records(artifact_paths["guidance_revision"]),
        "tone_changes": _read_csv_records(artifact_paths["tone_changes"]),
        "report": report_text,
        "run_meta": _read_json(artifact_paths["run_meta"]) or {},
        "source_artifacts": source_artifacts,
    }


def _extract_provider_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(payload, ensure_ascii=False)

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        if parts:
            return "\n".join(parts)

    return json.dumps(payload, ensure_ascii=False)


def _request_openai_compatible_summary(
    summary_input: dict[str, Any], config: SummaryConfig
) -> str:
    if not config.api_key_env:
        raise RuntimeError("Missing API key env var name for summary provider.")

    api_key = str(os.getenv(config.api_key_env, "")).strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key value in environment variable {config.api_key_env!r}."
        )

    system_prompt = (
        "Return only valid JSON with keys: executive_summary, key_signals, risks, "
        "evidence, limitations. Keep output grounded in the provided artifacts."
    )

    request_payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(summary_input, ensure_ascii=False)},
        ],
    }

    endpoint = str(config.base_url or "").rstrip("/") + "/chat/completions"
    req = request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=float(config.timeout_s)) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"Summary provider request failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Summary provider returned non-JSON response.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Summary provider returned invalid response payload.")

    return _extract_provider_content(payload)


def generate_optional_summary(
    out_dir: Path,
    config: SummaryConfig,
    verbose: bool = False,
) -> Path | None:
    """Generate optional narrative summary JSON from deterministic artifacts."""
    if not config.enabled:
        return None

    ok, message = run_summary_preflight(config)
    if not ok:
        raise RuntimeError(message)
    _log(verbose, message)

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    summary_input = build_summary_input(out_path)

    if config.provider == "openai_compatible":
        raw_output = _request_openai_compatible_summary(summary_input, config)
    else:
        raise RuntimeError(f"Unsupported summary provider: {config.provider}")

    parsed = safe_json_loads(raw_output)
    normalized = (
        normalize_summary_payload(parsed)
        if isinstance(parsed, dict)
        else normalize_summary_payload({})
    )

    payload: dict[str, Any] = {
        **normalized,
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": config.provider,
        "model": config.model,
        "source_artifacts": summary_input.get("source_artifacts", []),
        "schema_version": SUMMARY_SCHEMA_VERSION,
    }

    summary_path = out_path / "llm_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
