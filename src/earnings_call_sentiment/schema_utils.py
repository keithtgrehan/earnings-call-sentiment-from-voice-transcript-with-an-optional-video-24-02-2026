"""Utilities for resilient JSON parsing and normalized summary payloads."""

from __future__ import annotations

import json
import re
from typing import Any

_SUMMARY_KEYS = (
    "executive_summary",
    "key_signals",
    "risks",
    "evidence",
    "limitations",
)

_WRAPPER_KEYS = ("summary", "result", "data", "output")


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences around JSON-ish content."""
    value = str(text or "").strip()
    if not value:
        return ""

    direct = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", value, flags=re.IGNORECASE)
    if direct:
        return direct.group(1).strip()

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", value, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return value


def extract_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from free-form text."""
    content = str(text or "")
    if not content:
        return None

    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(content):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return content[start : idx + 1]

    return None


def _default_summary_payload(reason: str) -> dict[str, Any]:
    return {
        "executive_summary": "Optional summary unavailable; rely on deterministic artifacts.",
        "key_signals": [],
        "risks": [],
        "evidence": [],
        "limitations": [reason],
    }


def _unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    current: dict[str, Any] = payload
    for _ in range(4):
        if not isinstance(current, dict):
            break

        if len(current) == 1:
            key = next(iter(current))
            if key in _WRAPPER_KEYS and isinstance(current[key], dict):
                current = current[key]
                continue

        moved = False
        for key in _WRAPPER_KEYS:
            candidate = current.get(key)
            if isinstance(candidate, dict) and not any(item in current for item in _SUMMARY_KEYS):
                current = candidate
                moved = True
                break
        if not moved:
            break
    return current


def safe_json_loads(text: str) -> Any:
    """Parse JSON robustly and return a deterministic fallback payload on failure."""
    raw = str(text or "")
    cleaned = strip_markdown_fences(raw)

    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
        extracted = extract_json_object(cleaned)
        if extracted:
            candidates.append(extracted)

    extracted_raw = extract_json_object(raw)
    if extracted_raw:
        candidates.append(extracted_raw)

    seen: set[str] = set()
    for candidate in candidates:
        token = candidate.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return _unwrap_payload(parsed)
        return parsed

    return _default_summary_payload("LLM/provider output could not be parsed as JSON.")


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [segment.strip("- •\t ") for segment in re.split(r"\n+|;", text)]
        return [item for item in parts if item]

    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            key_text = str(key).strip()
            val_text = str(item).strip()
            if key_text and val_text:
                out.append(f"{key_text}: {val_text}")
        return out

    return []


def normalize_summary_payload(payload: dict) -> dict:
    """Normalize provider payload aliases into one canonical summary schema."""
    if not isinstance(payload, dict):
        return _default_summary_payload("Invalid summary payload type.")

    source = _unwrap_payload(dict(payload))

    aliases: dict[str, tuple[str, ...]] = {
        "executive_summary": (
            "executive_summary",
            "summary",
            "overview",
            "narrative",
            "executiveSummary",
            "conclusion",
        ),
        "key_signals": (
            "key_signals",
            "keySignals",
            "signals",
            "highlights",
            "key_points",
            "insights",
        ),
        "risks": ("risks", "risk", "risk_flags", "concerns"),
        "evidence": (
            "evidence",
            "citations",
            "supporting_evidence",
            "evidence_snippets",
            "proof",
            "proof_points",
        ),
        "limitations": ("limitations", "caveats", "notes", "assumptions"),
    }

    normalized: dict[str, Any] = {
        "executive_summary": "",
        "key_signals": [],
        "risks": [],
        "evidence": [],
        "limitations": [],
    }

    for canonical, keys in aliases.items():
        selected: Any = None
        for key in keys:
            if key in source and source[key] is not None:
                selected = source[key]
                break
        if selected is None:
            continue

        if canonical == "executive_summary":
            text = str(selected).strip()
            if text:
                normalized[canonical] = text
        else:
            normalized[canonical] = _as_str_list(selected)

    if not normalized["executive_summary"]:
        normalized["executive_summary"] = (
            "Optional summary unavailable; rely on deterministic artifacts."
        )

    if not normalized["limitations"]:
        normalized["limitations"] = [
            "Optional narrative output may be incomplete; deterministic artifacts remain source of truth."
        ]

    return normalized
