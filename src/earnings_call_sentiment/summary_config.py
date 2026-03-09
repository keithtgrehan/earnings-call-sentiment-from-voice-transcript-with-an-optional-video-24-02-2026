"""Configuration and preflight checks for optional summary generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

SummaryProvider = Literal["none", "openai_compatible"]


@dataclass(frozen=True)
class SummaryConfig:
    enabled: bool
    provider: SummaryProvider
    model: str | None
    base_url: str | None
    api_key_env: str | None
    timeout_s: float


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: str | None) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normalize_provider(value: str | None) -> SummaryProvider:
    token = str(value or "none").strip().lower()
    if token == "openai_compatible":
        return "openai_compatible"
    return "none"


def load_summary_config(
    *,
    enabled: bool = False,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_s: float = 30.0,
) -> SummaryConfig:
    """Load summary configuration, reading env vars only when enabled."""
    is_enabled = bool(enabled)

    if not is_enabled:
        return SummaryConfig(
            enabled=False,
            provider=_normalize_provider(provider),
            model=_clean(model),
            base_url=_clean(base_url),
            api_key_env=_clean(api_key_env),
            timeout_s=max(1.0, float(timeout_s)),
        )

    resolved_provider = _normalize_provider(provider or os.getenv("ECS_SUMMARY_PROVIDER"))
    resolved_model = _clean(model) or _clean(os.getenv("ECS_SUMMARY_MODEL"))
    resolved_base_url = _clean(base_url) or _clean(os.getenv("ECS_SUMMARY_BASE_URL"))
    resolved_api_key_env = _clean(api_key_env) or _clean(
        os.getenv("ECS_SUMMARY_API_KEY_ENV")
    )

    resolved_timeout = float(timeout_s)
    if timeout_s == 30.0:
        timeout_env = _clean(os.getenv("ECS_SUMMARY_TIMEOUT_S"))
        if timeout_env is not None:
            try:
                resolved_timeout = float(timeout_env)
            except ValueError:
                resolved_timeout = float(timeout_s)

    return SummaryConfig(
        enabled=True,
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
        timeout_s=max(1.0, resolved_timeout),
    )


def run_summary_preflight(config: SummaryConfig) -> tuple[bool, str]:
    """Validate summary config before any provider request."""
    if not config.enabled:
        return True, "Summary disabled; optional summary stage is skipped."

    if config.provider == "none":
        return (
            False,
            "Summary is enabled but provider='none'. Set --summary-provider openai_compatible or disable --llm-summary.",
        )

    if not config.model:
        return False, "Summary preflight failed: missing summary model."

    if not config.base_url:
        return False, "Summary preflight failed: missing summary base URL."

    if not config.api_key_env:
        return False, "Summary preflight failed: missing API key env var name."

    api_key = _clean(os.getenv(config.api_key_env))
    if not api_key:
        return (
            False,
            f"Summary preflight failed: environment variable {config.api_key_env!r} is not set.",
        )

    return True, "Summary preflight passed."
