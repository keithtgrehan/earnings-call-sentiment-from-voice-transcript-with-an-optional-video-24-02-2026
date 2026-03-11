"""Rule tables for deterministic tone and behavior signals."""

from __future__ import annotations

UNCERTAINTY_RULES = (
    {
        "pattern": r"\bmay\b|\bmight\b|\bcould\b",
        "matched_phrase": "modal uncertainty",
        "signal_type": "hedging_modal",
        "strength": 1,
        "notes": "Weak modal hedge.",
    },
    {
        "pattern": r"\b(?:expect|believe|anticipate)\b",
        "matched_phrase": "forward-looking expectation",
        "signal_type": "forward_expectation",
        "strength": 1,
        "notes": "Forward-looking expectation language can soften commitment.",
    },
    {
        "pattern": r"\bcontinue to monitor\b",
        "matched_phrase": "continue to monitor",
        "signal_type": "monitoring_language",
        "strength": 2,
        "notes": "Management explicitly signaled continued monitoring.",
    },
    {
        "pattern": r"\bsubject to\b",
        "matched_phrase": "subject to",
        "signal_type": "conditional_language",
        "strength": 2,
        "notes": "Management qualified the statement with a condition.",
    },
    {
        "pattern": r"\bvisibility (?:remains|is) limited\b|\blimited visibility\b",
        "matched_phrase": "limited visibility",
        "signal_type": "visibility_constraint",
        "strength": 3,
        "notes": "Management explicitly said visibility was limited.",
    },
    {
        "pattern": r"\bdifficult to predict\b|\bhard to predict\b",
        "matched_phrase": "difficult to predict",
        "signal_type": "prediction_difficulty",
        "strength": 3,
        "notes": "Management explicitly said the outcome was difficult to predict.",
    },
    {
        "pattern": r"\btiming uncertainty\b|\buncertain timing\b",
        "matched_phrase": "timing uncertainty",
        "signal_type": "timing_uncertainty",
        "strength": 3,
        "notes": "Management explicitly flagged timing uncertainty.",
    },
    {
        "pattern": r"\bmacro(?:economic)? uncertainty\b",
        "matched_phrase": "macro uncertainty",
        "signal_type": "macro_uncertainty",
        "strength": 3,
        "notes": "Management explicitly referenced macro uncertainty.",
    },
)

REASSURANCE_RULES = (
    {
        "pattern": r"\bremain(?:s)? confident\b|\bwe remain confident\b",
        "matched_phrase": "remain confident",
        "strength": 3,
        "topic_hint": "confidence",
        "notes": "Explicit management confidence language.",
    },
    {
        "pattern": r"\bdemand remains strong\b|\bdemand is strong\b",
        "matched_phrase": "demand remains strong",
        "strength": 2,
        "topic_hint": "demand",
        "notes": "Management explicitly framed demand as strong.",
    },
    {
        "pattern": r"\bfundamentals remain intact\b|\bour fundamentals remain intact\b",
        "matched_phrase": "fundamentals remain intact",
        "strength": 2,
        "topic_hint": "fundamentals",
        "notes": "Management explicitly said fundamentals remain intact.",
    },
    {
        "pattern": r"\bpipeline remains healthy\b|\bour pipeline remains healthy\b",
        "matched_phrase": "pipeline remains healthy",
        "strength": 2,
        "topic_hint": "pipeline",
        "notes": "Management explicitly framed the pipeline as healthy.",
    },
    {
        "pattern": r"\bwell positioned\b|\bwe are well positioned\b",
        "matched_phrase": "well positioned",
        "strength": 3,
        "topic_hint": "positioning",
        "notes": "Management explicitly said the company is well positioned.",
    },
    {
        "pattern": r"\bremain(?:s)? on track\b|\bwe remain on track\b",
        "matched_phrase": "remain on track",
        "strength": 3,
        "topic_hint": "execution",
        "notes": "Management explicitly said execution remains on track.",
    },
    {
        "pattern": r"\bencouraged by\b",
        "matched_phrase": "encouraged by",
        "strength": 1,
        "topic_hint": "momentum",
        "notes": "Management used mild reassurance language.",
    },
)

SKEPTICISM_RULES = (
    {
        "pattern": r"\bhelp us understand why\b",
        "matched_phrase": "help us understand why",
        "strength": 2,
        "topic_hint": "clarification",
        "notes": "Analyst explicitly challenged the reasoning.",
    },
    {
        "pattern": r"\bwhat changed\b",
        "matched_phrase": "what changed",
        "strength": 2,
        "topic_hint": "change_driver",
        "notes": "Analyst explicitly asked what changed.",
    },
    {
        "pattern": r"\bwhy should we believe\b",
        "matched_phrase": "why should we believe",
        "strength": 3,
        "topic_hint": "credibility",
        "notes": "Analyst explicitly challenged management credibility.",
    },
    {
        "pattern": r"\bisn'?t that inconsistent\b",
        "matched_phrase": "isn't that inconsistent",
        "strength": 3,
        "topic_hint": "consistency",
        "notes": "Analyst explicitly called out inconsistency.",
    },
    {
        "pattern": r"\bare you seeing weakness\b",
        "matched_phrase": "are you seeing weakness",
        "strength": 2,
        "topic_hint": "demand",
        "notes": "Analyst explicitly probed for weakness.",
    },
    {
        "pattern": r"\bwhat gives you confidence\b",
        "matched_phrase": "what gives you confidence",
        "strength": 1,
        "topic_hint": "confidence",
        "notes": "Analyst asked management to justify confidence.",
    },
    {
        "pattern": r"\bhow sustainable is that\b",
        "matched_phrase": "how sustainable is that",
        "strength": 2,
        "topic_hint": "sustainability",
        "notes": "Analyst questioned durability.",
    },
    {
        "pattern": r"\bshould we think about this as weaker\b",
        "matched_phrase": "should we think about this as weaker",
        "strength": 3,
        "topic_hint": "weakness",
        "notes": "Analyst explicitly asked if the situation should be read as weaker.",
    },
)

QUESTION_PATTERNS = (
    r"\?$",
    r"\bhelp us understand why\b",
    r"\bwhat changed\b",
    r"\bwhy should we believe\b",
    r"\bisn'?t that inconsistent\b",
    r"\bare you seeing weakness\b",
    r"\bwhat gives you confidence\b",
    r"\bhow sustainable is that\b",
    r"\bshould we think about this as weaker\b",
)

OPERATOR_PATTERNS = (
    r"^operator\b",
    r"\bour next question\b",
    r"\bnext question comes from\b",
    r"\bplease go ahead\b",
)

TOPIC_PATTERNS = {
    "guidance": ("guidance", "outlook", "forecast"),
    "revenue": ("revenue", "sales", "arr"),
    "margin": ("margin", "profitability", "ebitda"),
    "demand": ("demand", "bookings", "orders"),
    "pipeline": ("pipeline", "backlog"),
    "macro": ("macro", "macroeconomic"),
    "execution": ("execution", "on track"),
}
