from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RiskResult:
    score_0_100: int
    triggers: list[str]
    metrics: dict[str, Any]


_TRIGGER_PATTERNS: list[tuple[str, str]] = [
    ("guidance_cut", r"\b(lower(ed)?|cut|reduce(d)?)\b.*\bguidance\b|\bguidance\b.*\b(lower(ed)?|cut|reduce(d)?)\b"),
    ("macro_uncertainty", r"\bmacro\b|\buncertain(ty)?\b|\bvolatil(e|ity)\b|\bheadwind(s)?\b"),
    ("demand_softness", r"\bsoft(en(ing)?)?\b.*\bdemand\b|\bweaker\b.*\bdemand\b"),
    ("margin_pressure", r"\bmargin(s)?\b.*\bpressure\b|\bcompression\b"),
    ("restructuring", r"\brestructur(ing|e)\b|\blayoff(s)?\b|\bcost[- ]cut(ting)?\b"),
    ("supply_constraints", r"\bsupply\b.*\bconstraint(s)?\b|\bshortage(s)?\b"),
    ("litigation_regulatory", r"\blitigation\b|\bregulator(y)?\b|\binvestigation\b"),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def export_sentiment_csv(scored_jsonl: Path, out_csv: Path) -> pd.DataFrame:
    rows = _read_jsonl(scored_jsonl)
    df = pd.DataFrame(rows)

    # keep only useful columns if present
    keep = [c for c in ["start", "end", "text", "label", "score", "sentiment", "sentiment_score"] if c in df.columns]
    if keep:
        df = df[keep]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def score_risk(df: pd.DataFrame, transcript_text: str | None = None) -> RiskResult:
    # sentiment column normalization
    score_col = "score" if "score" in df.columns else ("sentiment_score" if "sentiment_score" in df.columns else None)
    label_col = "label" if "label" in df.columns else ("sentiment" if "sentiment" in df.columns else None)

    if score_col is None or label_col is None:
        # fallback: can’t compute sentiment-derived metrics reliably
        base = 50
        triggers = []
        metrics = {"note": "missing sentiment columns"}
        return RiskResult(score_0_100=base, triggers=triggers, metrics=metrics)

    # assume POSITIVE/NEGATIVE style labels (sst2) or map otherwise
    labels = df[label_col].astype(str).str.upper()
    neg_mask = labels.str.contains("NEG")
    neg_rate = float(neg_mask.mean())

    # confidence-weighted negativity (higher = worse)
    conf = df[score_col].astype(float).clip(0, 1)
    neg_conf = float((conf * neg_mask.astype(float)).mean())

    # volatility proxy: std of signed sentiment (neg = -score, pos = +score)
    signed = conf.copy()
    signed[neg_mask] = -signed[neg_mask]
    vol = float(signed.std()) if len(signed) > 1 else 0.0

    # triggers from transcript text (or concatenate chunk texts)
    if transcript_text is None:
        transcript_text = " ".join(df["text"].astype(str).tolist()) if "text" in df.columns else ""
    t = transcript_text.lower()

    triggers: list[str] = []
    for name, pat in _TRIGGER_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            triggers.append(name)

    # score: start at 0, add components, clamp to 0..100
    # (simple, explainable heuristic)
    score = 0.0
    score += 60.0 * neg_rate          # up to +60
    score += 30.0 * neg_conf          # up to +30
    score += 20.0 * min(vol / 0.35, 1.0)  # up to +20 (scaled)

    # trigger bumps (each is material risk language)
    score += 5.0 * len(triggers)      # +5 each

    score_i = int(max(0, min(100, round(score))))

    metrics = {
        "neg_rate": neg_rate,
        "neg_conf": neg_conf,
        "volatility": vol,
        "trigger_count": len(triggers),
    }
    return RiskResult(score_0_100=score_i, triggers=triggers, metrics=metrics)


def write_risk_json(result: RiskResult, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "risk_score_0_100": result.score_0_100,
        "bearish_triggers": result.triggers,
        "metrics": result.metrics,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_postprocess(outputs_dir: Path) -> None:
    scored = outputs_dir / "chunks_scored.jsonl"
    if not scored.exists():
        raise FileNotFoundError(f"missing {scored}")

    df = export_sentiment_csv(scored, outputs_dir / "sentiment_segments.csv")

    # optional transcript text if you later write it; otherwise uses df text
    transcript_txt = outputs_dir / "transcript.txt"
    transcript_text = transcript_txt.read_text(encoding="utf-8") if transcript_txt.exists() else None

    risk = score_risk(df, transcript_text=transcript_text)
    write_risk_json(risk, outputs_dir / "risk_metrics.json")
