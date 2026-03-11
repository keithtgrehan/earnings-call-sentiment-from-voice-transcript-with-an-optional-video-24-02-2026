from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

from earnings_call_sentiment import cli as cli_module
from earnings_call_sentiment.pipeline.run import (
    build_sentiment_pipeline,
    score_segments_with_sentiment,
)
from earnings_call_sentiment.review_workflow import build_segments_from_text


ALLOWED_LABELS = {"raised", "maintained", "lowered", "withdrawn", "unclear"}
LABEL_ORDER = ["withdrawn", "lowered", "maintained", "raised"]
SENTENCE_CUES = (
    "guidance",
    "outlook",
    "forecast",
    "we expect",
    "we now expect",
    "we are providing",
    "guides",
    "reaffirm",
    "prior guidance",
)
PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "withdrawn": (
        (
            "withdrawn_guidance",
            r"\b(withdraw(?:n|s|ing)?|suspend(?:ed|s|ing)?|no longer provid(?:e|ing)|"
            r"not provid(?:e|ing)|discontinu(?:e|ed|ing))\b.{0,40}\b(guidance|outlook)\b",
        ),
        (
            "withdrawn_guidance_reverse",
            r"\b(guidance|outlook)\b.{0,40}\b(withdraw(?:n|s|ing)?|suspend(?:ed|s|ing)?|"
            r"no longer provid(?:e|ing)|not provid(?:e|ing)|discontinu(?:e|ed|ing))\b",
        ),
    ),
    "lowered": (
        ("lowered_guidance_down", r"\bguidance down\b"),
        ("lowered_outlook_down", r"\boutlook down\b"),
        (
            "lowered_updated_guidance",
            r"\b(updated?|updating)\b.{0,30}\b(earnings|eps|guidance|outlook)\b.{0,40}"
            r"\b(lower|down|reduce|cut|revised)\b",
        ),
        (
            "lowered_took_down_guidance",
            r"\b(taken down|lowered|reduced|cut)\b.{0,50}\b(guidance|outlook)\b",
        ),
    ),
    "maintained": (
        ("maintained_flat_guidance", r"\bguidance is flat\b"),
        (
            "maintained_reiterate_guidance",
            r"\b(reiterate(?:d|s|ing)?)\b.{0,50}\b(guidance|outlook|revenue guidance|revenue outlook)\b",
        ),
        (
            "maintained_reaffirm_guidance",
            r"\b(reaffirm(?:ed|ing)?|maintain(?:ed|ing)?)\b.{0,50}\b(guidance|outlook|"
            r"revenue guidance|revenue outlook)\b",
        ),
        (
            "maintained_guidance_reverse",
            r"\b(guidance|outlook)\b.{0,50}\b(reaffirm(?:ed|ing)?|maintain(?:ed|ing)?|"
            r"reiterate(?:d|s|ing)?|unchanged|flat)\b",
        ),
    ),
    "raised": (
        ("raised_guidance_phrase", r"\b(?:raise(?:d|s)?|raising)\b.{0,50}\b(guidance|outlook|guides)\b"),
        ("raised_guidance_reverse", r"\b(guidance|outlook)\b.{0,50}\b(?:raise(?:d|s)?|raising)\b"),
        (
            "raised_metric_guidance_phrase",
            r"\b(?:raise(?:d|s)?|raising)\b.{0,60}\b(revenue|earnings|eps)\b.{0,30}\b"
            r"(guid(?:ance|es)|outlook)\b",
        ),
        (
            "raised_now_expect_up_from_prior",
            r"\b(?:we\s+)?now\s+expect\b.{0,120}\bup from\b.{0,24}\b(?:our|the)?\s*"
            r"(?:prior|previous)\s+(?:estimate|guidance|outlook)\b",
        ),
        (
            "raised_higher_than_prior",
            r"\b(?:higher|above)\b.{0,20}\bthan\b.{0,20}\b(?:our|the)?\s*"
            r"(?:prior|previous)\s+(?:guidance|outlook|estimate)\b",
        ),
        (
            "raised_increase_from_prior",
            r"\bincreas(?:e|ed|es|ing)\b.{0,20}\bfrom\b.{0,20}\b(?:our|the)?\s*"
            r"(?:prior|previous)\s+(?:guidance|outlook|estimate)\b",
        ),
        (
            "raised_increase_guidance_comparator",
            r"\b(?:increas(?:e|ed|es|ing))\b.{0,40}\b(guidance|outlook)\b.{0,220}\b"
            r"(?:from|compared with|compared to|as compared to)\b.{0,40}\b(?:our|the)?\s*"
            r"(?:prior|previous)\s+(?:guidance|outlook|estimate|public comments)\b",
        ),
        (
            "raised_guidance_increase_comparator",
            r"\b(guidance|outlook)\b.{0,80}\b(?:increas(?:e|ed|es|ing))\b.{0,60}\b"
            r"(?:from|compared with|compared to|as compared to)\b.{0,40}\b(?:our|the)?\s*"
            r"(?:prior|previous)\s+(?:guidance|outlook|estimate|public comments)\b",
        ),
        (
            "raised_guidance_above_prior",
            r"\b(?:guidance|outlook|estimate)\b.{0,40}\b(?:higher|above|up)\b.{0,24}\b"
            r"(?:than|from)\b.{0,20}\b(?:our|the)?\s*(?:prior|previous)\b",
        ),
    ),
}

RULE_EXPLANATIONS = {
    "withdrawn_guidance": "Explicit withdrawal or suspension wording was matched against guidance or outlook language.",
    "withdrawn_guidance_reverse": "Guidance or outlook language was paired with an explicit withdrawal or suspension phrase.",
    "lowered_guidance_down": "The sentence explicitly said guidance was down.",
    "lowered_outlook_down": "The sentence explicitly said outlook was down.",
    "lowered_updated_guidance": "An updated guidance or outlook sentence explicitly paired the update with lower, down, reduce, or cut wording.",
    "lowered_took_down_guidance": "The sentence explicitly said guidance or outlook was taken down, lowered, reduced, or cut.",
    "maintained_flat_guidance": "The sentence explicitly said guidance was flat.",
    "maintained_reiterate_guidance": "The sentence explicitly said guidance or outlook was reiterated.",
    "maintained_reaffirm_guidance": "The sentence explicitly said guidance or outlook was reaffirmed or maintained.",
    "maintained_guidance_reverse": "Guidance or outlook language was paired with explicit maintained, reiterated, reaffirmed, unchanged, or flat wording.",
    "raised_guidance_phrase": "The sentence explicitly said guidance or outlook was being raised.",
    "raised_guidance_reverse": "Guidance or outlook language was paired with explicit raised wording.",
    "raised_metric_guidance_phrase": "The sentence explicitly raised revenue, earnings, or EPS guidance or outlook.",
    "raised_now_expect_up_from_prior": "The sentence used a now-expect phrase with an explicit up-from-prior comparison.",
    "raised_higher_than_prior": "The sentence explicitly compared current guidance or outlook above a prior value.",
    "raised_increase_from_prior": "The sentence explicitly said guidance or outlook increased from prior guidance, outlook, or estimate.",
    "raised_increase_guidance_comparator": "The sentence explicitly paired increasing guidance or outlook with a prior comparator.",
    "raised_guidance_increase_comparator": "Guidance or outlook language was paired with an explicit increase versus a prior comparator.",
    "raised_guidance_above_prior": "Guidance or outlook language explicitly said the current value was above or up from the prior value.",
    "fallback_no_directional_verb": "Guidance text was extracted, but no explicit direction-change phrase matched the closed label set.",
    "fallback_no_guidance_rows": "No guidance rows were extracted by the deterministic pipeline, so the prediction stayed unclear.",
}


@dataclass(frozen=True)
class Prediction:
    call_id: str
    predicted_label: str
    predicted_evidence_text: str
    evidence_type: str
    rule_family: str
    explanation: str
    notes: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _sentence_is_candidate(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(cue in lowered for cue in SENTENCE_CUES)


def _classify_sentence_detail(sentence: str) -> tuple[str, str]:
    lowered = sentence.lower()
    if not _sentence_is_candidate(sentence):
        return "unclear", "not_guidance_candidate"
    for label in LABEL_ORDER:
        for rule_family, pattern in PATTERNS[label]:
            if re.search(pattern, lowered):
                return label, rule_family
    return "unclear", "fallback_no_directional_verb"


def _classify_sentence(sentence: str) -> str:
    return _classify_sentence_detail(sentence)[0]


def _predict_from_transcript(
    transcript_text: str,
    *,
    sentiment_pipeline: Any,
) -> Prediction:
    segments = build_segments_from_text(transcript_text)
    if not segments:
        return Prediction(
            call_id="",
            predicted_label="unclear",
            predicted_evidence_text="",
            evidence_type="no_guidance_extracted",
            rule_family="fallback_no_guidance_rows",
            explanation=RULE_EXPLANATIONS["fallback_no_guidance_rows"],
            notes="No usable text segments extracted from transcript.",
        )

    sentiment_segments = score_segments_with_sentiment(
        segments,
        sentiment_pipeline=sentiment_pipeline,
    )
    chunks_scored_df = cli_module._build_chunks_scored_df(sentiment_segments)
    guidance_df = cli_module._extract_guidance_df(chunks_scored_df)

    candidates: list[tuple[str, float, str, str]] = []
    for _, row in guidance_df.iterrows():
        strength = float(row.get("guidance_strength", 0.0) or 0.0)
        for sentence in _normalize_sentences(str(row.get("text", ""))):
            label, rule_family = _classify_sentence_detail(sentence)
            if label != "unclear":
                candidates.append((label, strength, sentence, rule_family))

    if candidates:
        labels_present = {item[0] for item in candidates}
        chosen: tuple[str, float, str, str] | None = None
        for label in LABEL_ORDER:
            if label not in labels_present:
                continue
            same_label = [item for item in candidates if item[0] == label]
            same_label.sort(key=lambda item: -item[1])
            chosen = same_label[0]
            break
        assert chosen is not None
        label, _, evidence, rule_family = chosen
        return Prediction(
            call_id="",
            predicted_label=label,
            predicted_evidence_text=evidence,
            evidence_type="explicit_directional",
            rule_family=rule_family,
            explanation=RULE_EXPLANATIONS[rule_family],
            notes="Explicit directional phrase found within engine-extracted guidance text.",
        )

    fallback_sentence = ""
    if not guidance_df.empty:
        for sentence in _normalize_sentences(str(guidance_df.iloc[0].get("text", ""))):
            if _sentence_is_candidate(sentence):
                fallback_sentence = sentence
                break
        if not fallback_sentence:
            fallback_sentence = _normalize_sentences(str(guidance_df.iloc[0].get("text", "")))[0]

    note = (
        "Guidance text was extracted, but no explicit change verb matched the closed label set."
        if not guidance_df.empty
        else "No guidance rows were extracted by the current deterministic pipeline."
    )
    return Prediction(
        call_id="",
        predicted_label="unclear",
        predicted_evidence_text=fallback_sentence,
        evidence_type=(
            "forward_looking_non_directional" if not guidance_df.empty else "no_guidance_extracted"
        ),
        rule_family=(
            "fallback_no_directional_verb" if not guidance_df.empty else "fallback_no_guidance_rows"
        ),
        explanation=RULE_EXPLANATIONS[
            "fallback_no_directional_verb" if not guidance_df.empty else "fallback_no_guidance_rows"
        ],
        notes=note,
    )


def _write_predictions(path: Path, rows: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "call_id",
                "predicted_label",
                "predicted_evidence_text",
                "evidence_type",
                "rule_family",
                "explanation",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "call_id": row.call_id,
                    "predicted_label": row.predicted_label,
                    "predicted_evidence_text": row.predicted_evidence_text,
                    "evidence_type": row.evidence_type,
                    "rule_family": row.rule_family,
                    "explanation": row.explanation,
                    "notes": row.notes,
                }
            )


def _write_mismatches(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "call_id",
        "ticker",
        "gold_label",
        "predicted_label",
        "match_bool",
        "gold_evidence_text",
        "predicted_evidence_text",
        "evidence_type",
        "rule_family",
        "explanation",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _distribution_markdown(title: str, counts: Counter[str]) -> list[str]:
    return [
        f"### {title}",
        f"- raised: {counts.get('raised', 0)}",
        f"- maintained: {counts.get('maintained', 0)}",
        f"- lowered: {counts.get('lowered', 0)}",
        f"- withdrawn: {counts.get('withdrawn', 0)}",
        f"- unclear: {counts.get('unclear', 0)}",
        "",
    ]


def _write_summary(
    path: Path,
    *,
    benchmark_name: str,
    benchmark_root: Path,
    gold_rows: list[dict[str, str]],
    predictions: list[Prediction],
    comparison_rows: list[dict[str, str]],
) -> None:
    gold_counts = Counter(row["guidance_change_label"] for row in gold_rows)
    pred_counts = Counter(row.predicted_label for row in predictions)
    total = len(gold_rows)
    matches = sum(1 for row in comparison_rows if row["match_bool"] == "True")
    mismatches = total - matches
    false_directional = sum(
        1
        for row in comparison_rows
        if row["gold_label"] == "unclear" and row["predicted_label"] != "unclear"
    )
    directional_calls = [row for row in comparison_rows if row["gold_label"] != "unclear"]
    directional_matched = sum(1 for row in directional_calls if row["match_bool"] == "True")
    unclear_calls = [row for row in comparison_rows if row["gold_label"] == "unclear"]
    unclear_conservative = sum(
        1 for row in unclear_calls if row["predicted_label"] == "unclear"
    )

    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in comparison_rows:
        confusion_counts[(row["gold_label"], row["predicted_label"])] += 1

    lines: list[str] = [
        f"# {benchmark_name}",
        "",
        "## Scope",
        f"- Evaluation set: `{benchmark_root}`",
        f"- Gold source of truth: `{benchmark_root / 'labels.csv'}`",
        "- Baseline method: current transcript-to-guidance extraction path plus a fixed closed-set sentence mapper over extracted guidance text.",
        "",
        "## Headline Result",
        f"- Row count: {total}",
        f"- Exact-match accuracy: {matches}/{total} ({(matches / total) * 100:.1f}%)",
        f"- Mismatch count: {mismatches}",
        f"- False directional claims on gold `unclear` rows: {false_directional}",
        "",
    ]
    lines.extend(_distribution_markdown("Gold Label Distribution", gold_counts))
    lines.extend(_distribution_markdown("Predicted Label Distribution", pred_counts))

    lines.extend(
        [
            "## Per-Call Prediction Review",
            "| call_id | ticker | gold_label | predicted_label | evidence_type | rule_family | evidence |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['call_id']} | {row['ticker']} | {row['gold_label']} | {row['predicted_label']} | "
            f"{row['evidence_type']} | {row['rule_family']} | {row['predicted_evidence_text'][:100]} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Directional Call Check",
            f"- Directional calls matched: {directional_matched}/{len(directional_calls)}",
            f"- Unclear calls kept conservative: {unclear_conservative}/{len(unclear_calls)}",
            "",
            "| call_id | gold_label | predicted_label | evidence_type | rule_family | match |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in directional_calls:
        lines.append(
            f"| {row['call_id']} | {row['gold_label']} | {row['predicted_label']} | "
            f"{row['evidence_type']} | {row['rule_family']} | {row['match_bool']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Confusion Summary",
            "| gold_label | predicted_label | count |",
            "|---|---|---|",
        ]
    )
    for (gold_label, predicted_label), count in sorted(confusion_counts.items()):
        lines.append(f"| {gold_label} | {predicted_label} | {count} |")
    lines.append("")

    lines.extend(
        [
            "## Mismatch Review",
        ]
    )
    mismatch_rows = [row for row in comparison_rows if row["match_bool"] != "True"]
    if mismatch_rows:
        lines.extend(
            [
                "| call_id | ticker | gold_label | predicted_label | evidence_type | rule_family | gold_evidence | predicted_evidence |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in mismatch_rows:
            lines.append(
                f"| {row['call_id']} | {row['ticker']} | {row['gold_label']} | "
                f"{row['predicted_label']} | {row['evidence_type']} | {row['rule_family']} | "
                f"{row['gold_evidence_text'][:80]} | {row['predicted_evidence_text'][:80]} |"
            )
    else:
        lines.append(f"- No mismatches on `{benchmark_root}`.")
    lines.append("")

    lines.extend(
        [
            "## Prediction Review Notes",
            "- `explicit_directional`: an explicit rule-matched guidance phrase determined the label.",
            "- `forward_looking_non_directional`: guidance text was extracted, but no explicit closed-set direction phrase was present, so the label stayed `unclear`.",
            "- `no_guidance_extracted`: the deterministic pipeline did not extract a guidance row, so the label stayed `unclear`.",
            "",
        ]
    )

    if matches == total and false_directional == 0 and directional_matched == len(directional_calls):
        judgment = "good enough as a baseline"
        reasons = [
            f"It caught all {len(directional_calls)} directional benchmark calls correctly.",
            f"It stayed conservative on all {len(unclear_calls)} gold `unclear` cases.",
            "It made no false directional claims on ambiguous transcripts.",
        ]
    else:
        judgment = "needs one controlled refinement pass"
        reasons = [
            "The current baseline missed at least one directional benchmark call or overcalled an unclear case.",
            "The next pass should stay narrow and address only the observed mismatch pattern.",
        ]

    lines.extend(
        [
            "## Judgment",
            f"- Conclusion: **{judgment}**",
        ]
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Limits",
            "- This benchmark is small and dominated by `unclear` labels.",
            "- This is a benchmark-evaluation result, not a predictive or trading-performance result.",
            "- No statistical-significance claim should be made from this pass.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Evaluate a guidance-change benchmark package.")
    parser.add_argument(
        "--benchmark-root",
        default="data/gold_guidance_calls",
        help="Benchmark package root relative to the repo root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gold_eval",
        help="Output directory for predictions, mismatches, and summary relative to the repo root.",
    )
    parser.add_argument(
        "--benchmark-name",
        default="Frozen Gold Benchmark Evaluation",
        help="Markdown report title.",
    )
    args = parser.parse_args()

    benchmark_root = repo_root / args.benchmark_root
    labels_path = benchmark_root / "labels.csv"
    outputs_dir = repo_root / args.output_dir
    predictions_path = outputs_dir / "predictions.csv"
    mismatches_path = outputs_dir / "mismatches.csv"
    summary_path = outputs_dir / "evaluation_summary.md"

    gold_rows = sorted(_read_csv(labels_path), key=lambda row: row["call_id"])
    for row in gold_rows:
        label = row["guidance_change_label"]
        if label not in ALLOWED_LABELS:
            raise SystemExit(f"Invalid gold label {label!r} for {row['call_id']}")

    sentiment_pipeline = build_sentiment_pipeline()

    predictions: list[Prediction] = []
    comparison_rows: list[dict[str, str]] = []
    mismatch_rows: list[dict[str, str]] = []
    for gold in gold_rows:
        transcript_path = repo_root / gold["source_path"]
        transcript_text = transcript_path.read_text(encoding="utf-8")
        prediction = _predict_from_transcript(
            transcript_text,
            sentiment_pipeline=sentiment_pipeline,
        )
        prediction = Prediction(
            call_id=gold["call_id"],
            predicted_label=prediction.predicted_label,
            predicted_evidence_text=prediction.predicted_evidence_text,
            evidence_type=prediction.evidence_type,
            rule_family=prediction.rule_family,
            explanation=prediction.explanation,
            notes=prediction.notes,
        )
        predictions.append(prediction)
        match = prediction.predicted_label == gold["guidance_change_label"]
        comparison = {
            "call_id": gold["call_id"],
            "ticker": gold["ticker"],
            "gold_label": gold["guidance_change_label"],
            "predicted_label": prediction.predicted_label,
            "match_bool": "True" if match else "False",
            "gold_evidence_text": gold["evidence_text"],
            "predicted_evidence_text": prediction.predicted_evidence_text,
            "evidence_type": prediction.evidence_type,
            "rule_family": prediction.rule_family,
            "explanation": prediction.explanation,
            "notes": prediction.notes,
        }
        comparison_rows.append(comparison)
        if not match:
            mismatch_rows.append(comparison)

    _write_predictions(predictions_path, predictions)
    _write_mismatches(mismatches_path, mismatch_rows)
    _write_summary(
        summary_path,
        benchmark_name=args.benchmark_name,
        benchmark_root=Path(args.benchmark_root),
        gold_rows=gold_rows,
        predictions=predictions,
        comparison_rows=comparison_rows,
    )

    print(f"wrote {predictions_path}")
    print(f"wrote {mismatches_path}")
    print(f"wrote {summary_path}")
    print(f"rows={len(predictions)}")
    print(f"matches={sum(1 for row in comparison_rows if row['match_bool'] == 'True')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
