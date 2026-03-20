from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT / "data" / "processed" / "multimodal" / "visual"
CURATED_STATUS_PATH = VISUAL_ROOT / "curated_clip_run_status.json"
OUT_PATH = VISUAL_ROOT / "visual_status_report.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segment_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(VISUAL_ROOT.glob("*/segment_visual_features.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _metric_scope(row: dict[str, str]) -> str:
    derived: list[str] = []
    if row.get("mean_head_pose_change"):
        derived.append("head-pose proxy")
    if row.get("gaze_variability_proxy"):
        derived.append("gaze variability proxy")
    if row.get("blink_or_eye_closure_proxy"):
        derived.append("blink/eye-closure proxy")
    if row.get("au_intensity_mean") and row.get("au_intensity_supported", "").lower() == "true":
        derived.append("AU intensity summary")
    if derived:
        if row.get("segment_visual_usability") != "usable":
            return "face detection + limited low-confidence face-derived proxies"
        return "face detection + " + ", ".join(derived)
    return "face detection only"


def build_report() -> str:
    curated_status = _load_json(CURATED_STATUS_PATH)
    rows = _load_segment_rows()

    total_windows = len(rows)
    usable_windows = sum(row.get("segment_visual_usability") == "usable" for row in rows)
    unusable_windows = sum(row.get("segment_visual_usability") == "unusable" for row in rows)
    extraction_completed = sum(row.get("extraction_succeeded") == "true" for row in rows)
    reasons = Counter(row.get("usability_reason", "") for row in rows if row.get("usability_reason"))

    lines: list[str] = []
    lines.append("# Visual Status Report")
    lines.append("")
    lines.append("This report summarizes the current committed clip-based OpenFace visual sidecar outputs.")
    lines.append("Visual outputs are supporting evidence only and remain secondary to transcript-backed deterministic review truth.")
    lines.append("")
    lines.append("## Aggregate Snapshot")
    lines.append(f"- Sources with clip-based visual runs: `{curated_status.get('completed_clip_runs', 0)}`")
    lines.append(f"- Segment windows summarized: `{total_windows}`")
    lines.append(f"- Usable windows: `{usable_windows}`")
    lines.append(f"- Unusable windows: `{unusable_windows}`")
    lines.append(f"- Extraction-completed windows: `{extraction_completed}`")
    if reasons:
        lines.append(f"- Most common unusability reason: `{reasons.most_common(1)[0][0]}`")
    lines.append("")
    lines.append("## What Was Actually Extracted")
    lines.append("- Low-level observational summaries only: `frames_with_face`, `face_detection_rate`, `mean_head_pose_change`, `gaze_variability_proxy`, `blink_or_eye_closure_proxy`, and AU intensity summaries when supported.")
    lines.append("- No psychological labels, no deception labels, no truthfulness claims, and no market-edge claims.")
    lines.append("- These clips are approximate manual runtime-check windows. They are not transcript-aligned or timestamp-verified evidence windows.")
    lines.append("")
    lines.append("## Per-Source Windows")
    lines.append("| Source | Window | Usability | Reason | Face Frames | Face Detection Rate | Extracted Scope |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        source_id = row.get("source_id", "")
        window_id = row.get("segment_id", "")
        usability = row.get("segment_visual_usability", "")
        reason = row.get("usability_reason", "")
        face_frames = f"{row.get('frames_with_face', '')}/{row.get('frames_total', '')}"
        detection_rate = row.get("face_detection_rate", "")
        scope = _metric_scope(row)
        lines.append(
            f"| `{source_id}` | `{window_id}` | `{usability}` | `{reason}` | `{face_frames}` | `{detection_rate}` | {scope} |"
        )
    lines.append("")
    lines.append("## Practical Readout")
    lines.append("- `goog_q1_2025_example` is the only currently committed usable visual window.")
    lines.append("- `bac_q4_2025_example`, `dis_q1_fy26_example`, and `sbux_prepared_remarks_example` completed extraction but remained visually unusable because the face was too small or intermittent.")
    lines.append("- This makes the video layer credible as a supporting observational layer because it records both successful usable windows and honest unusable windows rather than pretending every run is informative.")
    lines.append("")
    lines.append("## What This Still Does Not Prove")
    lines.append("- It does not prove full-call visual validity.")
    lines.append("- It does not prove predictive lift.")
    lines.append("- It does not replace transcript-first deterministic review outputs.")
    lines.append("- It does not support psychological, deception, or hidden-state claims.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    OUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
