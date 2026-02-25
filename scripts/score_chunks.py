from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from earnings_call_sentiment.sentiment import score_chunks


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-jsonl", default="outputs/chunks.jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    ap.add_argument("--device", type=int, default=-1, help="-1 cpu, 0 gpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    chunks_path = Path(args.chunks_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_iter = score_chunks(
        read_jsonl(chunks_path),
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    scored_path = out_dir / "chunks_scored.jsonl"
    n = write_jsonl(scored_path, scored_iter)

    scores_only_path = out_dir / "chunk_scores.jsonl"
    csv_path = out_dir / "chunk_scores.csv"

    rows = list(read_jsonl(scored_path))

    write_jsonl(
        scores_only_path,
        (
            {
                "chunk_id": r["chunk_id"],
                "start": r.get("start"),
                "end": r.get("end"),
                "sentiment_label": r.get("sentiment_label"),
                "sentiment_score": r.get("sentiment_score"),
            }
            for r in rows
        ),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "start", "end", "sentiment_label", "sentiment_score"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "chunk_id": r["chunk_id"],
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "sentiment_label": r.get("sentiment_label"),
                    "sentiment_score": r.get("sentiment_score"),
                }
            )

    print(f"Wrote {n} scored chunks -> {scored_path}")
    print(f"Wrote scores -> {scores_only_path}")
    print(f"Wrote CSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
