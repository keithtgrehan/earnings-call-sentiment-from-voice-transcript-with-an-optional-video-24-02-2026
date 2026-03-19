#!/usr/bin/env python3
"""Run optional NLP assist scoring over transcript chunks or segment windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.nlp_sidecar import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEGMENT_MANIFEST,
    SUPPORTED_CHUNK_TYPES,
    default_nlp_dir,
    load_segment_inputs_from_chunks_csv,
    load_segment_inputs_from_chunks_jsonl,
    load_segment_inputs_from_manifest,
    mean_text_length,
    multimodal_nlp_defaults,
    score_segments_with_model,
    write_nlp_sidecar_outputs,
)


def parse_args() -> argparse.Namespace:
    defaults = multimodal_nlp_defaults()
    parser = argparse.ArgumentParser(
        description="Run optional FinBERT and supporting emotion-model scoring over transcript-side segments."
    )
    parser.add_argument(
        "--source-id",
        required=True,
        help="Stable source id used for output naming.",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--chunks-csv",
        help="CSV of transcript chunks, such as outputs/<source_id>/chunks_scored.csv.",
    )
    inputs.add_argument(
        "--chunks-jsonl",
        help="JSONL of transcript chunks.",
    )
    inputs.add_argument(
        "--segment-manifest",
        help=(
            "Segment manifest CSV to score. The repo's canonical path is "
            f"{DEFAULT_SEGMENT_MANIFEST.as_posix()}."
        ),
    )
    parser.add_argument(
        "--transcript-path",
        help="Timed transcript JSON used to resolve segment-manifest text windows.",
    )
    parser.add_argument(
        "--chunk-types",
        nargs="+",
        choices=sorted(SUPPORTED_CHUNK_TYPES),
        help="Optional chunk-type filter, for example prepared_remarks q_and_a_answer.",
    )
    parser.add_argument(
        "--primary-model",
        default=defaults["finbert_model"],
        help="Primary sidecar model id. Defaults to the configured FinBERT model.",
    )
    parser.add_argument(
        "--run-secondary-emotion",
        action="store_true",
        help="Also run the generic emotion support model as a secondary pass.",
    )
    parser.add_argument(
        "--secondary-model",
        default=defaults["emotion_model"],
        help="Secondary/supporting model id.",
    )
    parser.add_argument(
        "--device",
        default=defaults["device"],
        help="Model device. Defaults to the multimodal config device, usually cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--out-dir",
        help="Optional base output dir. Defaults to data/processed/multimodal/nlp/<source_id>/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunk_types = set(args.chunk_types) if args.chunk_types else None
    input_metadata: dict[str, object] = {}

    if args.chunks_csv:
        segment_rows = load_segment_inputs_from_chunks_csv(
            source_id=args.source_id,
            path=args.chunks_csv,
            chunk_types=chunk_types,
        )
        input_mode = "chunks_csv"
        input_metadata["input_path"] = str(Path(args.chunks_csv))
    elif args.chunks_jsonl:
        segment_rows = load_segment_inputs_from_chunks_jsonl(
            source_id=args.source_id,
            path=args.chunks_jsonl,
            chunk_types=chunk_types,
        )
        input_mode = "chunks_jsonl"
        input_metadata["input_path"] = str(Path(args.chunks_jsonl))
    else:
        manifest_path = args.segment_manifest or str(DEFAULT_SEGMENT_MANIFEST)
        segment_rows, manifest_metadata = load_segment_inputs_from_manifest(
            source_id=args.source_id,
            manifest_path=manifest_path,
            transcript_path=args.transcript_path,
            chunk_types=chunk_types,
        )
        input_mode = "segment_manifest"
        input_metadata.update(manifest_metadata)
        input_metadata["input_path"] = str(Path(manifest_path))
        if args.transcript_path:
            input_metadata["transcript_path"] = str(Path(args.transcript_path))

    if not segment_rows:
        raise RuntimeError(
            "No usable text rows were found for NLP sidecar scoring. "
            "If you are using a segment manifest, provide a timed transcript JSON or "
            "populate transcript_ref with real text."
        )

    input_metadata["segment_count"] = len(segment_rows)
    input_metadata["mean_text_length"] = mean_text_length(segment_rows)
    if chunk_types is not None:
        input_metadata["chunk_types_filter"] = sorted(chunk_types)

    scored_rows = score_segments_with_model(
        segment_rows=segment_rows,
        model_name=args.primary_model,
        model_role="primary_finbert",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if args.run_secondary_emotion:
        scored_rows.extend(
            score_segments_with_model(
                segment_rows=segment_rows,
                model_name=args.secondary_model,
                model_role="secondary_emotion",
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        )

    output_paths = write_nlp_sidecar_outputs(
        source_id=args.source_id,
        input_mode=input_mode,
        rows=scored_rows,
        out_dir=args.out_dir,
        input_metadata=input_metadata,
    )

    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "input_mode": input_mode,
                "rows_written": len(scored_rows),
                "models_run": sorted({row["model_name"] for row in scored_rows}),
                "out_dir": str(default_nlp_dir(source_id=args.source_id, out_dir=args.out_dir)),
                "artifacts": {key: str(value) for key, value in output_paths.items()},
                "notes": [
                    "Deterministic labels remain source of truth.",
                    "NLP outputs are supporting evidence only.",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
