from __future__ import annotations

import argparse
import json
from pathlib import Path

from earnings_call_sentiment.alignment_sidecar import (
    AlignmentRunConfig,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPERX_MODEL,
    default_alignment_dir,
    run_whisperx_alignment,
)
from earnings_call_sentiment.optional_runtime import load_multimodal_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optional WhisperX alignment sidecar with optional pyannote diarization."
    )
    parser.add_argument("--source-id", required=True, help="Stable source identifier for output folder naming.")
    parser.add_argument("--audio-path", required=True, help="Local audio file path.")
    parser.add_argument(
        "--transcript-path",
        default=None,
        help="Optional transcript file path (.json from transcript.json or plain text file).",
    )
    parser.add_argument(
        "--transcript-text",
        default=None,
        help="Optional raw transcript text. If provided, it takes precedence over --transcript-path.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code used for alignment when a transcript is provided directly.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cpu", "cuda"),
        help="Execution device. Defaults to EARNINGS_CALL_MULTIMODAL_DEVICE or cpu.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_WHISPERX_MODEL,
        help="WhisperX ASR model used only when no transcript is provided.",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="WhisperX compute type, e.g. int8 or float16.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="WhisperX batch size used only when no transcript is provided.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional base output directory. Defaults to data/processed/multimodal/alignment/<source_id>/.",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable experimental pyannote diarization sidecar if explicitly configured.",
    )
    parser.add_argument("--min-speakers", type=int, default=None, help="Optional lower bound for diarization.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Optional upper bound for diarization.")
    parser.add_argument(
        "--return-char-alignments",
        action="store_true",
        help="Ask WhisperX to include char-level alignments in the sidecar JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    multimodal_config = load_multimodal_config()
    config = AlignmentRunConfig(
        source_id=str(args.source_id).strip(),
        audio_path=Path(args.audio_path).expanduser().resolve(),
        transcript_path=(Path(args.transcript_path).expanduser().resolve() if args.transcript_path else None),
        transcript_text=args.transcript_text,
        language=str(args.language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE,
        device=args.device or multimodal_config.multimodal_device or "cpu",
        whisperx_model=str(args.model or DEFAULT_WHISPERX_MODEL).strip() or DEFAULT_WHISPERX_MODEL,
        compute_type=str(args.compute_type or DEFAULT_COMPUTE_TYPE).strip() or DEFAULT_COMPUTE_TYPE,
        batch_size=max(1, int(args.batch_size)),
        out_dir=default_alignment_dir(source_id=str(args.source_id).strip(), out_dir=args.out_dir),
        enable_diarization=bool(args.enable_diarization),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        return_char_alignments=bool(args.return_char_alignments),
    )

    summary = run_whisperx_alignment(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
