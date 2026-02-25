import argparse
import logging
from pathlib import Path

from earnings_call_sentiment.chunking import chunk_segments
from earnings_call_sentiment.downloader import download_audio
from earnings_call_sentiment.io_utils import read_jsonl, write_json, write_jsonl
from earnings_call_sentiment.sentiment import score_chunks
from earnings_call_sentiment.transcriber import transcribe_audio

log = logging.getLogger("earnings_call_sentiment")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="earnings-call-sentiment",
        description="Earnings call sentiment pipeline (YouTube URL input).",
    )

    p.add_argument("--youtube-url", required=True, help="YouTube URL for the earnings call.")
    p.add_argument(
        "--cache-dir", default="./cache", help="Directory for cached downloads/artifacts."
    )
    p.add_argument("--out-dir", default="./outputs", help="Directory for outputs (json/csv/txt).")
    p.add_argument(
        "--audio-format",
        default="wav",
        choices=["wav", "mp3", "m4a"],
        help="Preferred audio format to save.",
    )

    # pipeline switches
    p.add_argument("--download-only", action="store_true", help="Download audio only and exit.")
    p.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Download + transcribe + chunk, write outputs, then exit.",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Score existing chunks.jsonl and exit (no download/transcribe).",
    )

    # whisper
    p.add_argument(
        "--model",
        default="small",
        help="faster-whisper model size (tiny/base/small/medium/large-v3).",
    )
    p.add_argument("--device", default="auto", help="faster-whisper device (auto/cpu/cuda).")
    p.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type (auto/int8/float16/etc).",
    )

    # chunking
    p.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk window in seconds.")
    p.add_argument("--min-chars", type=int, default=1, help="Minimum chars per chunk to keep.")

    # sentiment
    p.add_argument(
        "--sentiment-model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HF sentiment model.",
    )
    p.add_argument(
        "--sentiment-device", type=int, default=-1, help="-1 cpu, 0 gpu (transformers pipeline)."
    )
    p.add_argument(
        "--sentiment-batch-size", type=int, default=8, help="Batch size for sentiment inference."
    )
    p.add_argument(
        "--sentiment-max-length",
        type=int,
        default=256,
        help="Tokenizer max_length (truncates longer chunks).",
    )

    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate args/paths and exit without running pipeline.",
    )
    return p


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("youtube_url=%s", args.youtube_url)
    log.info("cache_dir=%s", cache_dir.resolve())
    log.info("out_dir=%s", out_dir.resolve())
    log.info("audio_format=%s", args.audio_format)

    if args.dry_run:
        log.info("dry-run: exiting before pipeline execution")
        return 0

    # score-only mode: expects chunks.jsonl already produced
    if args.score_only:
        chunks_path = out_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise SystemExit(f"score-only requires {chunks_path} to exist")
        scored_path = out_dir / "chunks_scored.jsonl"
        scored = score_chunks(
            read_jsonl(chunks_path),
            model_name=args.sentiment_model,
            device=args.sentiment_device,
            batch_size=args.sentiment_batch_size,
            max_length=args.sentiment_max_length,
        )
        write_jsonl(scored_path, scored)
        log.info("wrote scored chunks to %s", scored_path.resolve())
        return 0

    # 1) download
    audio_path = download_audio(
        url=args.youtube_url,
        cache_dir=cache_dir,
        audio_format=args.audio_format,
    )
    log.info("audio saved to %s", audio_path)

    if args.download_only:
        log.info("download-only mode complete")
        return 0

    # 2) transcribe
    log.info(
        "transcribing with model=%s device=%s compute_type=%s",
        args.model,
        args.device,
        args.compute_type,
    )
    segments = transcribe_audio(
        audio_path=audio_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments_json = [s.to_dict() for s in segments]
    write_json(out_dir / "segments.json", segments_json)
    write_jsonl(out_dir / "segments.jsonl", segments_json)

    # 3) chunk
    chunks = chunk_segments(segments, window_seconds=args.chunk_seconds, min_chars=args.min_chars)
    chunks_json = [c.to_dict() for c in chunks]
    write_json(out_dir / "chunks.json", chunks_json)
    write_jsonl(out_dir / "chunks.jsonl", chunks_json)

    log.info("wrote outputs to %s", out_dir.resolve())

    if args.transcribe_only:
        log.info("transcribe-only mode complete")
        return 0

    # 4) sentiment score chunks
    scored_path = out_dir / "chunks_scored.jsonl"
    scored = score_chunks(
        iter(chunks_json),
        model_name=args.sentiment_model,
        device=args.sentiment_device,
        batch_size=args.sentiment_batch_size,
        max_length=args.sentiment_max_length,
    )
    write_jsonl(scored_path, scored)
    log.info("wrote scored chunks to %s", scored_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
