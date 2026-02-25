"""Canonical CLI entry point for earnings_call_sentiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from . import __version__
from earnings_call_sentiment.downloaders.youtube import download_audio
from earnings_call_sentiment.pipeline.run import (
    load_transcript_segments,
    normalize_audio_to_wav,
    run_pipeline,
    transcribe_audio,
    write_sentiment_artifacts,
    write_transcript_artifacts,
)
import earnings_call_sentiment.question_shifts as qs


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[verbose] {message}")


def _signed_score(label: str, score: float) -> float:
    normalized = label.strip().upper()
    if "NEG" in normalized:
        return -abs(score)
    if "POS" in normalized:
        return abs(score)
    return 0.0


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes:02d}:{remainder:02d}"


def _resolve_source_audio(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.audio_path:
        audio_path = Path(args.audio_path).expanduser().resolve()
        if not audio_path.exists() or not audio_path.is_file():
            raise RuntimeError(f"--audio-path not found: {audio_path}")
        return audio_path
    if args.youtube_url:
        return download_audio(
            youtube_url=args.youtube_url,
            cache_dir=cache_dir,
            audio_format=args.audio_format,
        )
    raise RuntimeError("Provide either --youtube-url or --audio-path.")


def _build_chunks_scored_df(sentiment_segments: list[dict[str, Any]]) -> pd.DataFrame:
    chunks_scored = pd.DataFrame(sentiment_segments)
    if chunks_scored.empty:
        return pd.DataFrame(
            columns=["start", "end", "text", "sentiment", "score", "signed_score"]
        )

    for column in ("start", "end", "score"):
        chunks_scored[column] = pd.to_numeric(
            chunks_scored[column], errors="coerce"
        ).fillna(0.0)
    labels = chunks_scored["sentiment"].astype(str).fillna("")
    chunks_scored["signed_score"] = [
        _signed_score(label, float(score))
        for label, score in zip(labels.tolist(), chunks_scored["score"].tolist())
    ]
    return chunks_scored


def _write_chunks_scored_jsonl(chunks_scored: pd.DataFrame, out_dir: Path) -> Path:
    jsonl_path = out_dir / "chunks_scored.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in chunks_scored.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonl_path


def _run_question_shift_analysis(
    segments: list[dict[str, Any]],
    chunks_scored_jsonl: Path,
    out_dir: Path,
    args: argparse.Namespace,
    console: Console,
) -> None:
    if not chunks_scored_jsonl.exists() or not chunks_scored_jsonl.is_file():
        raise RuntimeError(
            "Question shift analysis requires scored chunks jsonl, but it was not found: "
            f"{chunks_scored_jsonl}"
        )
    chunks_scored = pd.read_json(chunks_scored_jsonl, lines=True)
    question_df = qs.detect_question_shifts(
        segments=segments,
        chunks_scored=chunks_scored,
        before_window=float(args.pre_window_s),
        after_window=float(args.post_window_s),
        min_gap_s=float(args.min_gap_s),
        min_chars=int(args.min_chars),
    )
    question_csv_path = out_dir / "question_sentiment_shifts.csv"
    question_plot_path = out_dir / "question_shifts.png"
    question_df.to_csv(question_csv_path, index=False)
    fig = qs.plot_question_shifts(question_df, out_path=question_plot_path)
    fig.clear()

    console.print()
    console.print("[bold]Question Shift Artifacts[/bold]")
    console.print(f"[bold]Question Shift CSV:[/bold] {question_csv_path}")
    console.print(f"[bold]Question Shift Plot:[/bold] {question_plot_path}")
    console.print("[bold]Top 10 Most Negative Shifts[/bold]")
    if question_df.empty:
        console.print("No qualifying analyst questions detected.")
        return

    top_negative = question_df.nsmallest(10, "sentiment_shift")
    for _, row in top_negative.iterrows():
        time_label = _format_mmss(float(row["question_time"]))
        shift = float(row["sentiment_shift"])
        question_text = str(row["question_text"]).replace("\n", " ").strip()
        if len(question_text) > 140:
            question_text = f"{question_text[:137]}..."
        console.print(f"{time_label} | shift={shift:+.4f} | {question_text}")


def _print_outputs(console: Console, title: str, rows: list[tuple[str, str]]) -> None:
    console.print(f"[bold green]{title}[/bold green]")
    console.print()
    console.print("[bold]Output Files[/bold]")
    for label, value in rows:
        console.print(f"[bold]{label}:[/bold] {value}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="earnings-call-sentiment",
        description=(
            "Analyze earnings call sentiment from transcripts and optional video input."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--youtube-url",
        default=None,
        help="YouTube URL to process (required unless --audio-path is provided)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Local audio file path to process (skips YouTube download)",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory for downloaded/intermediate audio and model cache",
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs",
        help="Directory where transcript and output artifacts will be written",
    )
    parser.add_argument(
        "--audio-format",
        default="wav",
        choices=("wav", "mp3", "m4a"),
        help="Audio format for YouTube extraction (default: wav)",
    )
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument(
        "--device",
        default="auto",
        help="Whisper device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Whisper compute type (e.g. int8, float16)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Transcription chunk length in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate and print planned steps without running the pipeline",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep intermediate download/WAV artifacts.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        default=False,
        help="Download/extract audio then exit (no transcription).",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        default=False,
        help="Stop after transcript.json/transcript.txt are generated.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        default=False,
        help="Generate sentiment/risk artifacts from transcript segments.",
    )
    parser.add_argument(
        "--question-shifts",
        action="store_true",
        default=False,
        help="Detect question-related sentiment shifts and write CSV/PNG outputs.",
    )
    parser.add_argument(
        "--pre-window-s",
        type=float,
        default=60.0,
        help="Seconds before each question used for baseline sentiment.",
    )
    parser.add_argument(
        "--post-window-s",
        type=float,
        default=120.0,
        help="Seconds after each question used for post-question sentiment.",
    )
    parser.add_argument(
        "--min-gap-s",
        type=float,
        default=30.0,
        help="Minimum seconds between detected questions.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=15,
        help="Minimum question-text length for question-shift detection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    mode_flags = [args.download_only, args.transcribe_only, args.score_only]
    if sum(bool(flag) for flag in mode_flags) > 1:
        parser.error(
            "Use at most one of --download-only, --transcribe-only, --score-only"
        )

    if not args.youtube_url and not args.audio_path:
        parser.error("--youtube-url is required when --audio-path is not provided")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    console = Console()

    if args.verbose:
        _log(args.verbose, f"args={args}")

    if args.dry_run:
        print("Dry run enabled; skipping execution.")
        print(f"youtube_url={args.youtube_url}")
        print(f"audio_path={args.audio_path}")
        print(f"cache_dir={cache_dir}")
        print(f"out_dir={out_dir}")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        source_audio = _resolve_source_audio(args, cache_dir)
        print(f"Download complete: {source_audio}")
        return 0

    if args.transcribe_only:
        source_audio = _resolve_source_audio(args, cache_dir)
        _log(args.verbose, f"source_audio={source_audio}")
        normalized_wav = cache_dir / "audio_normalized.wav"
        normalize_audio_to_wav(source_audio, normalized_wav, verbose=args.verbose)
        segments = transcribe_audio(
            str(normalized_wav),
            verbose=args.verbose,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            chunk_seconds=float(args.chunk_seconds),
        )
        transcript_json, transcript_txt = write_transcript_artifacts(segments, out_dir)
        _print_outputs(
            console,
            "Transcription Complete",
            [
                ("Audio", str(normalized_wav)),
                ("Transcript JSON", str(transcript_json)),
                ("Transcript Text", str(transcript_txt)),
            ],
        )
        if args.question_shifts:
            console.print(
                "[yellow]Skipping --question-shifts in --transcribe-only mode "
                "(no sentiment scoring step).[/yellow]"
            )
        return 0

    if args.score_only:
        transcript_json = out_dir / "transcript.json"
        transcript_txt = out_dir / "transcript.txt"
        if transcript_json.exists() and transcript_json.is_file():
            segments = load_transcript_segments(transcript_json)
        else:
            source_audio = _resolve_source_audio(args, cache_dir)
            _log(args.verbose, f"source_audio={source_audio}")
            normalized_wav = cache_dir / "audio_normalized.wav"
            normalize_audio_to_wav(source_audio, normalized_wav, verbose=args.verbose)
            segments = transcribe_audio(
                str(normalized_wav),
                verbose=args.verbose,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                chunk_seconds=float(args.chunk_seconds),
            )
            transcript_json, transcript_txt = write_transcript_artifacts(
                segments, out_dir
            )
        if not transcript_txt.exists():
            transcript_txt.write_text(
                "\n".join(
                    item.get("text", "") for item in segments if item.get("text")
                ),
                encoding="utf-8",
            )

        artifacts = write_sentiment_artifacts(segments=segments, output_path=out_dir)
        sentiment_segments = artifacts["sentiment_segments"]
        chunks_scored_df = _build_chunks_scored_df(sentiment_segments)
        chunks_scored_csv = out_dir / "chunks_scored.csv"
        chunks_scored_df.to_csv(chunks_scored_csv, index=False)
        chunks_scored_jsonl = _write_chunks_scored_jsonl(chunks_scored_df, out_dir)

        _print_outputs(
            console,
            "Scoring Complete",
            [
                ("Transcript JSON", str(transcript_json)),
                ("Sentiment Segments", str(artifacts["sentiment_segments_csv"])),
                ("Sentiment Timeline", str(artifacts["sentiment_timeline_png"])),
                ("Risk Metrics", str(artifacts["risk_metrics_json"])),
            ],
        )
        if args.question_shifts:
            _run_question_shift_analysis(
                segments=segments,
                chunks_scored_jsonl=chunks_scored_jsonl,
                out_dir=out_dir,
                args=args,
                console=console,
            )
        return 0

    resolved_audio_path = None
    resolved_youtube_url = args.youtube_url
    if args.audio_path:
        resolved_audio = Path(args.audio_path).expanduser().resolve()
        if not resolved_audio.exists() or not resolved_audio.is_file():
            parser.error(f"--audio-path not found: {resolved_audio}")
        resolved_audio_path = str(resolved_audio)
        resolved_youtube_url = None

    result = run_pipeline(
        youtube_url=resolved_youtube_url,
        audio_path=resolved_audio_path,
        cache_dir=str(cache_dir),
        out_dir=str(out_dir),
        verbose=bool(args.verbose),
        audio_format=args.audio_format,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        chunk_seconds=float(args.chunk_seconds),
    )

    sentiment_segments_path = Path(str(result["sentiment_segments_csv"]))
    sentiment_df = pd.read_csv(sentiment_segments_path)
    sentiment_records = sentiment_df.to_dict(orient="records")
    chunks_scored_df = _build_chunks_scored_df(sentiment_records)
    chunks_scored_csv = out_dir / "chunks_scored.csv"
    chunks_scored_df.to_csv(chunks_scored_csv, index=False)
    chunks_scored_jsonl = _write_chunks_scored_jsonl(chunks_scored_df, out_dir)

    _print_outputs(
        console,
        "Earnings Call Analysis Complete",
        [
            ("Audio", str(result["audio"])),
            ("Transcript JSON", str(result["transcript_json"])),
            ("Transcript Text", str(result["transcript_txt"])),
            ("Sentiment Segments", str(result["sentiment_segments_csv"])),
            ("Sentiment Timeline", str(result["sentiment_timeline_png"])),
            ("Risk Metrics", str(result["risk_metrics_json"])),
        ],
    )

    if args.question_shifts:
        segments = load_transcript_segments(Path(str(result["transcript_json"])))
        _run_question_shift_analysis(
            segments=segments,
            chunks_scored_jsonl=chunks_scored_jsonl,
            out_dir=out_dir,
            args=args,
            console=console,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
