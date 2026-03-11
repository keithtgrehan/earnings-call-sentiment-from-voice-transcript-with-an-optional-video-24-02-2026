from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from earnings_call_sentiment.pipeline.run import (
    DEFAULT_SENTIMENT_MODEL_NAME,
    DEFAULT_SENTIMENT_MODEL_REVISION,
)
from earnings_call_sentiment.review_workflow import (
    DEFAULT_UI_PROMPT,
    SUPPORTED_DOCUMENT_SUFFIXES,
    SUPPORTED_MEDIA_SUFFIXES,
    extract_text_from_document,
    load_artifact_bundle,
    load_artifact_bundle_for_dir,
    prepare_review_run,
    run_document_review,
    run_media_review,
)
from earnings_call_sentiment.summary_config import load_summary_config

JOB_LOCK = Lock()
JOBS: dict[str, dict[str, Any]] = {}


def create_review_app(
    *,
    template_dir: Path,
    static_dir: Path,
    repo_root: Path,
    output_root: Path | None = None,
    cache_root: Path | None = None,
    benchmark_root: Path | None = None,
    ui_meta: dict[str, str] | None = None,
) -> Flask:
    resolved_repo = repo_root.resolve()
    resolved_output_root = (output_root or (resolved_repo / "outputs" / "ui_runs")).resolve()
    resolved_cache_root = (cache_root or (resolved_repo / "cache" / "ui_runs")).resolve()
    resolved_benchmark_root = (benchmark_root or (resolved_repo / "data" / "gold_guidance_calls")).resolve()
    metadata = {
        "title": "Earnings Call Review Lab",
        "eyebrow": "Local review console",
        "lede": (
            "Run the existing deterministic pipeline against YouTube, local media, or a "
            "transcript document. Optional narrative mode stays additive and never replaces "
            "the core artifacts."
        ),
        "variant": "default",
    }
    if ui_meta:
        metadata.update(ui_meta)

    app = Flask(
        __name__,
        template_folder=str(template_dir.resolve()),
        static_folder=str(static_dir.resolve()),
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
    app.config["REPO_ROOT"] = str(resolved_repo)
    app.config["OUTPUT_ROOT"] = str(resolved_output_root)
    app.config["CACHE_ROOT"] = str(resolved_cache_root)
    app.config["BENCHMARK_ROOT"] = str(resolved_benchmark_root)
    app.config["UI_META"] = metadata

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            form_state=_default_form_state(),
            suggested_prompt=DEFAULT_UI_PROMPT,
            result=None,
            job=None,
            benchmark_rows=_load_benchmark_subset(resolved_benchmark_root),
            recent_runs=_load_recent_runs(resolved_output_root),
            ui_meta=metadata,
            error=None,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )

    @app.post("/analyze")
    def analyze():
        form_state = _merge_form_state(request.form)
        review_run = None
        try:
            review_run = prepare_review_run(
                repo_root=resolved_repo,
                source_label=_source_label(form_state),
                cache_base=resolved_cache_root,
                out_base=resolved_output_root,
            )
            job_payload = _prepare_job_payload(form_state, request, review_run)
            _register_job(review_run=review_run, form_state=form_state)
            _start_review_job(review_run=review_run, form_state=form_state, payload=job_payload)
            return redirect(url_for("review_run", run_id=review_run.run_id), code=303)
        except Exception as exc:
            return render_template(
                "index.html",
                form_state=form_state,
                suggested_prompt=DEFAULT_UI_PROMPT,
                result=None,
                job=None,
                benchmark_rows=_load_benchmark_subset(resolved_benchmark_root),
                recent_runs=_load_recent_runs(resolved_output_root),
                ui_meta=metadata,
                error=str(exc),
                generated_at=datetime.now().isoformat(timespec="seconds"),
            ), 400

    @app.get("/review/<run_id>")
    def review_run(run_id: str) -> str:
        job = _get_job(run_id)
        if job is None:
            # support viewing completed historical runs directly
            historical_dir = resolved_output_root / run_id
            if not historical_dir.exists():
                abort(404)
            result = load_artifact_bundle_for_dir(
                out_dir=historical_dir,
                run_id=run_id,
                cache_dir=resolved_cache_root / run_id,
            )
            return render_template(
                "index.html",
                form_state=_default_form_state(),
                suggested_prompt=DEFAULT_UI_PROMPT,
                result=result,
                job={"run_id": run_id, "status": "complete", "form_state": _default_form_state()},
                benchmark_rows=_load_benchmark_subset(resolved_benchmark_root),
                recent_runs=_load_recent_runs(resolved_output_root),
                ui_meta=metadata,
                error=None,
                generated_at=datetime.now().isoformat(timespec="seconds"),
            )
        result = None
        if job["status"] == "complete":
            result = load_artifact_bundle(job["review_run"])
        return render_template(
            "index.html",
            form_state=job["form_state"],
            suggested_prompt=DEFAULT_UI_PROMPT,
            result=result,
            job=job,
            benchmark_rows=_load_benchmark_subset(resolved_benchmark_root),
            recent_runs=_load_recent_runs(resolved_output_root),
            ui_meta=metadata,
            error=(job.get("error") if job["status"] == "error" else None),
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )

    @app.get("/runs/<run_id>/<path:filename>")
    def serve_artifact(run_id: str, filename: str):
        run_dir = (resolved_output_root / run_id).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            abort(404)
        target = (run_dir / filename).resolve()
        if run_dir not in target.parents and target != run_dir:
            abort(404)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_file(target)

    return app


def _load_benchmark_subset(benchmark_root: Path, limit: int = 5) -> list[dict[str, Any]]:
    manifest_path = benchmark_root / "call_manifest.csv"
    labels_path = benchmark_root / "labels.csv"
    if not labels_path.exists():
        return []
    labels = pd.read_csv(labels_path)
    if labels.empty:
        return []
    merged = labels.copy()
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if not manifest.empty:
            merged = merged.merge(
                manifest[["call_id", "source_url", "benchmark_quality_flag", "notes"]],
                on="call_id",
                how="left",
            )
    label_priority = {"raised": 0, "lowered": 1, "maintained": 2, "withdrawn": 3, "unclear": 4}
    quality_priority = {"good": 0, "acceptable": 1, "caution": 2, "poor": 3}
    merged["label_rank"] = merged["guidance_change_label"].map(label_priority).fillna(9)
    merged["quality_rank"] = merged["benchmark_quality_flag"].map(quality_priority).fillna(9)
    merged = merged.sort_values(["label_rank", "quality_rank", "call_id"]).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "call_id": str(row["call_id"]),
                "ticker": str(row["ticker"]),
                "company": str(row["company"]),
                "quarter": str(row["quarter"]),
                "label": str(row.get("guidance_change_label", "")),
                "quality": str(row.get("benchmark_quality_flag", "")),
                "source_url": str(row.get("source_url", "")),
                "notes": str(row.get("notes", "")),
                "transcript_path": str(row.get("source_path", "")),
            }
        )
    return rows


def _load_recent_runs(output_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    if not output_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        run_meta = path / "run_meta.json"
        report = path / "report.md"
        metrics = path / "metrics.json"
        row = {
            "run_id": path.name,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "symbol": "UNKNOWN",
            "event_dt": "",
            "has_report": report.exists() and report.stat().st_size > 0,
            "has_metrics": metrics.exists() and metrics.stat().st_size > 0,
        }
        if run_meta.exists() and run_meta.stat().st_size > 0:
            try:
                import json

                payload = json.loads(run_meta.read_text(encoding="utf-8"))
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                row["symbol"] = str(payload.get("symbol", "UNKNOWN"))
                row["event_dt"] = str(payload.get("event_dt", ""))
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _register_job(*, review_run, form_state: dict[str, Any]) -> None:
    with JOB_LOCK:
        JOBS[review_run.run_id] = {
            "run_id": review_run.run_id,
            "status": "queued",
            "error": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": None,
            "finished_at": None,
            "form_state": form_state,
            "review_run": review_run,
        }


def _set_job_state(run_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = JOBS.get(run_id)
        if job is None:
            return
        job.update(updates)


def _get_job(run_id: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        job = JOBS.get(run_id)
        return dict(job) if job is not None else None


def _prepare_job_payload(form_state: dict[str, Any], request_obj, review_run) -> dict[str, Any]:
    source_mode = form_state["source_mode"]
    payload: dict[str, Any] = {"source_mode": source_mode}
    if source_mode == "youtube":
        youtube_url = form_state["youtube_url"].strip()
        if not youtube_url:
            raise RuntimeError("Provide a YouTube URL.")
        payload["youtube_url"] = youtube_url
        return payload

    if source_mode == "media":
        uploaded = request_obj.files.get("media_file")
        if uploaded is None or not uploaded.filename:
            raise RuntimeError("Upload an audio or video file.")
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in SUPPORTED_MEDIA_SUFFIXES:
            raise RuntimeError(
                "Unsupported media type. Use one of: " + ", ".join(sorted(SUPPORTED_MEDIA_SUFFIXES))
            )
        file_name = secure_filename(uploaded.filename)
        input_path = review_run.input_dir / file_name
        uploaded.save(input_path)
        payload["audio_path"] = str(input_path)
        return payload

    doc_text = form_state["document_text"].strip()
    uploaded = request_obj.files.get("document_file")
    if uploaded is not None and uploaded.filename:
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise RuntimeError(
                "Unsupported document type. Use one of: " + ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
            )
        file_name = secure_filename(uploaded.filename)
        input_path = review_run.input_dir / file_name
        uploaded.save(input_path)
        doc_text = extract_text_from_document(input_path)
    if not doc_text:
        raise RuntimeError("Upload a document or paste transcript text.")
    payload["document_text"] = doc_text
    return payload


def _start_review_job(*, review_run, form_state: dict[str, Any], payload: dict[str, Any]) -> None:
    thread = Thread(
        target=_run_review_job,
        kwargs={
            "review_run": review_run,
            "form_state": dict(form_state),
            "payload": dict(payload),
        },
        daemon=True,
    )
    thread.start()


def _run_review_job(*, review_run, form_state: dict[str, Any], payload: dict[str, Any]) -> None:
    _set_job_state(review_run.run_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
    try:
        summary_config = load_summary_config(
            enabled=form_state["analysis_mode"] == "summary",
            provider=form_state["summary_provider"] or None,
            model=form_state["summary_model"] or None,
            base_url=form_state["summary_base_url"] or None,
            api_key_env=form_state["summary_api_key_env"] or None,
            timeout_s=float(form_state["summary_timeout_s"]),
        )
        source_mode = payload["source_mode"]
        if source_mode == "youtube":
            run_media_review(
                review_run=review_run,
                youtube_url=str(payload["youtube_url"]),
                audio_path=None,
                symbol=form_state["symbol"],
                event_dt=form_state["event_dt"],
                audio_format=form_state["audio_format"],
                model=form_state["whisper_model"],
                device=form_state["device"],
                compute_type=form_state["compute_type"],
                chunk_seconds=float(form_state["chunk_seconds"]),
                vad=form_state["vad"],
                prior_guidance_path=form_state["prior_guidance_path"] or None,
                tone_change_threshold=float(form_state["tone_change_threshold"]),
                sentiment_model=form_state["sentiment_model"],
                sentiment_revision=form_state["sentiment_revision"],
                question_shift_enabled=form_state["question_shifts"],
                pre_window_s=float(form_state["pre_window_s"]),
                post_window_s=float(form_state["post_window_s"]),
                min_gap_s=float(form_state["min_gap_s"]),
                min_chars=int(form_state["min_chars"]),
                summary_config=summary_config,
                verbose=form_state["verbose"],
            )
        elif source_mode == "media":
            run_media_review(
                review_run=review_run,
                youtube_url=None,
                audio_path=Path(str(payload["audio_path"])),
                symbol=form_state["symbol"],
                event_dt=form_state["event_dt"],
                audio_format=form_state["audio_format"],
                model=form_state["whisper_model"],
                device=form_state["device"],
                compute_type=form_state["compute_type"],
                chunk_seconds=float(form_state["chunk_seconds"]),
                vad=form_state["vad"],
                prior_guidance_path=form_state["prior_guidance_path"] or None,
                tone_change_threshold=float(form_state["tone_change_threshold"]),
                sentiment_model=form_state["sentiment_model"],
                sentiment_revision=form_state["sentiment_revision"],
                question_shift_enabled=form_state["question_shifts"],
                pre_window_s=float(form_state["pre_window_s"]),
                post_window_s=float(form_state["post_window_s"]),
                min_gap_s=float(form_state["min_gap_s"]),
                min_chars=int(form_state["min_chars"]),
                summary_config=summary_config,
                verbose=form_state["verbose"],
            )
        else:
            run_document_review(
                text=str(payload["document_text"]),
                review_run=review_run,
                symbol=form_state["symbol"],
                event_dt=form_state["event_dt"],
                prior_guidance_path=form_state["prior_guidance_path"] or None,
                tone_change_threshold=float(form_state["tone_change_threshold"]),
                sentiment_model=form_state["sentiment_model"],
                sentiment_revision=form_state["sentiment_revision"],
                question_shift_enabled=form_state["question_shifts"],
                pre_window_s=float(form_state["pre_window_s"]),
                post_window_s=float(form_state["post_window_s"]),
                min_gap_s=float(form_state["min_gap_s"]),
                min_chars=int(form_state["min_chars"]),
                summary_config=summary_config,
                verbose=form_state["verbose"],
            )
        _set_job_state(review_run.run_id, status="complete", finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as exc:
        _set_job_state(review_run.run_id, status="error", error=str(exc), finished_at=datetime.now().isoformat(timespec="seconds"))


def _default_form_state() -> dict[str, Any]:
    return {
        "source_mode": "youtube",
        "analysis_mode": "deterministic",
        "youtube_url": "",
        "symbol": "UNKNOWN",
        "event_dt": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "audio_format": "wav",
        "whisper_model": "small",
        "device": "auto",
        "compute_type": "int8",
        "chunk_seconds": "30",
        "tone_change_threshold": "2.0",
        "prior_guidance_path": "",
        "sentiment_model": DEFAULT_SENTIMENT_MODEL_NAME,
        "sentiment_revision": DEFAULT_SENTIMENT_MODEL_REVISION,
        "summary_provider": "openai_compatible",
        "summary_model": os.getenv("ECS_SUMMARY_MODEL", "gpt-4.1-mini"),
        "summary_base_url": os.getenv("ECS_SUMMARY_BASE_URL", "https://api.openai.com/v1"),
        "summary_api_key_env": os.getenv("ECS_SUMMARY_API_KEY_ENV", "OPENAI_API_KEY"),
        "summary_timeout_s": "60",
        "question_shifts": True,
        "vad": False,
        "verbose": True,
        "pre_window_s": "60",
        "post_window_s": "120",
        "min_gap_s": "30",
        "min_chars": "15",
        "document_text": "",
    }


def _merge_form_state(form_data: Any) -> dict[str, Any]:
    state = _default_form_state()
    for key in state:
        if key in {"question_shifts", "vad", "verbose"}:
            state[key] = form_data.get(key) == "on"
        else:
            state[key] = form_data.get(key, state[key])
    return state


def _source_label(form_state: dict[str, Any]) -> str:
    source_mode = form_state["source_mode"]
    if source_mode == "youtube":
        return form_state["symbol"] or form_state["youtube_url"] or "youtube"
    if source_mode == "media":
        return form_state["symbol"] or "media"
    return form_state["symbol"] or "document"
