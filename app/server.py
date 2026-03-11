from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request, send_file
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
    prepare_review_run,
    run_document_review,
    run_media_review,
)
from earnings_call_sentiment.summary_config import load_summary_config

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ui_runs"
CACHE_ROOT = REPO_ROOT / "cache" / "ui_runs"


def create_app() -> Flask:
    app_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(app_dir / "templates"),
        static_folder=str(app_dir / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            form_state=_default_form_state(),
            suggested_prompt=DEFAULT_UI_PROMPT,
            result=None,
            error=None,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )

    @app.post("/analyze")
    def analyze() -> str:
        form_state = _merge_form_state(request.form)
        review_run = None
        try:
            source_mode = form_state["source_mode"]
            source_label = _source_label(form_state)
            review_run = prepare_review_run(
                repo_root=REPO_ROOT,
                source_label=source_label,
                cache_base=CACHE_ROOT,
                out_base=OUTPUT_ROOT,
            )
            summary_config = load_summary_config(
                enabled=form_state["analysis_mode"] == "summary",
                provider=form_state["summary_provider"] or None,
                model=form_state["summary_model"] or None,
                base_url=form_state["summary_base_url"] or None,
                api_key_env=form_state["summary_api_key_env"] or None,
                timeout_s=float(form_state["summary_timeout_s"]),
            )

            if source_mode == "youtube":
                youtube_url = form_state["youtube_url"].strip()
                if not youtube_url:
                    raise RuntimeError("Provide a YouTube URL.")
                run_media_review(
                    review_run=review_run,
                    youtube_url=youtube_url,
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
                uploaded = request.files.get("media_file")
                if uploaded is None or not uploaded.filename:
                    raise RuntimeError("Upload an audio or video file.")
                suffix = Path(uploaded.filename).suffix.lower()
                if suffix not in SUPPORTED_MEDIA_SUFFIXES:
                    raise RuntimeError(
                        "Unsupported media type. Use one of: "
                        + ", ".join(sorted(SUPPORTED_MEDIA_SUFFIXES))
                    )
                file_name = secure_filename(uploaded.filename)
                input_path = review_run.input_dir / file_name
                uploaded.save(input_path)
                run_media_review(
                    review_run=review_run,
                    youtube_url=None,
                    audio_path=input_path,
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
                doc_text = form_state["document_text"].strip()
                uploaded = request.files.get("document_file")
                if uploaded is not None and uploaded.filename:
                    suffix = Path(uploaded.filename).suffix.lower()
                    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
                        raise RuntimeError(
                            "Unsupported document type. Use one of: "
                            + ", ".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
                        )
                    file_name = secure_filename(uploaded.filename)
                    input_path = review_run.input_dir / file_name
                    uploaded.save(input_path)
                    doc_text = extract_text_from_document(input_path)
                if not doc_text:
                    raise RuntimeError("Upload a document or paste transcript text.")
                run_document_review(
                    text=doc_text,
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

            result = load_artifact_bundle(review_run)
            return render_template(
                "index.html",
                form_state=form_state,
                suggested_prompt=DEFAULT_UI_PROMPT,
                result=result,
                error=None,
                generated_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            return render_template(
                "index.html",
                form_state=form_state,
                suggested_prompt=DEFAULT_UI_PROMPT,
                result=None,
                error=str(exc),
                generated_at=datetime.now().isoformat(timespec="seconds"),
            ), 400

    @app.get("/runs/<run_id>/<path:filename>")
    def serve_artifact(run_id: str, filename: str):
        run_dir = (OUTPUT_ROOT / run_id).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            abort(404)
        target = (run_dir / filename).resolve()
        if run_dir not in target.parents and target != run_dir:
            abort(404)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_file(target)

    return app


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


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "7860")), debug=False)
