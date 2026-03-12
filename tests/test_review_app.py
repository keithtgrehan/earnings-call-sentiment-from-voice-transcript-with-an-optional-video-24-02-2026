from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_app_module(module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "app" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"review_app_{module_name}", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_app_index_renders() -> None:
    server = _load_app_module("server")
    app = server.create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Earnings Call Review Lab" in text
    assert "Deterministic only" in text
    assert "Run the existing deterministic pipeline" in text


def test_site_app_index_renders() -> None:
    site_server = _load_app_module("site_server")
    app = site_server.create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Earnings Call Signal Engine" in text
    assert "Transcript-first AI review" in text
    assert (
        "Transcript-first AI tool for extracting structured signals from earnings call "
        "audio and video sources using NLP." in text
    )
    assert "Recent local runs" in text


def test_review_app_document_post_uses_review_workflow(monkeypatch, tmp_path: Path) -> None:
    import earnings_call_sentiment.web_backend as web_backend
    from earnings_call_sentiment.review_workflow import ReviewRun

    server = _load_app_module("server")
    app = server.create_app()
    review_run = ReviewRun(
        run_id="demo-run",
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "outputs",
        input_dir=tmp_path / "outputs" / "inputs",
    )
    review_run.input_dir.mkdir(parents=True, exist_ok=True)
    review_run.out_dir.mkdir(parents=True, exist_ok=True)

    called: dict[str, object] = {}

    def fake_prepare_review_run(*, repo_root, source_label, cache_base=None, out_base=None):
        called["source_label"] = source_label
        return review_run

    def fake_run_document_review(**kwargs):
        called["document_text"] = kwargs["text"]
        (review_run.out_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (review_run.out_dir / "report.md").write_text("# Report", encoding="utf-8")
        (review_run.out_dir / "transcript.txt").write_text("text", encoding="utf-8")
        return {"run_id": review_run.run_id}

    def fake_load_artifact_bundle(arg):
        assert arg == review_run
        return {
            "run_id": review_run.run_id,
            "out_dir": str(review_run.out_dir),
            "artifacts": {"metrics.json": str(review_run.out_dir / 'metrics.json')},
            "tables": {},
            "json": {"metrics.json": {"sentiment_mean": 0.0}},
            "text": {"report.md": "# Report", "transcript.txt": "text"},
        }

    def fake_start_review_job(*, review_run, form_state, payload):
        fake_run_document_review(text=payload["document_text"])
        web_backend._set_job_state(
            review_run.run_id,
            status="complete",
            finished_at="2026-03-11T09:05:00",
        )

    monkeypatch.setattr(web_backend, "prepare_review_run", fake_prepare_review_run)
    monkeypatch.setattr(web_backend, "load_artifact_bundle", fake_load_artifact_bundle)
    monkeypatch.setattr(web_backend, "_start_review_job", fake_start_review_job)

    client = app.test_client()
    response = client.post(
        "/analyze",
        data={
            "source_mode": "document",
            "analysis_mode": "deterministic",
            "symbol": "TEST",
            "event_dt": "2026-03-11T09:00:00-05:00",
            "document_text": "We raised revenue guidance for the year.",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["Location"].endswith("/review/demo-run")

    review_response = client.get("/review/demo-run")
    assert review_response.status_code == 200
    text = review_response.get_data(as_text=True)
    assert "demo-run" in text
    assert called["document_text"] == "We raised revenue guidance for the year."


def test_site_app_review_page_renders_reviewer_scorecard(monkeypatch, tmp_path: Path) -> None:
    import earnings_call_sentiment.web_backend as web_backend
    from earnings_call_sentiment.review_workflow import ReviewRun

    site_server = _load_app_module("site_server")
    app = site_server.create_app()
    review_run = ReviewRun(
        run_id="demo-run",
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "outputs",
        input_dir=tmp_path / "outputs" / "inputs",
    )
    review_run.input_dir.mkdir(parents=True, exist_ok=True)
    review_run.out_dir.mkdir(parents=True, exist_ok=True)

    def fake_prepare_review_run(*, repo_root, source_label, cache_base=None, out_base=None):
        return review_run

    def fake_start_review_job(*, review_run, form_state, payload):
        (review_run.out_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (review_run.out_dir / "report.md").write_text("# Report", encoding="utf-8")
        (review_run.out_dir / "transcript.txt").write_text("text", encoding="utf-8")
        web_backend._set_job_state(
            review_run.run_id,
            status="complete",
            finished_at="2026-03-12T08:00:00",
        )

    def fake_load_artifact_bundle(arg):
        assert arg == review_run
        return {
            "run_id": review_run.run_id,
            "out_dir": str(review_run.out_dir),
            "artifacts": {"metrics.json": str(review_run.out_dir / "metrics.json")},
            "tables": {},
            "json": {
                "metrics.json": {
                    "overall_review_signal": "green",
                    "review_confidence_pct": 84,
                    "review_scorecard": {
                        "overall_review_signal": "green",
                        "overall_score": 7.8,
                        "review_confidence_pct": 84,
                        "confidence_note": "Confidence reflects how clear and well-supported the deterministic interpretation is, not investment conviction.",
                        "ranked_categories": [
                            {
                                "rank": 1,
                                "name": "Guidance Strength",
                                "score": 9,
                                "color_band": "green",
                                "explanation": "Guidance reads stronger versus prior guidance.",
                                "strongest_evidence": ["raised: We raised revenue guidance for the year."],
                            }
                        ],
                    },
                }
            },
            "text": {"report.md": "# Report", "transcript.txt": "text"},
        }

    monkeypatch.setattr(web_backend, "prepare_review_run", fake_prepare_review_run)
    monkeypatch.setattr(web_backend, "_start_review_job", fake_start_review_job)
    monkeypatch.setattr(web_backend, "load_artifact_bundle", fake_load_artifact_bundle)

    client = app.test_client()
    post_response = client.post(
        "/analyze",
        data={
            "source_mode": "document",
            "analysis_mode": "deterministic",
            "symbol": "TEST",
            "event_dt": "2026-03-12T08:00:00+01:00",
            "document_text": "We raised revenue guidance for the year.",
        },
        follow_redirects=False,
    )

    assert post_response.status_code == 303
    response = client.get("/review/demo-run")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Reviewer scorecard" in text
    assert "At-a-glance review" in text
    assert "Guidance Strength" in text
    assert "84%" in text
