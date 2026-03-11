from __future__ import annotations

import importlib.util
from pathlib import Path

from earnings_call_sentiment.review_workflow import ReviewRun


def _load_server_module():
    module_path = Path(__file__).resolve().parents[1] / "app" / "server.py"
    spec = importlib.util.spec_from_file_location("review_app_server", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_app_index_renders() -> None:
    server = _load_server_module()
    app = server.create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Earnings Call Review Lab" in text
    assert "Deterministic only" in text


def test_review_app_document_post_uses_review_workflow(monkeypatch, tmp_path: Path) -> None:
    server = _load_server_module()
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

    monkeypatch.setattr(server, "prepare_review_run", fake_prepare_review_run)
    monkeypatch.setattr(server, "run_document_review", fake_run_document_review)
    monkeypatch.setattr(server, "load_artifact_bundle", fake_load_artifact_bundle)

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
    )

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "demo-run" in text
    assert called["document_text"] == "We raised revenue guidance for the year."
