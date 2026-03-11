from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from earnings_call_sentiment.pipeline import run as run_module


def test_build_sentiment_pipeline_passes_model_and_revision(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    tokenizer_calls: dict[str, Any] = {}

    class DummyPipeline:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                name_or_path="dummy/model",
                config=SimpleNamespace(_commit_hash="dummyrev"),
            )

        def __call__(self, _: str):
            return [{"label": "POSITIVE", "score": 0.9}]

    class DummyTokenizer:
        pass

    def fake_from_pretrained(model_name: str, **kwargs: Any):
        tokenizer_calls["model_name"] = model_name
        tokenizer_calls.update(kwargs)
        return DummyTokenizer()

    def fake_hf_pipeline(task: str, **kwargs: Any):
        captured["task"] = task
        captured.update(kwargs)
        return DummyPipeline()

    monkeypatch.setattr(run_module.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(run_module, "hf_pipeline", fake_hf_pipeline)
    pipeline_obj = run_module.build_sentiment_pipeline(
        sentiment_model="unit/model",
        sentiment_revision="abc123",
    )
    assert callable(pipeline_obj)
    assert captured["task"] == "sentiment-analysis"
    assert captured["model"] == "unit/model"
    assert captured["revision"] == "abc123"
    assert isinstance(captured["tokenizer"], DummyTokenizer)
    assert tokenizer_calls["model_name"] == "unit/model"
    assert tokenizer_calls["revision"] == "abc123"
