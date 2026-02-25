from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

from transformers import pipeline


@dataclass(frozen=True)
class SentimentScore:
    chunk_id: int
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: int = -1,  # -1 cpu, 0 gpu
):
    return pipeline("sentiment-analysis", model=model_name, device=device)


def score_texts(
    texts: list[str],
    *,
    model_name: str,
    device: int = -1,
    batch_size: int = 8,
    truncation: bool = True,
    max_length: int = 256,
) -> list[dict[str, Any]]:
    """
    Returns raw HF pipeline outputs, one per input text:
      {"label": "...", "score": 0.99}
    """
    pipe = build_sentiment_pipeline(model_name=model_name, device=device)
    return pipe(
        texts,
        batch_size=batch_size,
        truncation=truncation,
        max_length=max_length,
    )


def score_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    model_name: str,
    device: int = -1,
    batch_size: int = 8,
    max_length: int = 256,
) -> Iterator[dict[str, Any]]:
    """
    Streams over chunks dicts (must include chunk_id and text).
    Yields merged dict with sentiment fields added.
    """
    pipe = build_sentiment_pipeline(model_name=model_name, device=device)

    batch: list[dict[str, Any]] = []
    texts: list[str] = []

    def flush():
        nonlocal batch, texts
        if not batch:
            return
        outs = pipe(
            texts,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length,
        )
        for ch, out in zip(batch, outs, strict=True):
            merged = dict(ch)
            merged["sentiment_label"] = out["label"]
            merged["sentiment_score"] = float(out["score"])
            yield merged
        batch = []
        texts = []

    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        batch.append(ch)
        texts.append(text)
        if len(batch) >= batch_size:
            yield from flush()

    yield from flush()
