import importlib


def test_imports():
    mods = [
        "earnings_call_sentiment",
        "earnings_call_sentiment.cli",
        "earnings_call_sentiment.downloader",
        "earnings_call_sentiment.transcriber",
        "earnings_call_sentiment.sentiment",
    ]
    for m in mods:
        importlib.import_module(m)
