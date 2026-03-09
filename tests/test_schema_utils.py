from __future__ import annotations

from earnings_call_sentiment.schema_utils import normalize_summary_payload, safe_json_loads


def test_malformed_json_fences_and_wrapper_keys_are_normalized() -> None:
    text = """
    Notes before output.
    ```json
    {
      "summary": {
        "executiveSummary": "Revenue guidance tightened with mixed tone.",
        "signals": ["Raised full-year range", "Q3 cautionary language"],
        "risk": ["Margin pressure"],
        "evidence_snippets": ["We are narrowing guidance to..."],
        "caveats": ["Single-call sample"]
      }
    }
    ```
    trailing commentary
    """

    parsed = safe_json_loads(text)
    assert isinstance(parsed, dict)

    normalized = normalize_summary_payload(parsed)
    assert normalized["executive_summary"].startswith("Revenue guidance tightened")
    assert normalized["key_signals"]
    assert normalized["risks"] == ["Margin pressure"]
    assert normalized["evidence"] == ["We are narrowing guidance to..."]
    assert normalized["limitations"] == ["Single-call sample"]


def test_safe_json_loads_returns_fallback_payload_on_failure() -> None:
    parsed = safe_json_loads("not-json-at-all")
    assert isinstance(parsed, dict)

    normalized = normalize_summary_payload(parsed)
    assert normalized["executive_summary"]
    assert isinstance(normalized["key_signals"], list)
    assert isinstance(normalized["limitations"], list)
    assert normalized["limitations"]


def test_wrapper_aliases_result_data_output_are_supported() -> None:
    wrapped = safe_json_loads(
        '{"result":{"output":{"data":{"executive_summary":"OK","signals":["s1"]}}}}'
    )
    assert isinstance(wrapped, dict)
    normalized = normalize_summary_payload(wrapped)
    assert normalized["executive_summary"] == "OK"
    assert normalized["key_signals"] == ["s1"]
