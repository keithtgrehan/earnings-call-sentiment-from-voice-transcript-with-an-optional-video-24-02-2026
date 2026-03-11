from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_eval_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_gold_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_gold_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_sentence_marks_comparative_upgrade_as_raised() -> None:
    mod = _load_eval_module()
    sentence = (
        "We now expect CapEx to be in the range of $91 billion to $93 billion in 2025, "
        "up from our previous estimate of $85 billion."
    )
    assert mod._classify_sentence(sentence) == "raised"


def test_classify_sentence_keeps_nondirectional_outlook_unclear() -> None:
    mod = _load_eval_module()
    sentence = (
        "We expect continued momentum from Windows 10 end of support, although growth "
        "rates will be impacted by elevated inventory levels at the end of Q1."
    )
    assert mod._classify_sentence(sentence) == "unclear"


def test_classify_sentence_preserves_existing_directional_patterns() -> None:
    mod = _load_eval_module()
    assert mod._classify_sentence("Yeah, so our guidance is flat.") == "maintained"
    assert (
        mod._classify_sentence(
            "That's why we have to adjust our full year non-GAAP guidance down for both EBIT margin and EPS."
        )
        == "lowered"
    )
    assert (
        mod._classify_sentence(
            "We are withdrawing our full-year outlook until market conditions stabilize."
        )
        == "withdrawn"
    )
