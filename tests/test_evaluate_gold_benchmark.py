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


def test_classify_sentence_marks_gerund_raised_outlook_as_raised() -> None:
    mod = _load_eval_module()
    sentence = (
        "Given the strength of our business, we are raising our full-year outlook "
        "for revenue growth and free cash flow."
    )
    assert mod._classify_sentence(sentence) == "raised"


def test_classify_sentence_marks_gerund_raised_guidance_as_raised() -> None:
    mod = _load_eval_module()
    assert mod._classify_sentence("we are raising our 2025 guidance.") == "raised"


def test_classify_sentence_marks_contracted_gerund_raised_guidance_as_raised() -> None:
    mod = _load_eval_module()
    sentence = (
        "we're raising our fiscal year 2025 guidance to project revenue between "
        "$1.69 billion and $1.73 billion."
    )
    assert mod._classify_sentence(sentence) == "raised"


def test_classify_sentence_marks_increase_vs_prior_public_comments_as_raised() -> None:
    mod = _load_eval_module()
    sentence = (
        "Our guidance contemplates IFP growth of 32%, a 200bps increase as compared "
        "to our prior public comments."
    )
    assert mod._classify_sentence(sentence) == "raised"


def test_classify_sentence_marks_increasing_outlook_from_prior_guidance_as_raised() -> None:
    mod = _load_eval_module()
    sentence = (
        "We are increasing our annualized ARR outlook for our core AI infrastructure "
        "business to more than $120 million by the end of 2025 from our previous guidance of $105 million."
    )
    assert mod._classify_sentence(sentence) == "raised"


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
