import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import repro_reset as rr  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"


def test_mcnemar_exact():
    assert rr.mcnemar_exact(0, 0) == 1.0
    assert abs(rr.mcnemar_exact(5, 0) - 2 / 32) < 1e-12           # all five discordant pairs one way
    assert rr.mcnemar_exact(3, 3) == 1.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")
def test_batch1_arms_reproduce_the_reference():
    """In batch 1, the free and reset arms (prompt state copied, and with reset every step) must reproduce the
    reference exactly; the reference must repeat bit for bit."""
    import explore_llm_soft_loop as esl
    import repro_task as rt
    loop = esl.SoftLoop("qwen0.5b", 12, 8, "open", ela.OPENER, 1, "float32", "cpu")
    tok = loop.sm.tok
    prob = {"a": 23, "b": 45, "carry": False, "answer": 68}
    ids = rt.chat_ids(tok, prob)
    dis = [rt.chat_ids(tok, {"a": 41, "b": 37})]
    cue = tok(rt.CUE, add_special_tokens=False)["input_ids"]
    rec = rr.run_problem(loop, tok, ids, dis, 1.0, 6, cue, 4, prob)
    assert rec["batch1_checks_pass"]
    assert max(rec["reset_logit_distance"]) < 1e-3                   # one step in the batch layout: rounding only


def test_answers_compare_parsed_integers_not_text():
    """The answer is the first integer: the same integer in different text is no change, INVALID against a number
    is one, and two INVALIDs agree (as in tools/repro_summary.py)."""
    ref = {"reference": {"text": "340<|im_end|>\n", "parsed": 340}}
    same = dict(ref, free={"text": "340<|im_end|>\n<|endoftext|>", "parsed": 340})
    assert not rr.answer_differs(same, "free") and rr.text_differs(same, "free")
    other = dict(ref, free={"text": "34 0", "parsed": 34})
    assert rr.answer_differs(other, "free")
    none = dict(ref, free={"text": " Paris", "parsed": None})
    assert rr.answer_differs(none, "free")
    both_none = {"reference": {"text": " Paris", "parsed": None}, "free": {"text": " France", "parsed": None}}
    assert not rr.answer_differs(both_none, "free")
