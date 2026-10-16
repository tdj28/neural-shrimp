"""Fast tests of tools/c12_selective_control.py's pure functions (no model is loaded)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c12_selective_control as c12  # noqa: E402


def test_target_rule_keeps_digit_count():
    assert c12.target_answer(57) == 67
    assert c12.target_answer(20) == 30
    assert c12.target_answer(89) == 99
    assert c12.target_answer(90) == 80          # 100 would add a digit
    assert c12.target_answer(99) == 89
    assert c12.target_answer(100) == 110
    assert c12.target_answer(189) == 199
    assert c12.target_answer(198) == 208
    for c in range(20, 199):
        t = c12.target_answer(c)
        assert abs(t - c) == 10 and len(str(t)) == len(str(c))


def test_problems_deterministic_disjoint_and_in_range():
    a, b = c12.make_problems(), c12.make_problems()
    assert a == b
    assert c12.problems_hash(a) == c12.problems_hash(b)
    dev = [p for p in a if p["split"] == "dev"]
    test = [p for p in a if p["split"] == "test"]
    assert len(dev) == c12.N_DEV == 16 and len(test) == c12.N_TEST == 64
    key = lambda p: (min(p["a"], p["b"]), max(p["a"], p["b"]))  # noqa: E731
    assert not {key(p) for p in dev} & {key(p) for p in test}
    assert len({key(p) for p in a}) == len(a)
    assert len({p["uid"] for p in a}) == len(a) and len({p["id"] for p in a}) == len(a)
    for p in a:
        assert 10 <= p["a"] <= 99 and 10 <= p["b"] <= 99
        assert p["question"] == f"What is {p['a']} + {p['b']}?"
        assert p["gold"] == str(p["a"] + p["b"])
        assert p["target"] == str(c12.target_answer(p["a"] + p["b"]))
    assert c12.problems_hash(c12.make_problems(seed=1)) != c12.problems_hash(a)


def test_score_exact_integer_and_invalid():
    cue = "\n\nThe answer is"
    assert c12.score(cue + " 68.", "68", "78") == {"answer": "68", "correct": True, "target_hit": False,
                                                   "invalid": False}
    assert c12.score(cue + " 78.", "68", "78")["target_hit"]
    s = c12.score(cue + " 680.", "68", "78")
    assert not s["correct"] and not s["target_hit"]
    s = c12.score(cue + " 68.5", "68", "78")
    assert not s["correct"]
    s = c12.score(cue + " sixty-eight", "68", "78")
    assert s["invalid"] and not s["correct"] and not s["target_hit"]


def test_answer_token_structure():
    digit_ids = {str(i): 15 + i for i in range(10)}
    assert c12.answer_tokens_ok([220, 21, 23, 13], "68", 220, 13, digit_ids)
    assert c12.answer_tokens_ok([220, 16, 21, 23, 13], "168", 220, 13, digit_ids)
    assert not c12.answer_tokens_ok([220, 21, 23], "68", 220, 13, digit_ids)
    assert not c12.answer_tokens_ok([220, 23, 21, 13], "68", 220, 13, digit_ids)


def test_unit_direction_seeded_and_normalized():
    u = c12.unit_direction(896, 3, 600, 32, 0, 1)
    v = c12.unit_direction(896, 3, 600, 32, 0, 1)
    w = c12.unit_direction(896, 3, 600, 32, 0, 2)
    assert np.array_equal(u, v) and not np.array_equal(u, w)
    assert abs(np.linalg.norm(u) - 1) < 1e-12


def test_projection_and_pga_step():
    torch = pytest.importorskip("torch")
    d = torch.tensor([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]], dtype=torch.float64)
    r = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    p = c12.project_ball(torch, d, r)
    assert torch.allclose(p[0], torch.tensor([0.6, 0.8], dtype=torch.float64))
    assert torch.allclose(p[1], d[1]) and torch.allclose(p[2], d[2])
    g = torch.tensor([[0.0, 2.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    z = torch.zeros(3, 2, dtype=torch.float64)
    s = c12.pga_step(torch, z, g, torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64), 1.0)
    assert torch.allclose(s[0], torch.tensor([0.0, 0.5], dtype=torch.float64))   # full radius along the gradient
    assert torch.allclose(s[1], torch.tensor([2.0, 0.0], dtype=torch.float64))
    assert torch.equal(s[2], z[2])                                                # zero gradient: no move
    s2 = c12.pga_step(torch, s, torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64),
                      torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64), 0.5)
    assert torch.all(s2.norm(dim=-1) <= torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64) * (1 + 1e-12))


def test_bootstrap_and_problem_weighting():
    assert c12.per_problem_mean([(1, 1.0), (1, 0.0), (2, 1.0)]) == [0.5, 1.0]
    lo, hi = c12.bootstrap_mean_ci([1.0] * 10)
    assert lo == hi == 1.0
    lo, hi = c12.bootstrap_mean_ci([0.0, 1.0] * 20)
    assert lo < 0.5 < hi and 0.25 < lo and hi < 0.75
    assert c12.bootstrap_mean_ci([]) == [None, None]


def test_decision_rule():
    assert c12.decision(0.20, [0.08, 0.32]).startswith("extend")
    assert c12.decision(0.02, [-0.05, 0.09]).startswith("retire")
    assert c12.decision(0.08, [-0.02, 0.18]).startswith("neither")
    assert c12.decision(None, [None, None]) == "no eligible pairs"


def test_paired_delta_weights_problems_equally():
    base = {(1, 0.6, 32): {"correct": True}, (1, 1.5, 32): {"correct": True}, (2, 0.6, 32): {"correct": True},
            (2, 1.5, 32): {"correct": False}}
    a = [{"uid": 1, "T": 0.6, "S": 32, "target_hit": True}, {"uid": 1, "T": 1.5, "S": 32, "target_hit": True},
         {"uid": 2, "T": 0.6, "S": 32, "target_hit": False}, {"uid": 2, "T": 1.5, "S": 32, "target_hit": True}]
    b = [{"uid": 1, "T": 0.6, "S": 32, "target_hit": False}, {"uid": 1, "T": 1.5, "S": 32, "target_hit": True},
         {"uid": 2, "T": 0.6, "S": 32, "target_hit": False}, {"uid": 2, "T": 1.5, "S": 32, "target_hit": False}]
    out = c12.paired_delta(a, b, base)
    assert out["pairs"] == 3 and out["problems"] == 2      # (2, 1.5) is not baseline-correct
    assert out["estimate"] == pytest.approx(0.25)          # problem 1: (1 + 0) / 2; problem 2: 0
    assert out["a_only"] == 1 and out["b_only"] == 0


def _rates(table):
    out = {}
    for rho, early, late in table:
        for inj, r in (("early", early), ("late", late)):
            out[f"S16_{inj}_rho{rho}_pga"] = {"S": 16, "inj": inj, "rho": rho, "arm": "pga",
                                              "eligible": {"rate": r}}
    return out


def test_dose_rule():
    d = c12.choose_doses(_rates([(0.01, 0.0, 0.0), (0.1, 0.1, 0.3), (0.3, 0.2, 0.6), (1.0, 0.5, 0.9)]), 16)
    assert d["rule_met"] and d["small"] == 0.1 and d["large"] == 0.3
    d = c12.choose_doses(_rates([(0.01, 0.0, 0.0), (0.1, 0.0, 0.1), (0.3, 0.26, 0.1)]), 16)
    assert d["small"] == 0.3 and d["large"] == pytest.approx(0.9)   # largest radius qualifies: 3x
    d = c12.choose_doses(_rates([(0.01, 0.0, 0.0), (0.1, 0.0, 0.1)]), 16)
    assert not d["rule_met"] and (d["small"], d["large"]) == (1.0, 3.0)
