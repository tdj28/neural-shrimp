"""Fast tests of tools/c12_coupled_agents.py (no model is loaded; the prompt-length test needs only the tokenizer)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c12_coupled_agents as c3  # noqa: E402


def test_problems_deterministic_disjoint():
    a, b = c3.make_problems(), c3.make_problems()
    assert a == b and c3.problems_hash(a) == c3.problems_hash(b)
    assert sum(p["split"] == "dev" for p in a) == 16 and sum(p["split"] == "test" for p in a) == 64
    assert sum(p["split"] == "smoke" for p in a) == 2
    keys = [frozenset((p["a"], p["b"])) for p in a]
    assert len(set(keys)) == len(keys)
    for p in a:
        assert 11 <= p["a"] <= 99 and 11 <= p["b"] <= 99 and p["a"] % 10 and p["b"] % 10 and p["a"] != p["b"]
        assert p["gold"] == str(p["a"] * p["b"])
    assert c3.problems_hash(c3.make_problems(seed=1)) != c3.problems_hash(a)


def test_parse_answer():
    assert c3.parse_answer(" 1,702.") == "1702"
    assert c3.parse_answer(" $1702$.\n") == "1702"
    assert c3.parse_answer(" not known") is None


def test_coupling_schedules():
    assert not any(c3.coupled("independent", t) for t in range(48))
    assert [t for t in range(48) if c3.coupled("live", t)] == list(range(1, 48))
    assert [t for t in range(20) if c3.coupled("sparse", t)] == [4, 8, 12, 16]
    for arm in ("replay", "unrelated", "selfreplay", "live50", "replay50"):
        assert [t for t in range(48) if c3.coupled(arm, t)] == list(range(1, 48))
    assert sum(c3.coupled("sparse", t) for t in range(48)) == 11


def test_pairs_and_sources():
    starts = [c3.pair_starts(k) for k in range(8)]
    assert all(a != b for a, b in starts) and len(set(starts)) == 8
    assert c3.replay_source("replay", 7, 3, 64) == (3, 0) == c3.replay_source("replay50", 7, 3, 64)
    assert c3.replay_source("unrelated", 2, 63, 64) == (0, 2)
    assert c3.replay_source("selfreplay", 5, 3, 64) == (3, 5)
    with pytest.raises(ValueError):
        c3.replay_source("live", 0, 0, 1)
    l1 = c3.phase_rows(1)
    assert len(l1) == len(c3.phase_rows(2)) == 64
    assert set(c3.LIVE_OF) == set(c3.PHASE2) and set(c3.LIVE_OF.values()) <= set(c3.PHASE1)
    assert {r[0] for r in l1} == set(c3.PHASE1)


def test_received_swaps_roles():
    torch = pytest.importorskip("torch")
    lm = torch.arange(16, dtype=torch.float32)[:, None, None].expand(16, 3, 2)   # row r = pair r//2, agent r%2
    got = c3.received(lm, [1, 0])
    assert got[:, 0, 0].tolist() == [3.0, 2.0, 1.0, 0.0]          # pair 0 hears pair 1 (A gets B's, B gets A's)


def test_coupling_of():
    assert c3.coupling_of("live") == c3.coupling_of("replay") == c3.coupling_of("sparse") == 0.25
    assert c3.coupling_of("live50") == c3.coupling_of("replay50") == 0.5


def test_mix():
    e, m = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.allclose(c3.mix(e, m, 0.25), [0.75, 0.25])
    assert np.allclose(c3.mix(e, m, 0.0), e)


def test_problem_metrics():
    gold = "10"
    ans = {(k, ag): "10" for k in range(8) for ag in (0, 1)}
    m = c3.problem_metrics(ans, gold)
    assert m["acc"] == 1 and m["swa"] == 0 and m["agree"] == 1 and m["both_correct"] == 1
    ans[(0, 0)] = ans[(0, 1)] = "11"           # one same-wrong pair
    ans[(1, 0)], ans[(1, 1)] = "12", "13"      # wrong, different
    ans[(2, 0)], ans[(2, 1)] = None, None       # unparsed is never an agreement
    m = c3.problem_metrics(ans, gold)
    assert m["swa"] == pytest.approx(1 / 8) and m["agree"] == pytest.approx(6 / 8)
    assert m["acc"] == pytest.approx(10 / 16) and m["unparsed"] == pytest.approx(2 / 16)
    assert m["detector_swa"] == m["detector_expected"] == pytest.approx(4 / 16)
    # chance pairs exclude same-start agents: pair i agent A has start i, pair j agent B start j+1
    n_cross = sum(1 for i in range(8) for j in range(8) if i != j and i != (j + 1) % 8)
    assert n_cross == 48


def test_classify_and_choose():
    hi = {"mean": 0.15, "ci": [0.05, 0.25]}
    assert "reciprocity" in c3.classify(hi, {"mean": 0.1, "ci": [0.01, 0.2]})
    assert "message content" in c3.classify(hi, {"mean": 0.0, "ci": [-0.1, 0.1]})
    assert "below" in c3.classify({"mean": 0.05, "ci": [0.01, 0.1]}, {"mean": 0.05, "ci": [0.01, 0.1]})
    assert "no resolved" in c3.classify({"mean": 0.05, "ci": [-0.01, 0.1]}, {"mean": 0.0, "ci": [-0.1, 0.1]})
    r = [{"model": "qwen0.5b", "acc": 0.1, "unparsed": 0.0}, {"model": "qwen1.5b", "acc": 0.6, "unparsed": 0.0}]
    assert c3.choose_model(r) == {"model": "qwen1.5b", "qualified": True}
    r = [{"model": "qwen0.5b", "acc": 0.45, "unparsed": 0.0}, {"model": "qwen1.5b", "acc": 0.55, "unparsed": 0.0}]
    assert c3.choose_model(r)["model"] == "qwen0.5b"
    r = [{"model": "qwen0.5b", "acc": 0.05, "unparsed": 0.0}, {"model": "qwen1.5b", "acc": 0.9, "unparsed": 0.0}]
    assert c3.choose_model(r) == {"model": "qwen1.5b", "qualified": False}


def test_bootstrap():
    lo, hi = c3.bootstrap_mean_ci(np.ones(10))
    assert lo == hi == 1.0
    assert c3.bootstrap_mean_ci([]) == [None, None]


def test_prompt_lengths_equal():
    tr = pytest.importorskip("transformers")
    pi = pytest.importorskip("pi_harness")
    try:
        tok = tr.AutoTokenizer.from_pretrained(pi.snapshot(pi.MODELS["qwen0.5b"]))
    except SystemExit:
        pytest.skip("no local tokenizer")
    assert len({len(pi.prompt_ids(tok, p["question"])) for p in c3.make_problems()}) == 1
