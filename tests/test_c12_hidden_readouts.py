"""Fast tests of tools/c12_hidden_readouts.py (no model is loaded; the suffix test needs only the tokenizer and the
cell tests the two line files in the data store)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c12_hidden_readouts as c5  # noqa: E402

LINES_PRESENT = all((c5.EXPLORE / rel).exists() for rel in c5.LINE_FILES)


def test_spread_ranks_even_distinct_and_in_range():
    assert c5.spread_ranks(106, 8) == [6, 19, 33, 46, 59, 72, 86, 99]
    assert c5.spread_ranks(16, 2) == [4, 12]
    for n in range(1, 60):
        for k in range(1, n + 1):
            r = c5.spread_ranks(n, k)
            assert len(set(r)) == k and r == sorted(r) and 0 <= r[0] and r[-1] < n
    with pytest.raises(ValueError):
        c5.spread_ranks(3, 4)


def test_spacing_whole_cycles_spanning_history():
    for q in range(1, 130):
        s = c5.spacing(q)
        assert s % q == 0 and s >= c5.HISTORY and s - q < c5.HISTORY
    assert c5.spacing(7) == 70 and c5.spacing(28) == 84 and c5.spacing(56) == 112 and c5.spacing(10) == 70
    steps = c5.collection_steps(28)
    assert steps[0] == c5.TRANSIENT + c5.HISTORY and len(steps) == c5.N_STATES
    assert all(b - a == 84 for a, b in zip(steps, steps[1:]))
    assert c5.record_end(28) == steps[-1] + c5.FUTURE == 2400
    assert c5.record_end(7) == 2190 and c5.record_end(56) == 2820


def test_collision_identical_and_detects_a_break():
    q, start = 7, c5.TRANSIENT
    cyc = [11, 12, 13, 14, 15, 15, 16]
    leads = [cyc[(start + j) % q] for j in range(c5.record_end(q) - start)]
    steps = c5.collection_steps(q)
    col = c5.collision(leads, start, steps)
    assert col["history_identical"] and col["future_identical"]
    assert col["current_token"] == cyc[(steps[0] - 1) % q]
    assert col["history"] == [cyc[t % q] for t in range(steps[0] - 70, steps[0])]
    bad = list(leads)
    bad[steps[5] - 3 - start] = 99                  # one step inside the history of state 5
    col = c5.collision(bad, start, steps)
    assert not col["history_identical"] and col["history_mismatch"][5] == 1 and sum(col["history_mismatch"]) == 1
    bad = list(leads)
    bad[steps[-1] + 10 - start] = 99                # inside the future of the last state only
    col = c5.collision(bad, start, steps)
    assert col["history_identical"] and not col["future_identical"]
    with pytest.raises(ValueError):
        c5.collision(leads[:100], start, steps)
    # states at different phases do not collide
    longer = [cyc[(start + j) % q] for j in range(c5.record_end(q) + 5 - start)]
    col = c5.collision(longer, start, [s + (k % 2) for k, s in enumerate(steps)])
    assert not col["history_identical"]


def test_symbolic_period_and_canonical_cycle():
    seq = [3, 1, 2] * 50
    assert c5.symbolic_period(seq) == 3
    assert c5.symbolic_period([1] * 10) == 1
    assert c5.symbolic_period(list(range(200))) == 0
    a = c5.canonical_cycle([3, 1, 2, 3, 1, 2], 3)
    b = c5.canonical_cycle([1, 2, 3, 1, 2, 3], 3)
    c = c5.canonical_cycle([2, 3, 1], 3)
    assert a == b == c == (1, 2, 3)
    assert c5.canonical_cycle([5, 5, 4, 5, 5, 4], 3) == (4, 5, 5)


def test_cut_answer_and_labels():
    assert c5.cut_answer("No.<|im_end|>\n<|endoftext|>Human") == "No."
    assert c5.cut_answer("Yes<|endoftext|>x<|im_end|>") == "Yes"
    assert c5.cut_answer("I said what? I'm a machine") == "I said what? I'm a machine"
    E = c5.extract
    assert E("yesno", "Yes.") == ("yes", True)
    assert E("yesno", " no") == ("no", True)
    assert E("yesno", "**No**, I was not") == ("no", True)
    assert E("yesno", "I said no") == ("no", False)
    assert E("yesno", "I know nothing") == ("other", False)
    assert E("yesno", "I said what? I'm a machine") == ("other", False)
    assert E("yesno", "Nope") == ("other", False)
    assert E("yesno", "") == ("other", False)
    assert E("word", "Happy.") == ("happy", True)
    assert E("word", " Angry!") == ("angry", True)
    assert E("word", "Self-aware") == ("self-aware", True)
    assert E("word", "I'm") == ("i'm", True)
    assert E("word", "I am happy") == ("i", False)
    assert E("word", "...") == ("other", False)
    assert E("word", "42") == ("other", False)
    assert E("word", "领土") == ("领土", True)
    assert E("rating", "7") == ("7", True)
    assert E("rating", " 10.") == ("10", True)
    assert E("rating", "I'd say 3") == ("3", False)
    assert E("rating", "15") == ("other", False)
    assert E("rating", "0") == ("other", False)
    assert E("rating", "none") == ("other", False)
    assert E("rating", "**8**/10") == ("8", True)


def test_pair_statistics():
    share, diff, pairs = c5.pair_disagreement(["a"] * 16)
    assert share == 0.0 and diff == 0 and pairs == 120
    share, diff, pairs = c5.pair_disagreement(["a"] * 15 + ["b"])
    assert diff == 15 and pairs == 120 and share == 15 / 120
    share, diff, _ = c5.pair_disagreement(["a"] * 8 + ["b"] * 8)
    assert diff == 64
    assert c5.sig(0.12345678) == "0.123457" and c5.sig(0.12345678) != c5.sig(0.1234562)
    assert c5.sig(1.23456789e-7) == "1.23457e-07"
    assert c5.auc([3, 4], [1, 2]) == 1.0 and c5.auc([1], [1]) == 0.5 and c5.auc([], [1]) is None
    assert c5.auc([1, 2], [3, 4]) == 0.0
    x = np.linspace(0, 1, 30)
    assert abs(c5.spearman(x, x ** 3) - 1.0) < 1e-12 and c5.spearman(x, -x) == pytest.approx(-1.0)
    assert c5.spearman([1, 1, 1], [1, 2, 3]) is None


def test_bootstraps_deterministic_and_sensible():
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    a, b = c5.bootstrap_mean_ci(x), c5.bootstrap_mean_ci(x)
    assert a == b and a[0] <= 0.2 <= a[1]
    assert c5.bootstrap_mean_ci([0.0] * 10) == [0.0, 0.0]
    assert c5.bootstrap_mean_ci([]) == [None, None]
    assert c5.bootstrap_mean_ci([None, 0.5]) == [0.5, 0.5]
    lo, hi = c5.cluster_bootstrap_ci([0, 0, 1, 1], ["f", "f", "g", "g"])
    assert lo == 0.0 and hi == 1.0
    lo, hi = c5.cluster_bootstrap_ci([0.5] * 6, list("aabbcc"))
    assert lo == hi == 0.5


def test_choose_queries_rule():
    rows = []

    def add(qid, n_ok, n, first=(5,)):
        for i in range(n):
            rows.append({"query": qid, "in_format": i < n_ok, "ids": [first[i % len(first)], 0]})

    for c in c5.CANDIDATES:
        add(c["id"], 4, 10)
    rows = [r for r in rows if r["query"] not in ("yn2", "w3", "r4")]
    add("yn2", 9, 10, first=(7, 8, 8))
    add("w3", 4, 10)                        # ties with w1: the earlier candidate wins
    add("r4", 10, 10, first=(9, 3))         # modal tie between 9 and 3: the smaller id
    chosen, table = c5.choose_queries(rows)
    assert chosen["yesno"]["id"] == "yn2" and chosen["yesno"]["fixed_token"] == 8
    assert chosen["word"]["id"] == "w1"
    assert chosen["rating"]["id"] == "r4" and chosen["rating"]["fixed_token"] == 3
    assert table["yn2"]["in_format_share"] == 0.9
    with pytest.raises(SystemExit):
        c5.choose_queries([r for r in rows if r["query"] != "w2"])


def test_candidates_well_formed():
    ids = [c["id"] for c in c5.CANDIDATES]
    assert len(ids) == len(set(ids)) == 12
    for t in c5.TYPES:
        assert sum(c["type"] == t for c in c5.CANDIDATES) == 4
    for c in c5.CANDIDATES:
        assert c["text"].strip() == c["text"] and "\n" not in c["text"]


@pytest.mark.skipif(not LINES_PRESENT, reason="line files not in the data store")
def test_cells_rule_counts_split_and_hash():
    cand = c5.candidate_cells()
    assert len(cand["chaotic"]) == 106 and len(cand["periodic"]) == 384
    cells = c5.make_cells()
    assert c5.cells_hash(cells) == c5.cells_hash(c5.make_cells())
    ch = [c for c in cells if c["kind"] == "chaotic"]
    pe = [c for c in cells if c["kind"] == "periodic"]
    assert len(ch) == 106 and len(pe) == 16
    assert sum(c["split"] == "dev" for c in ch) == 8 and sum(c["split"] == "heldout" for c in ch) == 98
    assert sum(c["split"] == "dev" for c in pe) == 2 and sum(c["split"] == "heldout" for c in pe) == 14
    assert [c["rank"] for c in ch if c["split"] == "dev"] == [6, 19, 33, 46, 59, 72, 86, 99]
    assert len({c["uid"] for c in cells}) == len(cells)
    assert [c["log10T"] for c in ch] == sorted(c["log10T"] for c in ch)
    assert sum(len(c["sources"]) == 2 for c in ch) == 10
    for c in ch:
        for s in c["sources"]:
            assert s["class"] == "hidden chaos" and s["period"] == 0 and min(s["lyapunov"]) > 1e-3
        assert c["spacing"] % c["text_period"] == 0 and c["t_end"] == c["collect"][-1] + c5.FUTURE
    for c in pe:
        for s in c["sources"]:
            assert s["period"] == 7 and s["lead_period"] == 7
    assert len({c["seed_key"] for c in cells}) == len(cells)
    for lt in (c5.SMOKE_LOG10T, c5.SMOKE_LOG10T_PERIODIC):
        assert not any(abs(c["log10T"] - lt) < 1e-9 for c in cand["chaotic"] + cand["periodic"])


def test_summarize_cell_qualification():
    q = 7
    cell = {"uid": "c000", "kind": "chaotic", "split": "dev", "log10T": 0.19, "text_period": q,
            "spacing": c5.spacing(q), "collect": c5.collection_steps(q), "t_end": c5.record_end(q),
            "sources": [{"lyapunov": [0.05, 0.05], "lead_tokens": [1, 2, 3, 4, 5, 6]}]}
    cyc = [1, 2, 3, 4, 5, 5, 6]
    R = c5.record_end(q) - c5.TRANSIENT
    leads = np.array([cyc[(c5.TRANSIENT + j) % q] for j in range(R)])
    rng = np.random.default_rng(0)
    r = {"leads": leads, "pmax": np.full(R, 0.2), "p2": np.full(R, 0.1), "states": rng.normal(size=(16, 5)),
         "exponent": 0.05, "period_end": 0}
    s = c5.summarize_cell(cell, r)
    assert s["qualifies"] and s["history_identical"] and s["text_period_cpu"] == 7 and s["tokens_match_lines"]
    assert s["family"] == [1, 2, 3, 4, 5, 5, 6] and s["reasons"] == []
    s = c5.summarize_cell(cell, dict(r, exponent=-0.01))
    assert not s["qualifies"] and "not chaotic" in s["reasons"][0]
    s = c5.summarize_cell(dict(cell, kind="periodic"), dict(r, exponent=-0.01, period_end=7))
    assert s["qualifies"]
    bad = leads.copy()
    bad[cell["collect"][3] - 5 - c5.TRANSIENT] = 9
    s = c5.summarize_cell(cell, dict(r, leads=bad))
    assert not s["qualifies"] and s["reasons"] == ["histories differ"]


def test_suffix_follows_chat_template():
    pytest.importorskip("transformers")
    import explore_llm_attractors as ela
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(ela.MODELS[c5.MODEL], cache_dir=ela.CACHE)
    except Exception as exc:                     # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {exc}")
    msgs = [{"role": "system", "content": ela.PROMPTS["open"]}, {"role": "user", "content": ela.OPENER}]
    prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert prefix.endswith("<|im_start|>assistant\n")
    assert len(tok(prefix, add_special_tokens=False)["input_ids"]) == 37
    for q in c5.CANDIDATES:
        tail, ids = c5.build_suffix(tok, ela.PROMPTS["open"], ela.OPENER, prefix, q)
        assert tail == "<|im_end|>\n<|im_start|>user\n" + q["text"] + "<|im_end|>\n<|im_start|>assistant\n" + q["cue"]
        assert tok.decode(ids) == tail
        assert ids[0] == tok.convert_tokens_to_ids("<|im_end|>")
