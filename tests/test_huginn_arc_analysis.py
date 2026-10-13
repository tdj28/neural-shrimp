import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_arc_analysis as ha  # noqa: E402


def test_lam_critical_and_star():
    mus = np.array([0.9136, -0.775 + 0.062j, -0.775 - 0.062j, 0.246 + 0.834j, 0.246 - 0.834j])
    lc, mu = ha.lam_critical(mus)
    assert abs(ha.rho(mus, lc) - 1.0) < 1e-9 and mu.real < 0          # the negative pair sets lam_c
    assert 1.12 < lc < 1.13
    ls, rs = ha.lam_star(mus)
    assert 1.0 < ls < lc and rs < ha.rho(mus, 1.0)


def test_mcnemar_and_equivalence():
    base = np.array([True] * 60 + [False] * 40)
    other = base.copy()
    other[:15] = False                                              # 15 losses, no gains
    p, b, c = ha.mcnemar_less(base, other)
    assert (b, c) == (15, 0) and p < 1e-4
    ok, mean, lo, hi = ha.equivalent(base, base.copy(), margin=0.015)
    assert ok and mean == 0.0


def test_breakpoint_fit_recovers_a_cliff():
    rng = np.random.default_rng(1)
    lams = np.array([1.0, 1.05, 1.10, 1.13, 1.16, 1.20, 1.25, 1.30])
    p = np.where(lams <= 1.13, 0.40, 0.40 - 1.5 * (lams - 1.13))
    correct = rng.random((len(lams), 4000)) < p[:, None]
    lb, c, s, _ = ha.breakpoint_fit(lams, correct)
    assert abs(lb - 1.13) < 0.03 and abs(c - 0.40) < 0.02 and s > 0.5


def test_per_item_effect_detects_own_threshold_losses(tmp_path):
    rng = np.random.default_rng(2)
    lam_c = {q: float(rng.uniform(1.06, 1.16)) for q in range(80)}
    gold = {"1.0": {q: (1.0, -10.0 + rng.normal(0, 0.2)) for q in lam_c}}
    for lam in (1.05, 1.10, 1.15, 1.20):
        gold[str(lam)] = {q: (lam, gold["1.0"][q][1] - (3.0 if lam > lam_c[q] else 0.0) + rng.normal(0, 0.5))
                          for q in lam_c}
    beta, p, ci, counts, n = ha.per_item_effect(gold, lam_c, n_boot=500)
    assert beta < -2.0 and p < 0.01 and ci[1] < 0 and n == 80
    # no effect when losses ignore the question's own threshold
    for lam in (1.05, 1.10, 1.15, 1.20):
        gold[str(lam)] = {q: (lam, gold["1.0"][q][1] - 1.0 + rng.normal(0, 0.5)) for q in lam_c}
    beta, p, ci, counts, n = ha.per_item_effect(gold, lam_c, n_boot=500)
    assert ci[0] < 0 < ci[1]


def test_stability_interval_and_two_way_effect():
    mus = np.array([-1.072, 0.5 + 0.3j, 0.5 - 0.3j])            # native loop beyond a flip
    lo, up = ha.stability_interval(mus)
    assert lo < 0.05 and abs(up - 2 / 2.072) < 2e-4 and up < 1
    rng = np.random.default_rng(3)
    arms = [0.55, 0.75, 1.0, 1.1, 1.2]
    lam_up = {q: float(rng.uniform(0.9, 1.25)) for q in range(120)}
    base = {q: rng.normal(-10, 2) for q in lam_up}                # question difficulty (fixed effect)
    outcome = {str(a): {q: base[q] - 0.3 * a - (2.0 if a > lam_up[q] else 0.0) + rng.normal(0, 0.3) for q in lam_up}
               for a in arms}
    unstable = {str(a): {q: a > lam_up[q] for q in lam_up} for a in arms}
    beta, p, ci, cells, n = ha.two_way_effect(outcome, unstable, n_boot=400)
    assert abs(beta + 2.0) < 0.3 and p < 0.01 and n == 120


def fake_item_row(q, lam, resid, rng, gold=1, unstable=False, split="test", seed=0):
    steps = {str(k): {"loglik": [0.0, float(rng.normal() if unstable else -1.0)]} for k in range(24, 33)}
    trace = [[1.0, 1.0] for _ in range(31)] + [[0.5, float(resid)]]
    return {"index": q, "lam": lam, "split": split, "seed": seed * 100003 + q, "gold": gold,
            "native_residual_rows": trace, "by_steps": steps, "gold_ids_sha256": f"h{q}"}


def write_items_fixture(root, qs, lams, rng, constant=False):
    import json

    spec_dir = root / "explore" / "spec"
    for q in qs:
        d = spec_dir / f"q{q:04d}"
        d.mkdir(parents=True)
        mu = -0.7 - 0.02 * q
        d.joinpath("fixed-spectrum.json").write_text(json.dumps({
            "fixed_point_residual_F": 1e-6, "token_ids_sha256": f"h{q}", "arnoldi_converged": {"LM": True},
            "spectra_of_F": {"LM": [[mu, 0.0], [0.5, 0.1], [0.5, -0.1]], "SR": [[mu, 0.0]], "LR": [[0.5, 0.1]]}}))
    rows_dir = root / "explore" / "rows"
    rows_dir.mkdir(parents=True)
    import huginn_arc_analysis as ha
    with open(rows_dir / "rows.jsonl", "w") as f:
        for q in qs:
            mus = np.array([complex(-0.7 - 0.02 * q, 0), 0.5 + 0.1j, 0.5 - 0.1j])
            for lam in lams:
                r = ha.rho(mus, lam)
                resid = 0.01 if constant else r ** 32 * (1 + 0.01 * rng.normal())
                f.write(json.dumps(fake_item_row(q, lam, resid, rng, unstable=r >= 1)) + "\n")


def items_args(**kw):
    import argparse

    base = dict(spectra="spec", rows="rows", start=0, count=20, lams=[1.0, 1.1], fixed_tol=1e-4, boot=50,
                split="test", seed=0, out="items-out")
    base.update(kw)
    return argparse.Namespace(**base)


def test_items_primary_and_admission(tmp_path, monkeypatch):
    import json

    import explore_llm_attractors as ela
    import huginn_arc_analysis as ha

    monkeypatch.setattr(ela, "DATA", tmp_path)
    write_items_fixture(tmp_path, range(20), [1.0, 1.1], np.random.default_rng(0))
    ha.items_cmd(items_args())
    res = json.loads((tmp_path / "explore" / "items-out" / "items.json").read_text())
    assert res["cohort"] == 20 and res["primary"]["success"] and res["primary"]["statistic"] > 0.9
    # a missing cell, a duplicate row and a token-id mismatch are each refused or excluded
    rows = (tmp_path / "explore" / "rows" / "rows.jsonl").read_text().splitlines()
    (tmp_path / "explore" / "rows" / "rows.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(SystemExit, match="missing"):
        ha.items_cmd(items_args(out="o2"))
    (tmp_path / "explore" / "rows" / "rows.jsonl").write_text("\n".join(rows + rows[:1]) + "\n")
    with pytest.raises(SystemExit, match="duplicate"):
        ha.items_cmd(items_args(out="o3"))
    (tmp_path / "explore" / "rows" / "rows.jsonl").write_text("\n".join(rows) + "\n")
    extra = tmp_path / "explore" / "spec" / "q0099"
    extra.mkdir()
    with pytest.raises(SystemExit, match="outside the frozen panel"):
        ha.items_cmd(items_args(out="o4"))
    extra.rmdir()
    spec = tmp_path / "explore" / "spec" / "q0003" / "fixed-spectrum.json"
    d = json.loads(spec.read_text())
    d["token_ids_sha256"] = "other"
    spec.write_text(json.dumps(d))
    ha.items_cmd(items_args(out="o5"))
    res = json.loads((tmp_path / "explore" / "o5" / "items.json").read_text())
    assert res["cohort"] == 19 and "token ids differ" in res["dispositions"]["3"]["status"]


def test_items_token_hash_checked_in_every_arm(tmp_path, monkeypatch):
    import json

    import explore_llm_attractors as ela
    import huginn_arc_analysis as ha

    monkeypatch.setattr(ela, "DATA", tmp_path)
    write_items_fixture(tmp_path, range(20), [1.0, 1.1], np.random.default_rng(0))
    path = tmp_path / "explore" / "rows" / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:                                     # change only the non-first arm's hash of question 5
        if r["index"] == 5 and r["lam"] == 1.1:
            r["gold_ids_sha256"] = "changed"
        if r["index"] == 7 and r["lam"] == 1.1:
            del r["gold_ids_sha256"]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ha.items_cmd(items_args())
    res = json.loads((tmp_path / "explore" / "items-out" / "items.json").read_text())
    assert res["cohort"] == 18
    assert "token ids differ" in res["dispositions"]["5"]["status"]
    assert "token ids differ" in res["dispositions"]["7"]["status"]


def test_items_constant_arm_fails_the_primary(tmp_path, monkeypatch):
    import json

    import explore_llm_attractors as ela
    import huginn_arc_analysis as ha

    monkeypatch.setattr(ela, "DATA", tmp_path)
    write_items_fixture(tmp_path, range(20), [1.0, 1.1], np.random.default_rng(0), constant=True)
    ha.items_cmd(items_args())
    res = json.loads((tmp_path / "explore" / "items-out" / "items.json").read_text())
    assert not res["primary"]["success"] and not np.isfinite(res["primary"]["statistic"])


def test_subset_spectral_radius_can_miss_the_shifted_mode():
    """Codex's counterexample: LM/SR/LR subsets of J can omit the dominant mode of (1 - lam) I + lam J."""
    import huginn_arc_analysis as ha

    full = np.array([0.991 + 0.001 * i for i in range(8)] + [-0.890 - 0.001 * i for i in range(8)]
                    + [complex(-0.88, 0.45), complex(-0.88, -0.45)])
    lm = full[np.argsort(-np.abs(full))][:8]
    sr = full[np.argsort(full.real)][:8]
    lr = full[np.argsort(-full.real)][:8]
    subset = np.unique(np.concatenate([lm, sr, lr]))
    assert ha.rho(subset, 1.05) < 1.0 < ha.rho(full, 1.05)


def test_generation_cmd_primary_logic(tmp_path, monkeypatch):
    import argparse
    import json

    import explore_llm_attractors as ela
    import huginn_arc_analysis as ha

    monkeypatch.setattr(ela, "DATA", tmp_path)
    rng = np.random.default_rng(1)
    folder = tmp_path / "explore" / "g"
    folder.mkdir(parents=True)
    with open(folder / "rows.jsonl", "w") as f:
        for i in range(200):
            base = rng.uniform() < 0.4
            for arm, ok in (("1.0", base), ("1.05", base), ("1.15", base and rng.uniform() < 0.7),
                            ("1.20", base and rng.uniform() < 0.3)):
                f.write(json.dumps({"arm": arm, "lam": float(arm), "index": i, "correct": bool(ok),
                                    "correct_strict": bool(ok), "new_tokens": 80, "strict_pred": "1", "split": "test",
                                    "steps": 32, "prompt_style": "cot8", "seed": i}) + "\n")
    ns = argparse.Namespace(rows=["g"], start=0, count=200, arms=["1.0", "1.05", "1.15", "1.20"], base="1.0",
                            collapse_arm="1.20", break_arm="1.15", safe_arm="1.05", margin=0.04, out="gout",
                            split="test", steps=32, prompt_style="cot8", seed=0, max_new=256)
    ha.generation_cmd(ns)
    res = json.loads((tmp_path / "explore" / "gout" / "generation.json").read_text())
    assert res["B1_decline"]["pass"] and res["B3_noninferior"]["pass"] and res["primary_success"]
    with pytest.raises(SystemExit):
        ha.generation_cmd(argparse.Namespace(**{**vars(ns), "count": 201, "out": "gout2"}))
    with pytest.raises(SystemExit, match="frozen"):                 # a train or r = 8 file with the same indices
        ha.generation_cmd(argparse.Namespace(**{**vars(ns), "steps": 8, "out": "gout3"}))
