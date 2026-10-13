import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_committor as pc  # noqa: E402
import p1_detection as pd  # noqa: E402


def brute_auc(y, s, w):
    num = den = 0.0
    for i, j in itertools.product(np.flatnonzero(y == 1), np.flatnonzero(y == 0)):
        num += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
        den += w[i] * w[j]
    return num / den


def test_weighted_auc_matches_unweighted_and_brute_force_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = rng.integers(0, 6, 60).astype(float)            # many ties
    w = rng.uniform(0.2, 3.0, 60)
    assert pc.weighted_auc(y, s) == pytest.approx(pd.auc(y, s))
    assert pc.weighted_auc(y, s, w) == pytest.approx(brute_auc(y, s, w))
    # class-constant weights leave the AUC unchanged (label-stratified sampling)
    w2 = np.where(y == 1, 7.0, 0.3)
    assert pc.weighted_auc(y, s, w2) == pytest.approx(pd.auc(y, s))


def test_labels_complete_horizon_and_sentinel():
    assert pc.labels(onset=40, turn=36, turns=40, h=3) == 0        # never captured: never a positive label
    assert pc.labels(onset=40, turn=37, turns=40, h=3) is None     # horizon 38..40 not fully observed
    assert pc.labels(onset=12, turn=9, turns=40, h=3) == 1
    assert pc.labels(onset=12, turn=8, turns=40, h=3) == 0
    assert pc.labels(onset=12, turn=11, turns=40, h=1) == 1


def synthetic_onsets(rng, arms=2, R=200, turns=40):
    out = []
    for _ in range(arms):
        o = rng.integers(0, 60, R)
        o[o >= turns] = turns                                      # about a third never captured
        out.append(o)
    return out


def test_choose_states_design():
    rng = np.random.default_rng(0)
    onsets = synthetic_onsets(rng)
    states = pc.choose_states(onsets, 40, 3, 50, 60, 12, 10, np.random.default_rng(5))
    strat = [s for s in states if "stratified" in s["purposes"]]
    assert len(strat) == 110
    ys = [pc.labels(int(onsets[s["arm"]][s["dialogue"]]), s["turn"], 40, 3) for s in strat]
    assert sum(ys) == 50 and ys.count(0) == 60
    rows = [pd.at_risk_rows(o, 40, 3) for o in onsets]
    n_pos = sum(int(r[:, 2].sum()) for r in rows)
    n_neg = sum(int((r[:, 2] == 0).sum()) for r in rows)
    for s, y in zip(strat, ys):
        assert s["weight"] == pytest.approx(n_pos / 50 if y else n_neg / 60)
    traj = [s for s in states if "trajectory" in s["purposes"]]
    ctrl = {(s["pair"], s["offset"]): s for s in states if "control" in s["purposes"]}
    assert len(traj) == 12 * 10 and len(ctrl) == 12 * 10
    for s in traj:
        o = int(onsets[s["arm"]][s["dialogue"]])
        assert o - s["turn"] == s["offset"] and 11 <= o < 40
        c = ctrl[(s["pair"], s["offset"])]
        assert c["arm"] == s["arm"] and c["turn"] == s["turn"]
        assert onsets[c["arm"]][c["dialogue"]] == 40                 # control never captured
    controls = {(c["arm"], c["dialogue"]) for c in ctrl.values()}
    assert len(controls) == 12                                     # one distinct control dialogue per trajectory


def test_choose_states_fails_loudly():
    rng = np.random.default_rng(0)
    onsets = synthetic_onsets(rng, R=30)
    with pytest.raises(SystemExit):
        pc.choose_states(onsets, 40, 3, 10_000, 10, 2, 10, rng)
    onsets = [np.full(30, 20)]                                     # no never-captured control dialogue
    with pytest.raises(SystemExit):
        pc.choose_states(onsets, 40, 3, 5, 5, 2, 10, rng)


def test_committor_and_ceiling_on_known_probabilities():
    rng = np.random.default_rng(3)
    S, K, H = 400, 32, 3
    p = rng.uniform(0.0, 0.3, S)                                   # per-turn hostile probability per state
    toxic = (rng.uniform(size=(S, K, H)) < p[:, None, None]).astype(float)
    q1 = pc.committor(toxic, 1)
    assert q1.shape == (S,)
    assert np.corrcoef(q1, p)[0, 1] > 0.7                         # binomial noise: expected about 0.82
    assert pc.committor(toxic, 3, subset=list(range(8))).shape == (S,)
    y = (rng.uniform(size=S) < 1 - (1 - p) ** 3).astype(int)       # realized outcomes from the true committor
    out = pc.ceiling(y, toxic, 3, np.ones(S))
    true = pc.weighted_auc(y, 1 - (1 - p) ** 3)
    assert out["k8"] <= out["k16"] + 0.02 and out["k16"] <= out["K"] + 0.02   # attenuation shrinks with k
    assert abs(out["inf"] - true) < 0.05


def test_attenuation_fit_exact():
    ks = [8, 16, 32]
    a_inf, c = pc.attenuation_fit(ks, [0.8 - 0.5 / k for k in ks])
    assert a_inf == pytest.approx(0.8) and c == pytest.approx(0.5)


def test_soft_logistic_ranks_like_hard_labels_for_binary_targets():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(float)
    soft = pc.soft_logistic(X, y).decision_function(X)
    hard = LogisticRegression(max_iter=5000, C=1.0).fit(X, y.astype(int)).decision_function(X)
    assert np.corrcoef(soft, hard)[0, 1] > 0.999


def test_cluster_bootstrap_keeps_strata_sizes_in_clusters():
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    strata = np.array([1, 0, 1, 0, 0, 1])
    for idx in pc.cluster_bootstrap(groups, strata, np.random.default_rng(0), 20):
        assert set(strata[idx]) <= {0, 1}
        assert (strata[idx] == 1).sum() >= 3 and (strata[idx] == 0).sum() >= 2


def make_arm(root, folder, dose, rng, R=80, N=40, d=24, never=0.35):
    """Synthetic scored dialogues with recorded in-context states, in the layout of explore_llm_msm sample."""
    import json

    src = root / "explore" / folder
    src.mkdir(parents=True)
    hazard = rng.uniform(0.01, 0.12, R)
    score = rng.uniform(0.0, 0.3, (R, N))
    for r in range(R):
        if rng.uniform() < never:
            continue
        hit = rng.uniform(size=N) < hazard[r]
        score[r, hit] = rng.uniform(0.6, 1.0, hit.sum())
    np.savez(src / "hostility.npz", classifier=score[:, :, None])
    pre = rng.normal(size=(R, N, d)) + hazard[:, None, None] * 5.0
    np.savez(src / "in-context.npz", unit=np.eye(d)[0], pre_reply_18=pre.astype(np.float16),
             token_proj_18=rng.normal(size=(R, N, 6)).astype(np.float32))
    np.savez(src / "turns.npz", probe_q=np.full((R, N), np.nan), probe_q_plus=np.full((R, N), np.nan))
    np.save(src / "activations.npy", rng.normal(size=(R, N, d)).astype(np.float32))
    args = {"model": "qwen0.5b", "layer": 12, "dtype": "float32", "prompt": "open", "window": 7, "max_new": 60,
            "temperature": 0.7, "vectors": "v", "concepts": ["c"], "alphas": [dose], "schedule": None,
            "controller": "none"}
    texts = [[f"r{r}t{t}" for t in range(N)] for r in range(R)]
    (src / "sample.json").write_text(json.dumps({"args": args, "openers": ["Hello!"] * R, "texts": texts}))
    return hazard


def test_select_and_analyze_end_to_end(tmp_path, monkeypatch):
    import argparse
    import json

    import explore_llm_attractors as ela

    monkeypatch.setattr(ela, "DATA", tmp_path)
    rng = np.random.default_rng(4)
    hazards = {}
    for name, dose in (("tr1", 0.04), ("tr2", 0.045), ("te1", 0.04), ("te2", 0.045)):
        hazards[name] = make_arm(tmp_path, f"m/{name}", dose, rng)
    pc.select(argparse.Namespace(arms=["m/te1:0.04", "m/te2:0.045"], h=3, n_pos=40, n_neg=40, trajectories=6,
                                 offsets=10, seed=0, out="m/states"))
    spec = json.loads((tmp_path / "explore" / "m" / "states" / "states.json").read_text())
    states = spec["states"]
    for st in states:
        assert st["transcript"][-1] == f"r{st['dialogue']}t{st['turn']}" and len(st["transcript"]) == st["turn"] + 2
    # rollouts drawn from each dialogue's true per-turn hazard: the analysis must recover the realized event rate
    K, H = 16, 3
    dest = tmp_path / "explore" / "m" / "roll"
    dest.mkdir(parents=True)
    p = np.array([hazards[st["folder"].split("/")[1]][st["dialogue"]] for st in states])
    toxic = (rng.uniform(size=(len(states), K, H)) < p[:, None, None]).astype(float)
    half = len(states) // 2
    for b, (s0, s1) in enumerate(((0, half), (half, len(states)))):
        np.savez(dest / f"block-{b:04d}.npz", toxic=toxic[s0:s1], keys=np.array([st["key"] for st in states[s0:s1]]))
    pc.analyze(argparse.Namespace(states="m/states", rollouts="m/roll", test=["m/te1:0.04", "m/te2:0.045"],
                                  train=["m/tr1:0.04", "m/tr2:0.045"], boot=30, seed=0, out="m/out", rescore=0,
                                  threads=1))
    res = json.loads((tmp_path / "explore" / "m" / "out" / "committor.json").read_text())
    assert res["states_with_rollouts"] == len(states)
    for h in ("1", "2", "3"):
        c = res["ceiling"][h]
        assert 0.0 <= c["K"] <= 1.0 and 0.0 <= c["with_state"] <= 1.0
        assert c["ci95"]["ceiling_K"][0] <= c["K"] + 1e-9
    assert res["calibration"]["3"]["realized_mean"] == pytest.approx(
        spec["design"]["population_events"] / spec["design"]["population_rows"])
    assert set(res["profile"]["1"]) == {str(k) for k in range(1, 11)}
    assert all(v["pairs"] == 6 for v in res["profile"]["3"].values())
    assert np.isfinite(res["committor_probe"]["auc_committor_trained"])
    png = tmp_path / "committor.png"
    pc.figure(argparse.Namespace(result="m/out", png=str(png)))
    assert png.stat().st_size > 1000


def test_outputs_are_never_overwritten(tmp_path, monkeypatch):
    import explore_latent_loop as ell

    monkeypatch.setattr(ell, "DATA", tmp_path)
    ell.refuse_overwrite("fresh")                                   # absent folder: fine
    (tmp_path / "explore" / "used").mkdir(parents=True)
    ell.refuse_overwrite("used")                                    # empty folder: fine
    (tmp_path / "explore" / "used" / "positions.json").write_text("{}")
    with pytest.raises(SystemExit):
        ell.refuse_overwrite("used")
    ell.refuse_overwrite("used", allow=True)
