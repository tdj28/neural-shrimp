"""Unit tests of tools/e5_repetition_onset.py (no model): prompt rule, repetition rule, onsets and censoring,
statistics, the inverse-CDF sampler, the loop driver on synthetic maps (periodic, chaotic, irregular, lost
periodicity, retirement) and the analysis on synthetic data."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
import c12_compression_fidelity as cf  # noqa: E402
import e5_repetition_onset as e5  # noqa: E402


# ------------------------------------------------------------------------------------------------ prompts and grid

def test_prompt_rule_and_splits():
    dev, held = e5.e5_prompt_ids("dev"), e5.e5_prompt_ids("heldout")
    assert dev == [0, 2, 4, 6, 9, 11, 13, 15]
    assert len(held) == 24 and not set(dev) & set(held)
    for p in dev + held:
        s, o = divmod(p, 8)
        assert (s + o) % 2 == 0
    assert {p // 8 for p in dev} == {0, 1} and {p // 8 for p in held} == {2, 3, 4, 5, 6, 7}
    assert sorted(p % 8 for p in dev) == list(range(8))                       # every opener once
    assert all(sum(p % 8 == o for p in held) == 3 for o in range(8))           # every opener three times
    assert all(sum(p // 8 == s for p in held) == 4 for s in range(2, 8))


def test_prompt_lists_are_c12_lists_and_pinned():
    lists = e5.prompt_lists()
    assert lists["system_prompts"] == list(cf.SYSTEM_PROMPTS) and lists["openers"] == list(cf.OPENERS)
    assert e5.prompt_lists_hash() == "e184fcf23151d3c6bdb31d2b6e79c50c7fe95bcfdfbd1b064d3a0c0e80a6f363"
    rec = e5.prompt_record(57)
    assert (rec["system_index"], rec["opener_index"]) == (7, 1) and rec["opener"] == cf.OPENERS[1]


def test_loop_batches():
    held = e5.e5_prompt_ids("heldout")
    b = e5.loop_batches(held)
    assert len(b) == 3 and all(len(x) == 8 for x in b)
    assert [sorted({p // 8 for p in x}) for x in b] == [[2, 3], [4, 5], [6, 7]]
    assert e5.loop_batches(e5.e5_prompt_ids("dev")) == [e5.e5_prompt_ids("dev")]


def test_temperature_grid():
    assert len(e5.LOG_T) == 24 and e5.LOG_T[0] == -1.0 and abs(e5.LOG_T[-1] - 0.5) < 1e-15
    assert abs(e5.grid_step(e5.LOG_T) - 1.5 / 23) < 1e-15


# ------------------------------------------------------------------------------------------------ repetition rule

def periodic_seq(q, n=256, seed=0, head=None):
    rng = np.random.default_rng(seed)
    unit = rng.integers(1000, 2000, size=q)
    s = np.array([unit[i % q] for i in range(n)])
    if head is not None:                                  # a non-periodic head of `head` tokens
        s[:head] = rng.integers(5000, 9000, size=head)
    return s


@pytest.mark.parametrize("q", [1, 2, 7, 32])
def test_periodic_tails_up_to_32_repeat(q):
    assert e5.smallest_period(periodic_seq(q)) == q


def test_period_above_the_bound_does_not_repeat_but_is_seen_by_the_diagnostic():
    s = periodic_seq(33)
    assert e5.smallest_period(s) == 0
    assert e5.smallest_period(s, qmax=48) == 33


def test_smallest_period_wins():
    unit = [1, 2, 3, 1, 2, 4]
    s = np.array(unit * 50)
    assert e5.smallest_period(s) == 6
    assert e5.smallest_period(np.array([5, 6] * 128)) == 2


def test_window_is_exactly_the_last_96_tokens():
    s = periodic_seq(5, head=256 - 96)                    # periodic in exactly the last 96 tokens
    assert e5.smallest_period(s) == 5
    s2 = periodic_seq(5, head=256 - 95)                   # one token short
    assert e5.smallest_period(s2) == 0
    s3 = periodic_seq(5)
    s3[-50] += 1                                          # one mismatch inside the window
    assert e5.smallest_period(s3) == 0


def test_smallest_period_batches_and_refusals():
    toks = np.stack([np.stack([periodic_seq(3), periodic_seq(40, seed=1)]), np.stack([periodic_seq(1),
                                                                                       periodic_seq(9)])])
    assert e5.smallest_period(toks).tolist() == [[3, 0], [1, 9]]
    assert e5.repeat_fraction(toks).tolist() == [0.5, 1.0]
    with pytest.raises(ValueError):
        e5.smallest_period(np.zeros(50, dtype=int))
    with pytest.raises(ValueError):
        e5.smallest_period(np.zeros(256, dtype=int), qmax=96)


def test_periodic_tail_lengths():
    s = periodic_seq(4, head=100)
    assert e5.periodic_tail(s[None], np.array([4]))[0] == 156
    assert e5.periodic_tail(periodic_seq(3)[None], np.array([3]))[0] == 256
    assert e5.periodic_tail(s[None], np.array([0]))[0] == 0


# ------------------------------------------------------------------------------------------------ onsets

LT = np.array(e5.LOG_T)
STEP = e5.grid_step(LT)


def test_rep_onset_interpolates_in_log_temperature():
    frac = np.zeros(24)
    frac[:3] = [1.0, 1.0, 0.75]
    frac[3] = 0.25
    r = e5.rep_onset(LT, frac)
    assert r["censored"] is None and r["index"] == 3 and r["monotone"]
    assert abs(r["value"] - (LT[2] + 0.5 * STEP)) < 1e-12 and r["analysis_value"] == r["value"]


def test_rep_onset_exact_half_and_nonmonotone():
    frac = np.ones(24)
    frac[5], frac[6], frac[7] = 0.5, 0.25, 0.75
    frac[8:] = 0.0
    r = e5.rep_onset(LT, frac)
    assert abs(r["value"] - LT[5]) < 1e-12 and r["index"] == 6 and r["monotone"] is False


def test_rep_onset_censoring():
    lo = e5.rep_onset(LT, np.full(24, 0.25))
    assert lo["censored"] == "below" and lo["value"] is None and abs(lo["analysis_value"] - (LT[0] - STEP)) < 1e-12
    hi = e5.rep_onset(LT, np.full(24, 0.5))
    assert hi["censored"] == "above" and abs(hi["analysis_value"] - (LT[-1] + STEP)) < 1e-12


def test_chaos_onset_and_censoring():
    labels = ["periodic"] * 10 + ["irregular", "chaotic", "periodic", "chaotic"] + ["periodic"] * 10
    r = e5.chaos_onset(LT, labels)
    assert r["index"] == 11 and r["value"] == LT[11] and r["censored"] is None
    never = e5.chaos_onset(LT, ["periodic"] * 12 + ["irregular"] * 12)
    assert never["censored"] == "above" and abs(never["analysis_value"] - (LT[-1] + STEP)) < 1e-12


def test_entropy_and_entropy_temperature():
    assert abs(e5.entropy_of(np.zeros(1000), 0.7) - math.log(1000)) < 1e-12
    rng = np.random.default_rng(3)
    z = rng.normal(size=5000) * 4
    T = e5.entropy_temperature(z, 2.0)
    assert abs(e5.entropy_of(z, T) - 2.0) < 1e-9
    assert e5.entropy_of(z, 0.9 * T) < 2.0 < e5.entropy_of(z, 1.1 * T)
    with pytest.raises(ValueError):
        e5.entropy_temperature(z, math.log(5000) + 0.1)


# ------------------------------------------------------------------------------------------------ statistics

def test_spearman_matches_scipy_with_ties():
    from scipy import stats
    rng = np.random.default_rng(4)
    x = rng.integers(0, 5, size=30).astype(float)
    y = x + rng.integers(0, 3, size=30)
    assert abs(e5.spearman(x, y) - stats.spearmanr(x, y).statistic) < 1e-12
    assert math.isnan(e5.spearman(np.ones(5), np.arange(5)))


def test_partial_spearman_equals_residual_correlation_of_ranks():
    from scipy import stats
    rng = np.random.default_rng(5)
    z = rng.normal(size=40)
    x = z + 0.5 * rng.normal(size=40)
    y = z + 0.5 * rng.normal(size=40)
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    expect = np.corrcoef(res(rx), res(ry))[0, 1]
    assert abs(e5.partial_spearman(x, y, z) - expect) < 1e-10
    assert e5.spearman(x, y) > 0.5 and abs(e5.partial_spearman(x, y, z)) < e5.spearman(x, y)


def test_bootstrap_is_seeded_and_counts_nonfinite_draws():
    rng = np.random.default_rng(6)
    x = rng.normal(size=20)
    y = x + 0.1 * rng.normal(size=20)
    stat = lambda idx: e5.spearman(x[idx], y[idx])  # noqa: E731
    a, b = e5.bootstrap(stat, 20, 500), e5.bootstrap(stat, 20, 500)
    assert a == b and a["ci"][0] > 0.8 and a["nonfinite_draws"] == 0
    xc = np.array([0.0] * 19 + [1.0])
    c = e5.bootstrap(lambda idx: e5.spearman(xc[idx], y[idx]), 20, 500)
    assert c["nonfinite_draws"] > 0
    cl = e5.bootstrap(stat, 20, 300, clusters=np.arange(20) % 4)
    assert cl["clusters"] == 4 and cl["ci"][0] is not None


def test_permutation_p():
    rng = np.random.default_rng(7)
    x = rng.normal(size=24)
    assert e5.permutation_p(x, x + 0.1 * rng.normal(size=24), 400) < 0.01
    assert e5.permutation_p(x, rng.normal(size=24), 400) > 0.01


def test_stratified_difference_by_hand():
    frac = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([["chaotic", "periodic"], ["periodic", "chaotic"], ["periodic", "irregular"],
                       ["irregular", "periodic"]])
    diff, rows = e5.stratified_difference(frac, labels)
    # T0: chaotic {0.0} vs periodic {0.5, 1.0}: -0.75, weight 2/3; T1: chaotic {0.5} vs periodic {1.0, 1.0}: -0.5, 2/3
    assert abs(diff - (-0.625)) < 1e-12
    assert rows[0]["n_irregular"] == 1 and rows[1]["n_irregular"] == 1
    d2, _ = e5.stratified_difference(frac, labels, units=np.array([0, 0, 1, 2]))
    assert np.isfinite(d2)
    d3, _ = e5.stratified_difference(frac[2:], labels[2:])
    assert math.isnan(d3)


# ------------------------------------------------------------------------------------------------ sampler

def test_inverse_cdf_draws_follow_the_distribution():
    z = torch.tensor([[2.0, 1.0, 0.0, -1.0, 0.5]], dtype=torch.float32)
    T = 0.8
    p = torch.softmax(z.double()[0] / T, -1).numpy()
    N = 20000
    u = (np.arange(N) + 0.5) / N                       # stratified uniforms: counts equal N p to within one
    tok, ent = e5.draw_tokens(z.expand(N, -1), np.full(N, T), u, masked=())
    counts = np.bincount(tok.numpy(), minlength=5)
    assert np.all(np.abs(counts - N * p) <= 1.0)
    assert np.allclose(ent.numpy(), -(p * np.log(p)).sum(), atol=1e-12)


def test_masked_tokens_are_never_drawn_and_entropy_is_of_the_masked_distribution():
    z = torch.tensor([[5.0, 1.0, 0.0, 4.0]], dtype=torch.float32)
    N = 1000
    tok, ent = e5.draw_tokens(z.expand(N, -1), np.full(N, 1.0), (np.arange(N) + 0.5) / N, masked=(0, 3))
    assert set(tok.numpy().tolist()) <= {1, 2}
    q = torch.softmax(torch.tensor([1.0, 0.0], dtype=torch.float64), -1).numpy()
    assert np.allclose(ent.numpy(), -(q * np.log(q)).sum(), atol=1e-12)


def test_low_temperature_gives_the_argmax_and_u_zero_the_first_token():
    z = torch.tensor([[0.0, 3.0, 2.9]], dtype=torch.float32)
    tok, _ = e5.draw_tokens(z.expand(3, -1), np.full(3, 1e-3), np.array([0.0, 0.5, 0.999]), masked=())
    assert tok.tolist() == [1, 1, 1]
    tok0, _ = e5.draw_tokens(torch.zeros((1, 4)), np.ones(1), np.zeros(1), masked=(0,))
    assert tok0.tolist() == [1]


def test_sample_uniforms_are_fixed_per_prompt_and_temperature():
    a = e5.sample_uniforms(16, 3, 4, 10)
    assert a.shape == (12, 10)
    b = np.random.default_rng((e5.SEED_SAMPLES, 16, 2)).random((10, 4)).T
    assert np.array_equal(a[8:12], b)
    assert np.array_equal(e5.sample_uniforms(16, 3, 4, 10), a)
    assert not np.array_equal(e5.sample_uniforms(18, 3, 4, 10), a)


# ------------------------------------------------------------------------------------------------ loop driver

class FakeBackend:
    """Synthetic per-cell maps on R^3 with a three-token readout:
    0 contracting to a fixed point; 1 exact period 3 (cyclic permutation of the coordinates); 2 logistic map on
    the first coordinate (exponent log 2); 3 identity until `switch` calls, then like cell 2 (loses its period);
    4 irrational rotation of the first two coordinates (no period, exponent 0)."""

    def __init__(self, switch=10 ** 9):
        self.cells, self.calls, self.switch = None, 0, switch
        self.phi = 2 * math.pi * (math.sqrt(5) - 1) / 2

    def set_rows(self, cells):
        self.cells = np.asarray(cells)

    def map(self, c, x):
        if c == 0:
            return torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64) + 0.5 * (x - torch.tensor([1.0, 2.0, 0.5],
                                                                                              dtype=torch.float64))
        if c == 1:
            return x[[2, 0, 1]]
        if c == 2 or (c == 3 and self.calls >= self.switch):
            return torch.stack([4.0 * x[0] * (1.0 - x[0]), torch.ones((), dtype=torch.float64), 0.5 * x[2]])
        if c == 3:
            return x.clone()
        cs, sn = math.cos(self.phi), math.sin(self.phi)
        return torch.stack([cs * x[0] - sn * x[1], sn * x[0] + cs * x[1], x[2]])

    def step(self, X):
        self.calls += 1
        Y = torch.stack([self.map(int(c), X[i]) for i, c in enumerate(self.cells)])
        return Y, torch.softmax(Y, dim=-1)

    @staticmethod
    def start():
        return torch.tensor([[0.3, 0.1, 0.2], [0.3, 1.7, -0.4], [0.3141, 1.0, 0.2], [0.2718, 1.0, 0.2],
                             [1.0, 0.0, 0.0]], dtype=torch.float64)


CFG = {"transient": 150, "measured": 1000, "eps": list(e5.EPS), "tol": e5.TOL, "qmax": e5.QMAX,
       "retire_tol": e5.RETIRE_TOL, "retire_window": e5.RETIRE_WINDOW, "lambda_min": e5.LAMBDA_MIN, "retire": False}


def directions(cells, stream):
    return e5.twin_directions([(0, int(c)) for c in cells], 3, len(e5.EPS), stream)


def test_loop_driver_labels_without_retirement():
    be = FakeBackend(switch=CFG["transient"] + 50)
    res = e5.run_loop(be, be.start(), CFG, directions, retire=False)
    assert res["label"].tolist() == ["periodic", "periodic", "chaotic", "chaotic", "irregular"]
    assert res["period"].tolist() == [1, 3, 0, 0, 0]
    assert res["twins"].tolist() == [False, False, True, False, True]
    assert res["supplementary"].tolist() == [False, False, False, True, False]
    assert np.allclose(res["lyapunov"][:, 2], math.log(2.0), atol=0.1)
    assert np.allclose(res["lyapunov"][:, 3], math.log(2.0), atol=0.1)
    assert np.all(np.abs(res["lyapunov"][:, 4]) < 1e-3)
    assert not res["retired"].any()
    # virtual retirement: cells 0, 1 and 3 (identity before the switch) are recorded; only cell 3's label differs
    assert (res["retire_step"][[0, 1, 3]] > 0).all() and (res["retire_step"][[2, 4]] < 0).all()
    assert res["retire_q_tol"][[0, 1, 3]].tolist() == [1, 3, 1]
    bad = e5.virtual_retirement_check(res)
    assert [b["cell"] for b in bad] == [3]
    assert e5.gate_loop(res, 5, CFG) == []


def test_loop_driver_with_retirement_keeps_the_periodic_labels():
    be = FakeBackend()
    cfg = dict(CFG, retire=True)
    res = e5.run_loop(be, be.start(), cfg, directions, retire=True, rebuild_every=4)
    assert res["retired"].tolist() == [True, True, False, True, False]
    assert res["label"].tolist() == ["periodic", "periodic", "chaotic", "periodic", "irregular"]
    assert res["period"].tolist() == [1, 3, 0, 1, 0]
    assert e5.gate_loop(res, 5, cfg) == []
    assert e5.virtual_retirement_check(res) == []
    # the chaotic cell's exponents equal those of the same cell run without retirement (same map, same twins)
    full = e5.run_loop(FakeBackend(), FakeBackend.start(), CFG, directions, retire=False)
    assert np.array_equal(res["lyapunov"][:, 2], full["lyapunov"][:, 2])


def test_twin_subset_matches_c12_twins_alone():
    be = FakeBackend()
    res = e5.run_loop(be, be.start(), CFG, directions, retire=False)
    alone = FakeBackend()
    alone.set_rows([2])
    X = FakeBackend.start()[[2]]
    for _ in range(CFG["transient"]):
        X, _ = alone.step(X)
    dirs = [torch.as_tensor(u, dtype=torch.float64) for u in directions([2], 0)]
    tw = cf.Twins(X, e5.EPS, dirs)
    alone.set_rows([2, 2, 2])
    for _ in range(CFG["measured"]):
        tw.advance(lambda Z: alone.step(Z)[0])
    assert np.array_equal(tw.exponents()[:, 0], res["lyapunov"][:, 2])


def test_lag_closures_and_first_lag_match_min_period():
    rng = np.random.default_rng(8)
    c = rng.normal(size=(3, 4))
    ring = torch.zeros((e5.QMAX + 1, 3, 4), dtype=torch.float64)
    seq = [np.stack([c[t % 3], c[t % 2] * 2.0, rng.normal(size=4)]) for t in range(200)]
    for t, s in enumerate(seq):
        ring[t % (e5.QMAX + 1)] = torch.as_tensor(s)
    D = e5.lag_closures(ring, 199, e5.QMAX)
    q = e5.first_lag(D, e5.TOL)
    assert q.tolist() == cf.min_period(ring, 199, e5.TOL, e5.QMAX).tolist() == [3, 2, 0]


def test_gate_flags_inconsistent_rows():
    be = FakeBackend(switch=CFG["transient"] + 50)
    res = e5.run_loop(be, be.start(), CFG, directions, retire=False)
    bad = dict(res, label=res["label"].copy(), lyapunov=res["lyapunov"].copy())
    bad["label"][4] = "chaotic"
    assert any("chaotic cell" in f for f in e5.gate_loop(bad, 5, CFG))
    bad2 = dict(res, period=res["period"].copy())
    bad2["period"][2] = 5
    assert any("non-periodic cell with a period" in f for f in e5.gate_loop(bad2, 5, CFG))


# ------------------------------------------------------------------------------------------------ analysis

def synthetic_split(n=8, seed=9):
    """Prompt i repeats (period 5, all samples) below temperature index k_i and not above; its loop is chaotic from
    index c_i. Returns loops, samples, prompts, k and c."""
    rng = np.random.default_rng(seed)
    prompts = list(range(16, 16 + 2 * n, 2))
    k = rng.permutation(np.arange(4, 4 + n))
    c = k + rng.integers(0, 3, size=n)
    loops, samples = {}, {}
    for i, p in enumerate(prompts):
        toks = np.zeros((24, 16, 256), dtype=np.int32)
        for j in range(24):
            for s in range(16):
                toks[j, s] = periodic_seq(5, seed=1000 * i + j) if j < k[i] else rng.integers(0, 10 ** 6, size=256)
        labels = ["chaotic" if j >= c[i] and j < c[i] + 3 else "periodic" for j in range(24)]
        loops[p] = {"labels": labels, "cells": [{"lyapunov": [0.1, 0.1] if lab == "chaotic" else [None, None],
                                                 "period": 0 if lab == "chaotic" else 1} for lab in labels],
                    "entropy": {"prompt": p, "T_H": 10.0 ** LT[k[i]], "T_H_masked": 10.0 ** LT[k[i]],
                                "entropy_by_T": [1.0] * 24},
                    "gate_failures": [], "virtual_retirement_mismatches": [], "file": "x"}
        samples[p] = {"tokens": toks, "mean_entropy": np.ones((24, 16)), "gate_failures": [], "T_H_float32": 1.0}
    return loops, samples, prompts, k, c


def test_analysis_on_synthetic_data():
    loops, samples, prompts, k, c = synthetic_split()
    res = e5.analyze_split(loops, samples, prompts, e5.LOG_T, e5.REP_WINDOW, e5.REP_QMAX, e5.H_TARGET, n_boot=200)
    for r, kk, cc in zip(res["prompts"], k, c):
        assert abs(r["T_rep"]["value"] - (LT[kk - 1] + 0.5 * STEP)) < 1e-12
        assert r["T_chaos"]["value"] == LT[cc]
        assert r["repeat_fraction"][:kk] == [1.0] * kk and r["repeat_fraction"][kk:] == [0.0] * (24 - kk)
    from scipy import stats
    assert abs(res["primary"]["spearman"] - stats.spearmanr(c, k).statistic) < 1e-12
    assert res["censoring"]["T_chaos_never_chaotic"] == [] and res["censoring"]["T_rep_below_grid"] == []
    sm = res["secondary_matched"]
    assert sm["temperatures_with_both"] > 0 and -1.0 <= sm["difference"] <= 1.0
    # T_H placed exactly at k: T_H and T_rep are perfectly rank-correlated
    assert abs(res["secondary_partial"]["spearman_T_H_T_rep"] - 1.0) < 1e-12


# ------------------------------------------------------------------------------------------------ choices

def test_projected_loop_row_steps():
    steps = e5.projected_loop_row_steps(np.array([128, -1, 500, 1500]), np.array([False, True, False, False]), 1000,
                                        1000)
    assert steps == 128 + 2000 + 500 + 1500 + 2 * 1000


def write_synthetic_runs(root, loops, samples, prompts, retire_steps):
    import json
    ld, sd = root / "dev-loop-x", root / "dev-sample-x"
    ld.mkdir(parents=True)
    sd.mkdir(parents=True)
    (ld / "settings.json").write_text(json.dumps({"transient": 1000, "measured": 1000}))
    n = 24 * len(prompts)
    labels = np.array([lab for p in prompts for lab in loops[p]["labels"]])
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(len(prompts), 500)) * 3
    np.savez(ld / "loop-b00.npz", label=labels, twins=labels == "chaotic", supplementary=np.zeros(n, bool),
             retired=np.zeros(n, bool), retire_step=np.where(labels == "chaotic", -1, retire_steps),
             collapses=np.zeros((2, n), int), residual=np.where(labels == "chaotic", np.nan, 1e-12),
             seconds=np.array([3600.0]), batch_prompts=np.array(prompts), first_step_logits=z0)
    ents = [{"prompt": p, "T_H": loops[p]["entropy"]["T_H"], "T_H_masked": loops[p]["entropy"]["T_H"],
             "entropy_by_T": [1.0] * 24} for p in prompts]
    cells = [{"prompt": p, "t_index": j, "label": loops[p]["labels"][j], "lyapunov": [0.1, 0.1], "period": 1}
             for p in prompts for j in range(24)]
    (ld / "loop-b00.json").write_text(json.dumps({"batch": 0, "prompts": prompts, "gate_failures": [],
                                                  "virtual_retirement_mismatches": [], "first_step_entropy": ents,
                                                  "seconds": 3600.0, "cells": cells}))
    for p in prompts:
        np.savez(sd / f"samples-p{p:02d}.npz", tokens=samples[p]["tokens"], mean_entropy=samples[p]["mean_entropy"],
                 gate_failures=np.array("[]"), T_H_float32=np.array(1.0), seconds=np.array([500.0]))


def test_choices_apply_the_rubric(tmp_path, monkeypatch):
    import argparse
    import json
    loops, samples, prompts, k, c = synthetic_split(n=8)
    monkeypatch.setattr(e5, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(e5, "e5_prompt_ids", lambda split: prompts if split == "dev" else list(range(24)))
    write_synthetic_runs(tmp_path, loops, samples, prompts, retire_steps=150)
    out = tmp_path / "choices.json"
    e5.cmd_choices(argparse.Namespace(loop_runs=["dev-loop-x"], sample_runs=["dev-sample-x"], out=str(out)))
    ch = json.loads(out.read_text())
    assert ch["retire"] is True and ch["rep_window"] == 96 and ch["rep_qmax"] == 32 and ch["h_target"] == 2.0
    assert ch["repetition_rule"]["card_rule_degenerate"] is False
    hs = ch["heldout_samples"]
    # dev: 8 prompts in 3600 s; held-out loop scaled by 24 / 8 prompts and by the retirement projection
    assert hs["projected_sampling_hours_at_16"] == 500.0 * 24 / 3600
    assert ch["samples"] == (16 if hs["projected_total_hours"] <= 7.0 else 12 if hs["projected_total_hours"] <= 9.5
                             else 8)
    assert hs["projected_loop_hours"] < 3 * 1.0                     # retirement shortens the 3 x 1 h of the full runs


def test_buffered_sampler_draws_the_same_tokens():
    g = torch.Generator().manual_seed(11)
    z = (torch.randn(64, 3000, generator=g) * 4).to(torch.float32)
    rng = np.random.default_rng(12)
    temps = 10.0 ** rng.uniform(-1, 0.5, size=64)
    s = e5.InverseCDFSampler()
    for k in range(3):                                  # buffers reused across calls
        u = rng.random(64)
        t0, e0 = e5.draw_tokens(z, temps, u, masked=(7, 2999))
        t1, e1 = s(z, temps, u, masked=(7, 2999))
        assert torch.equal(t0, t1)
        assert torch.allclose(e0, e1, rtol=1e-12, atol=1e-12)
    zz = z.clone()
    zz[:, :10] = -1e4                                    # underflowing weights at low temperature
    u = rng.random(64)
    assert torch.equal(e5.draw_tokens(zz, temps * 0 + 0.1, u, masked=(7,))[0], s(zz, temps * 0 + 0.1, u, masked=(7,))[0])
