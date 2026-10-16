"""Unit tests of tools/c12_compression_fidelity.py (no model): quantizers, period detection and classification on
synthetic sequences, twin exponents, one-step metrics, the noise calibration, interval statistics, prompt layout."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
import c12_compression_fidelity as cf  # noqa: E402


def bf16_matrix(rng, shape, scale=0.05):
    """A float64 matrix of bfloat16 values (like the stored decoder weights)."""
    return torch.as_tensor(rng.normal(size=shape) * scale).to(torch.bfloat16).to(torch.float64)


# ------------------------------------------------------------------------------------------------- quantizers

def test_roundf_rounds_half_away_from_zero():
    below_half = np.nextafter(np.float32(0.5), np.float32(0))
    x = torch.tensor([0.5, -0.5, 1.5, 2.5, -2.5, below_half, -below_half, 0.0, 126.5, -126.4999], dtype=torch.float32)
    assert cf.roundf(x).tolist() == [1.0, -1.0, 2.0, 3.0, -3.0, 0.0, 0.0, 0.0, 127.0, -126.0]


@pytest.mark.parametrize("variant", ["q8", "q4"])
def test_representable_blocks_reconstruct_exactly(variant):
    """Blocks k * s with integers |k| <= Q, max |k| = Q and s a power of two (exact in fp16) come back exactly."""
    Q = cf.QMAX[variant]
    rng = np.random.default_rng(1)
    k = rng.integers(-Q, Q + 1, size=(3, 4, 32)).astype(float)
    k[:, :, 0] = rng.choice([-Q, Q], size=(3, 4))
    s = 2.0 ** rng.integers(-14, -4, size=(3, 4, 1))
    W = torch.as_tensor((k * s).reshape(3, 128))
    Wq, stats = cf.quantize_rows(W, Q)
    assert torch.equal(Wq, W)
    assert Wq.dtype == W.dtype and stats["blocks"] == 12


@pytest.mark.parametrize("variant", ["q8", "q4"])
def test_error_bound_and_levels(variant):
    """Every w' is an integer multiple k s of its block's fp16 scale s with |k| <= Q, and |w - w'| <= d/2 + Q |d - s|:
    the levels are chosen with the float32 scale d = a / Q (rounding error at most d/2) and reconstructed with
    s = fp16(d), as in ggml."""
    Q = cf.QMAX[variant]
    rng = np.random.default_rng(2)
    W = bf16_matrix(rng, (16, 256))
    Wq, _ = cf.quantize_rows(W, Q)
    wb, qb = W.reshape(16, 8, 32).numpy(), Wq.reshape(16, 8, 32).numpy()
    amax = np.abs(wb).max(-1, keepdims=True)
    d = (amax.astype(np.float32) / np.float32(Q)).astype(np.float64)
    s = d.astype(np.float16).astype(np.float64)
    k = qb / s
    assert np.array_equal(k, np.round(k)) and np.abs(k).max() <= Q
    bound = 0.5 * d * (1 + 1e-6) + Q * np.abs(d - s)
    assert np.all(np.abs(qb - wb) <= bound)
    assert np.abs(qb - wb).max() > 0.4 * d.min()                   # the bound is not vacuous
    rel = float(np.linalg.norm(qb - wb) / np.linalg.norm(wb))
    assert 0 < rel < (0.02 if variant == "q8" else 0.25)


def test_blocks_run_along_the_input_dimension():
    """A large weight in one block of a row does not coarsen the other blocks of that row or other rows."""
    W = torch.full((2, 64), 1e-3, dtype=torch.float64)
    W[0, 0] = 1.0
    W[0, 40] = 3e-3
    Wq, _ = cf.quantize_rows(W.to(torch.float32).to(torch.float64), 7)
    assert Wq[0, 1].item() == 0.0                                    # same block as the 1.0: rounds to zero
    assert Wq[0, 40].item() == pytest.approx(3e-3, rel=2e-3)         # its own block keeps it
    assert Wq[0, 41].item() == pytest.approx(1e-3, rel=0.2)
    assert torch.allclose(Wq[1], W[1].to(torch.float32).to(torch.float64), rtol=2e-3, atol=0)


def test_symmetry_zero_blocks_and_refusals():
    rng = np.random.default_rng(3)
    W = bf16_matrix(rng, (4, 64))
    for Q in (127, 7):
        assert torch.equal(cf.quantize_rows(-W, Q)[0], -cf.quantize_rows(W, Q)[0])
    Z = torch.zeros((2, 32), dtype=torch.float64)
    assert torch.equal(cf.quantize_rows(Z, 7)[0], Z)
    with pytest.raises(ValueError):
        cf.quantize_rows(torch.zeros((2, 48), dtype=torch.float64), 7)
    with pytest.raises(ValueError):
        cf.quantize_rows(torch.full((1, 32), 0.1, dtype=torch.float64), 7)     # 0.1 is not a float32 value


def ggml_q8_0_reference(w):
    """ggml's quantize_row_q8_0_ref and dequantize_row_q8_0, in numpy float32/float16 arithmetic."""
    x = w.astype(np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    d = amax / np.float32(127)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = (x * inv).astype(np.float32)
    t = np.trunc(v)
    q = t + np.where(np.abs(v - t) >= 0.5, np.sign(v), 0)            # roundf
    return (q * d.astype(np.float16).astype(np.float32)).astype(np.float32).reshape(w.shape)


def test_q8_equals_ggml_q8_0_reference():
    rng = np.random.default_rng(4)
    W = bf16_matrix(rng, (64, 1024), scale=0.03)
    Wq, _ = cf.quantize_rows(W, 127)
    ref = ggml_q8_0_reference(W.numpy())
    assert np.array_equal(Wq.numpy(), ref.astype(np.float64))


# ------------------------------------------------------------------------------------- period detection

def run_tracker(seq, steps=cf.STEPS, confirm=cf.CONFIRM):
    """Push a synthetic sequence (n_steps + 1, rows, d) through the Tracker and return its main labels."""
    X = torch.as_tensor(seq)
    tr = cf.Tracker(X[0], steps, confirm, cf.TOL)
    for t in range(1, steps + confirm + 1):
        tr.push(X[t])
    return tr, tr.labels()


def sequence(f, n=cf.STEPS + cf.CONFIRM, d=4):
    return np.stack([f(t) for t in range(n + 1)])[:, None, :] * np.ones((1, 1, d))


def test_converging_period_three_is_verified():
    rng = np.random.default_rng(5)
    c, v = rng.normal(size=(3, 4)), rng.normal(size=4)
    seq = np.stack([c[t % 3] + 0.5 ** t * v for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    tr, (cls, sub, period) = run_tracker(seq)
    assert (cls[0], sub[0], period[0]) == (0, cf.SUB_NONE, 3)
    assert tr.residual[0].item() < cf.TOL


def test_period_four_is_primitive_not_eight():
    rng = np.random.default_rng(6)
    c = rng.normal(size=(4, 4))
    seq = np.stack([c[t % 4] for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    _, (cls, _, period) = run_tracker(seq)
    assert (cls[0], period[0]) == (0, 4)


def test_slow_fixed_point_is_pending_then_unresolved():
    rng = np.random.default_rng(7)
    c, v = rng.normal(size=4), rng.normal(size=4)
    seq = np.stack([c + 0.97 ** t * v for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    _, (cls, sub, period) = run_tracker(seq)
    assert (cls[0], sub[0], period[0]) == (2, cf.SUB_PENDING, 0)
    c2, s2, p2 = cf.twin_labels(np.array([[-0.03], [-0.03]]), np.array([0]))
    assert (c2[0], s2[0], p2[0]) == (2, cf.SUB_IRREGULAR, 0)


def test_late_convergence_is_not_verified():
    """Closes at lag 1 only after step 256 (|x_t - x_{t-1}| = 0.05 * 0.95^(t-1) |v|): unresolved, late_convergence."""
    v = np.ones(4) / 2.0
    seq = np.stack([np.zeros(4) + 0.95 ** t * v for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    _, (cls, sub, period) = run_tracker(seq)
    assert (cls[0], sub[0], period[0]) == (2, cf.SUB_LATE_CONV, 1)


def test_long_period_is_unresolved_with_its_period():
    rng = np.random.default_rng(8)
    c = rng.normal(size=(12, 4))
    seq = np.stack([c[t % 12] for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    _, (cls, sub, period) = run_tracker(seq)
    assert (cls[0], sub[0], period[0]) == (2, cf.SUB_LONG, 12)


def test_broken_confirmation_is_not_verified():
    """Period 2 at step 256, a jump to another period-2 orbit at step 280: the confirmation fails; lag 2 closes again
    at step 320, so the start is unresolved (late_convergence)."""
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
    seq = np.stack([(a if t < 280 else b)[t % 2] for t in range(cf.STEPS + cf.CONFIRM + 1)])[:, None, :]
    tr, (cls, sub, period) = run_tracker(seq)
    assert int(tr.q_detect[0]) == 2 and not bool(tr.ok[0])
    assert (cls[0], sub[0], period[0]) == (2, cf.SUB_LATE_CONV, 2)


def test_tolerance_is_relative_above_norm_one_and_absolute_below():
    ring = torch.zeros((3, 4, 2), dtype=torch.float64)
    ring[1] = torch.tensor([[1000.0, 0.0], [1000.0, 0.0], [0.5, 0.0], [0.5, 0.0]])
    ring[2] = ring[1] + torch.tensor([[5e-6, 0.0], [2e-5, 0.0], [5e-9, 0.0], [2e-8, 0.0]])
    q = cf.min_period(ring, 2, cf.TOL, 1)
    assert q.tolist() == [1, 0, 1, 0]


def test_rows_are_independent_in_one_batch():
    rng = np.random.default_rng(10)
    c = rng.normal(size=(3, 4))
    n = cf.STEPS + cf.CONFIRM + 1
    seq = np.stack([np.stack([c[t % 3], c[0] + 0.97 ** t, np.full(4, float(t % 12))]) for t in range(n)])
    _, (cls, sub, period) = run_tracker(seq)
    assert cls.tolist() == [0, 2, 2] and period.tolist() == [3, 0, 12]
    assert sub.tolist() == [cf.SUB_NONE, cf.SUB_PENDING, cf.SUB_LONG]


def test_twin_labels():
    lam = np.array([[0.5, 0.5, 0.5, 2e-3, 5e-4], [0.4, -0.1, 0.4, 2e-3, 0.3]])
    q_end = np.array([0, 0, 4, 0, 0])
    cls, sub, period = cf.twin_labels(lam, q_end)
    assert cls.tolist() == [1, 2, 2, 1, 2]
    assert sub.tolist() == [cf.SUB_NONE, cf.SUB_IRREGULAR, cf.SUB_LATE_PERIOD, cf.SUB_NONE, cf.SUB_IRREGULAR]
    assert period.tolist() == [0, 0, 4, 0, 0]


def test_main_labels_direct():
    cls, sub, period = cf.main_labels(np.array([2, 2, 0, 0, 3]), np.array([True, False, False, False, True]),
                                      np.array([2, 2, 9, 0, 1]))
    assert cls.tolist() == [0, 2, 2, 2, 2]
    assert sub.tolist() == [cf.SUB_NONE, cf.SUB_LATE_CONV, cf.SUB_LONG, cf.SUB_PENDING, cf.SUB_LATE_CONV]
    assert period.tolist() == [2, 2, 9, 0, 1]


# ---------------------------------------------------------------------------------------------- twin exponents

def test_twins_on_a_linear_map_give_the_largest_log_stretch():
    A = torch.diag(torch.tensor([2.0, 0.5, 0.25], dtype=torch.float64))
    base = torch.tensor([[1.0, 1.0, 1.0], [0.3, -2.0, 1.0]], dtype=torch.float64)
    rng = np.random.default_rng(11)
    dirs = [torch.as_tensor(rng.normal(size=(2, 3))) for _ in cf.EPS]
    tw = cf.Twins(base, cf.EPS, dirs)
    for _ in range(400):
        tw.advance(lambda X: X @ A.T)
    lam = tw.exponents()
    assert np.allclose(lam, math.log(2.0), atol=0.01)
    assert tw.collapses.sum() == 0


def test_twins_on_the_logistic_map_give_log_two():
    base = torch.tensor([[0.3141], [0.2718]], dtype=torch.float64)
    dirs = [torch.ones((2, 1), dtype=torch.float64) for _ in cf.EPS]
    tw = cf.Twins(base, cf.EPS, dirs)
    for _ in range(4000):
        tw.advance(lambda X: 4.0 * X * (1.0 - X))
    assert np.allclose(tw.exponents(), math.log(2.0), atol=0.05)


def test_twin_collapse_is_counted_and_restarted():
    base = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    dirs = [torch.tensor([[1.0, 0.0]], dtype=torch.float64) for _ in cf.EPS]
    tw = cf.Twins(base, cf.EPS, dirs)
    for _ in range(3):
        tw.advance(lambda X: torch.ones_like(X))
    assert tw.collapses.tolist() == [[3], [3]]
    assert torch.isfinite(tw.X).all() and np.all(tw.exponents() < -100)


def test_twin_directions_do_not_depend_on_the_batch():
    a = cf.twin_directions([(16, 2, 1), (17, 0, 0)], 5)
    b = cf.twin_directions([(17, 0, 0)], 5)
    assert np.array_equal(a[0][1], b[0][0]) and np.array_equal(a[1][1], b[1][0])
    assert not np.array_equal(a[0][0], a[1][0])


def test_canonical_cycle():
    assert cf.canonical_cycle([5, 3, 9]) == cf.canonical_cycle([3, 9, 5]) == (3, 9, 5)
    assert cf.canonical_cycle([7]) == (7,) and cf.canonical_cycle([]) == ()


# ---------------------------------------------------------------------------------------------- one-step metrics

def test_one_step_metrics_against_numpy():
    rng = np.random.default_rng(12)
    V, d, n = 40, 6, 5
    z1, z2 = rng.normal(size=(n, V)) * 3, rng.normal(size=(n, V)) * 3
    E = rng.normal(size=(V, d))
    T = np.array([0.5, 1.0, 2.0, 0.3, 3.0])
    m = cf.one_step_metrics(torch.as_tensor(z1), torch.as_tensor(z2), torch.as_tensor(T), torch.as_tensor(E))

    def sm(z):
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    p, q = sm(z1 / T[:, None]), sm(z2 / T[:, None])
    assert np.allclose(m["kl_T"], (p * np.log(p / q)).sum(-1), rtol=1e-12)
    p1, q1 = sm(z1), sm(z2)
    assert np.allclose(m["kl_1"], (p1 * np.log(p1 / q1)).sum(-1), rtol=1e-12)
    assert np.allclose(m["tv_T"], 0.5 * np.abs(p - q).sum(-1))
    assert np.allclose(m["map_err"], np.linalg.norm(q @ E - p @ E, axis=1) / np.linalg.norm(p @ E, axis=1))
    assert m["top1"].tolist() == (z1.argmax(1) == z2.argmax(1)).tolist()
    same = cf.one_step_metrics(torch.as_tensor(z1), torch.as_tensor(z1), torch.as_tensor(T), torch.as_tensor(E))
    assert np.all(same["kl_T"] == 0) and np.all(same["map_err"] == 0) and same["top1"].all()


def test_noise_calibration_hits_the_target():
    rng = np.random.default_rng(13)
    z = torch.as_tensor(rng.normal(size=(20, 300)) * 4)
    b = torch.as_tensor(rng.standard_normal(size=(20, 300)))
    T = torch.full((20,), 0.7, dtype=torch.float64)
    f = lambda s: cf.mean_noise_kl(z, b, T, s)  # noqa: E731
    vals = [f(s) for s in (1e-3, 1e-2, 1e-1, 1.0)]
    assert all(a < b_ for a, b_ in zip(vals, vals[1:]))
    for target in (1e-6, 1e-3, 0.2):
        s = cf.calibrate_sigma(f, target)
        assert f(s) == pytest.approx(target, rel=1e-3)
    with pytest.raises(ValueError):
        cf.calibrate_sigma(f, 0.0)


def test_noise_vector_is_seeded_per_prompt():
    assert np.array_equal(cf.noise_vector(16, 10), cf.noise_vector(16, 10))
    assert not np.array_equal(cf.noise_vector(16, 10), cf.noise_vector(17, 10))


# --------------------------------------------------------------------------------------------------- statistics

def test_paired_interval_known_values():
    r = cf.paired_interval([0.1, 0.2, 0.3])
    assert r["mean"] == pytest.approx(0.2) and r["sd"] == pytest.approx(0.1)
    half = 4.302652729911275 * 0.1 / math.sqrt(3)
    assert r["ci"] == pytest.approx([0.2 - half, 0.2 + half])
    assert math.isnan(cf.paired_interval([0.5])["sd"])


def test_bootstrap_intervals_cover_the_mean():
    rng = np.random.default_rng(14)
    d = rng.normal(0.05, 0.1, size=48)
    lo, hi = cf.bootstrap_interval(d)
    assert lo < d.mean() < hi
    lo2, hi2 = cf.bootstrap_interval(d, clusters=np.arange(48) // 8)
    assert lo2 < d.mean() < hi2
    assert cf.bootstrap_interval(np.zeros(10)) == [0.0, 0.0]


# ----------------------------------------------------------------------------------------- prompts and layout

def test_prompt_splits_and_balanced_order():
    dev, held = cf.prompt_order("dev"), cf.prompt_order("heldout")
    assert len(dev) == 16 and len(held) == 48 and not set(dev) & set(held)
    assert sorted(dev + held) == list(range(64))
    assert {p // 8 for p in dev} == set(cf.DEV_SYSTEMS)
    batches = cf.batches_of("heldout")
    assert len(batches) == 8 and all(len(b) == 6 for b in batches)
    for b in batches:                                              # every batch: one opener x the 6 system prompts
        assert sorted(p // 8 for p in b) == list(cf.HELDOUT_SYSTEMS) and len({p % 8 for p in b}) == 1
    dev_batches = cf.batches_of("dev")                             # as run on 16 Oct (batches of 4)
    assert dev_batches == [[0, 8, 1, 9], [2, 10, 3, 11], [4, 12, 5, 13], [6, 14, 7, 15]]
    assert cf.prompt_messages(8 * 3 + 5) == [{"role": "system", "content": cf.SYSTEM_PROMPTS[3]},
                                             {"role": "user", "content": cf.OPENERS[5]}]


def test_prompt_lists_are_pinned():
    assert len(cf.SYSTEM_PROMPTS) == len(cf.OPENERS) == len(cf.START_TOKENS) == 8
    assert cf.prompt_lists_hash() == "359500d48a064cadfdc92ecd35e790481ba0028967c5e8b10a5446c6539848f1"


def test_batch_rows_layout():
    g, j, s = cf.batch_rows([16, 24], 3, 2)
    assert len(g) == 12
    for r in range(12):
        assert r == (g[r] * 3 + j[r]) * 2 + s[r]


def test_temperature_grid():
    assert len(cf.LOG_T) == 8 and cf.LOG_T[0] == -0.5 and cf.LOG_T[-1] == 0.5


def test_gate_rows_accepts_consistent_and_rejects_inconsistent_rows():
    cfg = {"log_t": [0.0], "starts": 2, "tol": cf.TOL}
    res = {"cls": np.array([0, 1], dtype=np.int8), "sub": np.array([0, 0], dtype=np.int8),
           "period": np.array([3, 0], dtype=np.int16), "q_detect": np.array([3, 0]), "confirmed": np.array([True, False]),
           "q_end": np.array([3, 0]), "residual": np.array([1e-12, np.nan]),
           "lam": np.array([[np.nan, 0.2], [np.nan, 0.3]]), "q_twin": np.array([0, 0]),
           "collapses": np.zeros((2, 2), dtype=int), "x_end": np.zeros((2, 3))}
    fails, info = cf.gate_rows(res, cfg, 1)
    assert fails == [] and info["classes"] == {"periodic": 1, "chaotic": 1, "unresolved": 0}
    bad = dict(res, residual=np.array([1e-6, np.nan]))
    assert "periodic row with closure residual >= tol" in cf.gate_rows(bad, cfg, 1)[0]
    bad2 = dict(res, lam=np.array([[np.nan, 0.2], [np.nan, -0.3]]))
    assert "chaotic row violating the chaos rule" in cf.gate_rows(bad2, cfg, 1)[0]


def test_merge_rows_concatenates_along_the_row_axes():
    def res(n, base):
        return {"batch": 0, "batch_prompts": np.array([16]), "kept_steps": np.array([0, 1]), "seconds": np.zeros(2),
                "cls": np.full(n, base), "lead": np.full((3, n), base), "kept": np.full((2, n, 4), base),
                "lam": np.full((2, n), base), "collapses": np.full((2, n), base), "x_end": np.full((n, 4), base),
                "cycle": np.full((n, 8), base)}
    m = cf.merge_rows(res(2, 0), res(3, 1))
    assert m["cls"].tolist() == [0, 0, 1, 1, 1]
    assert m["lead"].shape == (3, 5) and m["kept"].shape == (2, 5, 4) and m["lam"].shape == (2, 5)
    assert m["collapses"].shape == (2, 5) and m["x_end"].shape == (5, 4) and m["cycle"].shape == (5, 8)
    assert m["kept_steps"].tolist() == [0, 1] and m["batch_prompts"].tolist() == [16]
    empty = dict(res(2, 0), kept=np.zeros(0))
    assert cf.merge_rows(empty, dict(res(3, 1), kept=np.zeros(0)))["kept"].size == 0
