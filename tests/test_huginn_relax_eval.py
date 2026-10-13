import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_relax_eval as hr  # noqa: E402


def test_gold_and_extract():
    assert hr.gold("blah\n#### 1,234") == "1234"
    assert hr.extract("so 3 + 4 = 7. The answer is 7.") == "7"
    assert hr.extract("We get 12 apples and 30 pears, total 42") == "42"
    assert hr.extract("no number here") is None
    assert hr.same_number("42.0", "42") and not hr.same_number("41", "42") and not hr.same_number(None, "1")


def test_relaxed_wrapper_mixes_the_step():
    import torch

    class M:
        def core_block_forward(self, x, *a, **k):
            return 2 * x, k.get("idx", 0)

    m = M()
    r = hr.Relaxed(m)
    x = torch.ones(3)
    r.lam = 0.25
    y, _ = m.core_block_forward(x)
    assert torch.allclose(y, 0.75 * x + 0.25 * 2 * x)
    r.lam = 1.0
    y, _ = m.core_block_forward(x)
    assert torch.allclose(y, 2 * x)


class LinearCore:
    """A linear 'core block' x -> A x + c with Huginn's call signature (x, embeds, freqs, mask, cache, idx, step)."""

    def __init__(self, a, c):
        self.a, self.c = a, c

    def core_block_forward(self, x, embeds, freqs, mask, cache, idx, step):
        return x @ self.a.T + self.c, idx


def iterate(model, x, steps):
    for k in range(steps):
        x, _ = model.core_block_forward(x, None, None, None, None, 0, k)
    return x


def test_anderson_beats_picard_on_a_slow_linear_map():
    import torch

    torch.manual_seed(0)
    n = 12
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    a = q @ torch.diag(torch.linspace(-0.6, 0.95, n, dtype=torch.float64)) @ q.T
    c = torch.randn(n, dtype=torch.float64)
    star = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - a, c)
    x0 = torch.zeros(2, 3, n, dtype=torch.float64)
    m = LinearCore(a, c)
    r = hr.Relaxed(m)
    picard = (iterate(m, x0, 20) - star).norm()
    r.aa, r.lam = 4, 1.0
    anderson = (iterate(m, x0, 20) - star).norm()
    assert anderson < 0.1 * picard
    r.aa = n                                   # full memory on a linear map: GMRES, exact after n + 1 steps
    assert (iterate(m, x0, n + 2) - star).norm() < 1e-5 * star.norm()   # up to the ridge of the least squares
    r.aa = 4
    # memory resets at step 0: a second solve gives the same result
    again = (iterate(m, x0, 20) - star).norm()
    assert torch.isclose(again, anderson)


def test_set_arm_and_choice_scores():
    import torch

    class M:
        def core_block_forward(self, x, *a, **k):
            return x, 0

    r = hr.Relaxed(M())
    assert hr.set_arm(r, "0.55", 0, 3) == 0.55 and r.aa == 0
    assert hr.set_arm(r, "aa:3:0.9", 0, 3) == 0.9 and r.aa == 3
    lam1 = hr.set_arm(r, "rand:0.5:1.3", 0, 3)
    assert 0.5 <= lam1 <= 1.3 and hr.set_arm(r, "rand:0.5:1.3", 0, 3) == lam1 and r.aa == 0
    lp = torch.log(torch.tensor([[[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]]))
    targets = torch.tensor([[-100, 1, 0]])
    assert torch.allclose(hr.choice_scores(lp, targets), torch.log(torch.tensor([0.75 * 0.9])))
