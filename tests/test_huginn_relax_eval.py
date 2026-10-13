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
