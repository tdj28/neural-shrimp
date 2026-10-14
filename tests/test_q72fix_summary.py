import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import q72fix_summary as qs

HS = [1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]


def cell(tang, lyap, period=0, rough_below=1e-12):
    ratio = [1.0 if h >= rough_below else 3.0 for h in HS]
    return {"log10_temperature": 0.3, "alpha": 0.0, "eps": [1e-4, 1e-5, 1e-6, 1e-7], "lyapunov": lyap,
            "lyapunov_tangent": tang, "period": period, "entropy_mean": 5.0, "diameter_rel": 1.0,
            "fd_probe": {"h": HS, "central_ratio": ratio, "repeat_diff": 0.0}}


def test_verdicts():
    assert qs.verdict(cell(0.2, [0.2, 0.2, 0.2, 0.2]))[0] == "chaotic"
    assert qs.verdict(cell(-0.5, [-0.5] * 4))[0] == "not chaotic"
    assert qs.verdict(cell(0.2, [-0.1, -0.1, 0.0, 0.3]))[0] == "unresolved infinitesimal growth"
    assert qs.verdict(cell(-0.2, [0.1] * 4))[0] == "finite-size growth only"
    assert qs.verdict(cell(0.2, [0.2] * 4, period=3))[0] == "not chaotic"


def test_unresolved_scales_are_not_counted():
    # rough below 1e-6: the 1e-7 twin (positive) is ignored, so the cell is not chaotic
    v, counted = qs.verdict(cell(-0.3, [-0.3, -0.3, -0.2, 0.9], rough_below=1e-6))
    assert v == "not chaotic" and len(counted) == 3
