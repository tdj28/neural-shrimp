import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_crossing_finish import sensitivity


def test_orbit_returns_use_state_norm_even_below_one():
    import numpy as np
    from baby_shrimp_crossing_finish import primitive
    P = np.zeros((18, 896));P[:, 0] = .25 + np.arange(18) * .01
    distances = np.linalg.norm(P[1:] - P[0], axis=1) / np.linalg.norm(P[0])
    record = {"all_returns": distances.tolist() + [1e-14], "closed": True, "no_lower_period": True}
    assert primitive(P, record)


def test_nonzero_saddle_requires_convergence_and_scale_separation():
    rows = [{"stencil_step": h, "tau": t, "newton_step": [0, 0], "hessian": [[-1, 0], [0, 1]]}
            for h, t in ((.1, .0010001), (.03, .001))]
    end = {"tau": .001, "phase_trace_spread": 1e-10}
    assert sensitivity(end, rows)["numerically_resolved_nonzero_saddle"]
    rows[-1]["newton_step"] = [.01, 0]
    with pytest.raises(ValueError, match="unconverged"):
        sensitivity(end, rows)


def test_small_value_or_extremum_cannot_pass_as_nonzero_saddle():
    rows = [{"stencil_step": h, "tau": t, "newton_step": [0, 0], "hessian": [[-1, 0], [0, 1]]}
            for h, t in ((.1, .001), (.03, 1e-8))]
    end = {"tau": 1e-8, "phase_trace_spread": 1e-10}
    assert not sensitivity(end, rows)["numerically_resolved_nonzero_saddle"]
    for r in rows:
        r.update(tau=.001, hessian=[[1, 0], [0, 1]])
    end["tau"] = .001
    assert not sensitivity(end, rows)["numerically_resolved_nonzero_saddle"]
