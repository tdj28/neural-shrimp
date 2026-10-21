import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_crossing import trace_zero_slices


def plan():
    return {"slice_basis": np.eye(2).tolist(), "slice_offsets": [-.2, 0, .2],
            "root_bracket": .6, "root_coordinate_tolerance": 1e-7,
            "root_trace_tolerance": 1e-6, "root_max_iterations": 20}


def test_actual_zero_roots_preserve_positive_gap():
    rows = trace_zero_slices(lambda u: {"tau": .04-u[0]**2+u[1]**2}, plan())
    assert len(rows) == 6
    for negative, positive in zip(rows[::2], rows[1::2]):
        assert positive["along"] - negative["along"] >= .399999
        assert max(abs(r["point"]["tau"]) for r in (negative, positive)) < 1e-6


def test_missing_bracket_and_invalid_coordinates_fail():
    with pytest.raises(ValueError, match="no trace-zero bracket"):
        trace_zero_slices(lambda u: {"tau": 1 + u[0]**2}, plan())
    p = plan();p["slice_basis"] = [[1, 1], [0, 1]]
    with pytest.raises(ValueError, match="orthonormal"):
        trace_zero_slices(lambda u: {"tau": 1}, p)


def test_concrete_plan_binds_audited_saddle():
    import json, hashlib
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "docs/baby-shrimp-zoom/crossing-curve-plan.json").read_text())
    path = root / p["saddle_receipt"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == p["saddle_receipt_sha256"]
    r = json.loads(path.read_text())["summary"]
    assert r["qualification"]["numerically_resolved_nonzero_saddle"]
    assert p["start"] == [r["endpoint"]["alpha"], r["endpoint"]["log10_temperature"]]
    basis = np.array(p["slice_basis"])
    H = np.array(r["qualification"]["converged_scales"][-1]["hessian"])
    diagonal = basis.T @ H @ basis
    assert abs(diagonal[0, 1]) < 1e-12 and diagonal[0, 0] < 0 < diagonal[1, 1]
    from runpod_soft_float64 import UPLOAD_PATHS
    assert "docs/baby-shrimp-zoom/crossing-curve-plan.json" in UPLOAD_PATHS
    assert p["saddle_receipt"] in UPLOAD_PATHS
