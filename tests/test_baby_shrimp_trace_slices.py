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
