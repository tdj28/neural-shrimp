"""Synthetic saved-data branch controls; never load a model or target artifact."""
from copy import deepcopy

import numpy as np
import pytest

from codex.crosscheck.branch import branch_extrapolation


@pytest.fixture
def synthetic_branch():
    alpha_c = .063
    rows = []
    for index, delta in enumerate(np.linspace(5.1e-5, 3.99e-4, 18)):
        y = 123.1 - 90*np.sqrt(delta) + 600*delta
        mu = 1 - 2*np.sqrt(delta*y)
        rows.append({"index": 20+index, "lambda": alpha_c-delta,
                     "residual": 1e-14, "multipliers_converged": True,
                     "multipliers": [[mu, 0], [.01, 0]], "spectral_radius": mu})
    return rows, alpha_c


def test_known_quadratic_and_input_preservation(synthetic_branch):
    rows, alpha_c = synthetic_branch
    original = deepcopy(rows)
    result = branch_extrapolation(rows, alpha_c, reference_ab=122.93)
    assert result["passed"]
    coefficients = result["quadratic"]["coefficients_ascending"]
    assert abs(coefficients[0]-123.1) < 1e-10
    assert abs(coefficients[1]+90) < 1e-8
    assert abs(coefficients[2]-600) < 1e-6
    assert result["selected_rows"][0]["saved_index"] == 20
    assert not result["independent_forward_reevaluation"]
    assert rows == original


@pytest.mark.parametrize("failure", ["convergence", "closure", "complex", "negative", "spectral_metadata"])
def test_unqualified_selected_row_is_retained_and_blocks_fit(synthetic_branch, failure):
    rows, alpha_c = synthetic_branch
    if failure == "convergence":
        rows[3]["multipliers_converged"] = False
    elif failure == "closure":
        rows[3]["residual"] = 1e-10
    elif failure == "complex":
        rows[3]["multipliers"][0][1] = 1e-4
    elif failure == "negative":
        rows[3]["multipliers"][0][0] *= -1
    else:
        rows[3]["spectral_radius"] += .01
    result = branch_extrapolation(rows, alpha_c)
    assert not result["passed"]
    assert "quadratic" not in result
    assert len(result["selected_rows"]) == 18
    assert not result["selected_rows"][3]["passed"]


def test_expected_count_mismatch_blocks_fit(synthetic_branch):
    rows, alpha_c = synthetic_branch
    result = branch_extrapolation(rows[:-1], alpha_c)
    assert not result["passed"]
    assert not result["prerequisite_gates"]["selected_count"]


def test_scaled_parameter_coordinate_is_not_inferred(synthetic_branch):
    rows, alpha_c = synthetic_branch
    with pytest.raises(ValueError, match="physical"):
        branch_extrapolation(rows, alpha_c, alpha_key="theta")
