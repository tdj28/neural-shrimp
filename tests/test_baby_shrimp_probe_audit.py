from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from baby_shrimp_probe_audit import precision_gate


def row(errors,passed=True):
    return {"derivative_precision":{"central_difference_relative_steps":[1e-4,1e-5,1e-6],
        "central_difference_relative_errors":errors,"dense_vs_composed_jvp_relative_error":1e-10,
        "spectral_radius_relative_difference":1e-10,"refined_leading_relative_residual":1e-10,
        "passed":passed,"quadratic_regime_found":True}}


def test_numerical_precision_gate_recomputed():
    assert precision_gate(row([1e-3,1e-5,1e-7]))


def test_claimed_pass_cannot_hide_excess_error():
    with pytest.raises(ValueError,match="derivative gate"):
        precision_gate(row([1.,1e-2,1e-4]))


def test_claimed_quadratic_order_cannot_hide_rounding_floor():
    with pytest.raises(ValueError,match="derivative gate"):
        precision_gate(row([1e-7,1e-7,1e-7]))


def test_preserved_failed_gate_is_not_upgraded():
    assert not precision_gate(row([1.,1e-2,1e-4],passed=False))
