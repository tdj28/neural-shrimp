from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_orbits import primitive_gate


def test_closed_repeated_orbit_is_not_primitive():
    result = primitive_gate([.2, 1e-14, .2, 1e-14], 4)
    assert result["closed"] and not result["no_lower_period"]


def test_primitive_needs_closure_as_well_as_separation():
    result = primitive_gate([.2, .1, 2e-6], 3)
    assert result["no_lower_period"] and not result["closed"]


@pytest.mark.parametrize("values", [[.2, float("nan"), 1e-14], [.2, 1e-14]])
def test_missing_or_nonfinite_lag_is_refused(values):
    with pytest.raises(ValueError):
        primitive_gate(values, 3)
