"""Independent algebraic checks; no target pilot outcomes are generated."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("concepts",Path(__file__).resolve().parents[1]/"tools/build_concept_figures.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize("eta",[.5001,.52,.55,.6,.63,.65])
def test_exact_cycle_closure_and_independent_observable_errors(eta):
    result=module.cycle_values(eta);u,v=result["low"],result["high"]
    mapping=lambda x:x-eta*2*x*(x*x-1)
    np.testing.assert_allclose([mapping(u),mapping(v)],[v,u],atol=3e-15,rtol=3e-15)
    assert u!=v
    independent_multiplier=(1-eta*(6*u*u-2))*(1-eta*(6*v*v-2))
    assert result["multiplier"]==pytest.approx(independent_multiplier,abs=1e-13)
    assert result["mean_phase_mse"]==pytest.approx(((u*u-1)**2+(v*v-1)**2)/2,abs=1e-14)
    assert result["mean_prediction_mse"]==pytest.approx(((u*u+v*v)/2-1)**2,abs=1e-14)
    assert result["mean_weight_mse"]==pytest.approx((((u+v)/2)**2-1)**2,abs=1e-14)
    assert result["mean_prediction_mse"]<result["mean_weight_mse"]<result["mean_phase_mse"]


def test_stability_interval_and_branch_birth():
    upper=(np.sqrt(5)-1)/2
    assert module.cycle_values(upper)["multiplier"]==pytest.approx(-1)
    assert abs(module.cycle_values(.55)["multiplier"])<1
    assert abs(module.cycle_values(.64)["multiplier"])>1
    with pytest.raises(ValueError):module.cycle_values(.5)
