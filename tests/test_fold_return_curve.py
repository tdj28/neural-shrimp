import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('fold_return_curve',Path(__file__).parents[1]/'tools/fold_return_curve.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_polynomial_state_curve_preserves_critical_coordinate():
    centre=np.array([.2,.1,.4]);left=np.array([1.,0.,0.]);right=np.array([1.,2.,-1.])
    v=np.linspace(-.3,.7,30)
    states=centre+v[:,None]*right+np.column_stack([v*0,v**2,.4*v**3])
    model=m.fit_curve(states,centre,left,right,degree=3)
    unseen=np.linspace(-.25,.65,17)
    predicted=m.reconstruct(unseen,model,centre,right)
    expected=centre+unseen[:,None]*right+np.column_stack([unseen*0,unseen**2,.4*unseen**3])
    np.testing.assert_allclose(predicted,expected,atol=1e-14)
    np.testing.assert_allclose((predicted-centre)@left,unseen,atol=1e-15)


def test_curve_fit_refuses_a_zero_length_interval():
    with pytest.raises(ValueError):m.fit_curve(np.ones((10,3)),np.zeros(3),np.array([1.,0.,0.]),np.array([1.,0.,0.]))
