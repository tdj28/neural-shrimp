import numpy as np
from numpy.polynomial import Chebyshev
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_jones as jones


def test_two_distinct_turns_and_escaped_critical_orbit():
    x=np.linspace(0,1,301)
    y=10*(x-.2)*(x-.5)*(x-.8)+.5
    model=jones.fit(x,y,7)
    expected=np.sort(np.roots([3,-3,.66]))
    assert np.allclose(jones.critical(model),expected)
    escaped=Chebyshev([2],domain=[0,1])
    result=jones.itinerary(escaped,np.array([.2]),0)
    assert len(result["values"])==1 and result["reason"].startswith("stopped")


def test_multivalued_response_does_not_look_precise():
    rng=np.random.default_rng(4)
    x=rng.uniform(0,1,500)
    y=x*x+rng.choice([-.2,.2],500)
    model=jones.fit(x[:250],y[:250],7)
    assert np.sqrt(np.mean((model(x[250:])-y[250:])**2))>.1


def test_missing_support_fails_instead_of_inventing_curve():
    with pytest.raises(ValueError,match="insufficient"):
        jones.fit(np.array([0,.5,1]),np.array([1,0,.5]),7)
    with pytest.raises(ValueError,match="collapsed"):
        jones.reconstruct(np.ones((100,3)))
