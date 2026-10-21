import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_crossing as crossing


def test_trace_saddle_step_and_nonzero_saddle_value():
    h=.1;x0,y0=.2,-.3
    field=np.array([[(x+x0)**2-2*(y+y0)**2+.002 for x in (-h,0,h)] for y in (-h,0,h)])
    step,g,H=crossing.saddle_step(field,h)
    assert np.allclose(step,[-x0,-y0])
    assert np.allclose(np.linalg.eigvalsh(H),[-4,2])
    assert field[1,1]+g@step+.5*step@H@step==pytest.approx(.002)


def test_empty_flat_or_partial_stencil_does_not_default():
    with pytest.raises(ValueError):crossing.saddle_step(np.ones((3,3)),.1)
    with pytest.raises(ValueError):crossing.saddle_step(np.full((3,3),np.nan),.1)


def test_pair_consistency_ignores_dominance_order():
    a=np.array([.01,-.009])
    assert crossing.pair_discrepancy(a,a[::-1])==0
    assert crossing.pair_discrepancy(a,np.array([.01,-.008]))==pytest.approx(.001)
