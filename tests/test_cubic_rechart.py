from fractions import Fraction
import pytest
from flint import arb,ctx
from neural_shrimp.cubic_rechart import rotate_cubic_endpoint,adaptive_width
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_pal_rechart import rechart_seed


def test_cubic_endpoint_derivatives_and_new_gauge():
    with ctx.workprec(256):
        # p(d)=(d, d^3); actual endpoint derivative (1,3h^2), not (1,0).
        h=arb(1)/8;out=rotate_cubic_endpoint([h,h**3],[arb(1),arb(0)],[arb(0)]*2,[arb(0),arb(1)],h)
        n=[exact(v)for v in out['normal']];v=[exact(v)for v in out['slope']];a=[exact(v)for v in out['curvature']]
        assert abs(sum(n[i]*v[i]for i in range(2))-1)<arb((1,-245))
        assert abs(sum(n[i]*a[i]for i in range(2)))<arb((1,-245))
        assert abs(v[1]/v[0]-3*h*h)<arb((1,-245))
        oldv=[arb(1),3*h*h];oldacc=[arb(0),6*h];g=sum(n[i]*oldv[i]for i in range(2));gd=sum(n[i]*oldacc[i]for i in range(2))
        assert all(abs(a[i]-(oldacc[i]-v[i]*gd)/g**2)<arb((1,-245))for i in range(2))
        assert rechart_seed(out,'0')==out['origin']


def test_proposal_keeps_all_coordinates_and_rejects_bad_velocity():
    with ctx.workprec(128):
        n=705;point=[arb(i)/1024 for i in range(n)]
        o=rotate_cubic_endpoint(point,[arb(1)]+[arb(0)]*(n-1),[arb(0)]*n,[arb(0)]*n,arb(1)/32)
        assert len(o['origin'])==len(o['normal'])==len(o['slope'])==len(o['curvature'])==705
        with pytest.raises(ValueError):rotate_cubic_endpoint([arb(0)],[arb(0)],[arb(0)],[arb(0)],arb(1)/32)


def test_adaptation_only_retries_complete_root_bounds_and_stops_join_failure():
    assert adaptive_width(-5,root_passed=False,joined=False,other_gates=True)==(-6,'retry_halfwidth')
    assert adaptive_width(-6,root_passed=True,joined=True,other_gates=True)==(-6,'advance_rotate')
    for kwargs in [dict(root_passed=True,joined=False,other_gates=True),dict(root_passed=False,joined=False,other_gates=False)]:assert adaptive_width(-5,**kwargs)==(-5,'stop')
    assert adaptive_width(-9,root_passed=False,joined=False,other_gates=True)==(-9,'stop')
