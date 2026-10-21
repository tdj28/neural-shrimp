import importlib.util
from pathlib import Path
import numpy as np
import sys

spec = importlib.util.spec_from_file_location('henon_audit', Path(__file__).parents[1]/'tools/audit_henon_skeleton.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import audit_henon_skeleton_geometry as hg


def test_product_jacobian_against_complex_step():
    z = np.array([-.8, 1.2, -.5, .7, .3, 1.5, .17])
    numerical = np.column_stack([h.constraints(z+1e-25j*np.eye(len(z))[i]).imag/1e-25 for i in range(len(z))])
    np.testing.assert_allclose(h.jacobian(z), numerical, rtol=1e-12, atol=1e-12)


def test_determinant_identity_and_cyclic_invariance():
    xs = np.array([-.8, 1.2, -.5, .7, .3])
    b = .17
    m = h.monodromy(xs, b)
    for k in range(5):
        mk = h.monodromy(np.roll(xs, k), b)
        np.testing.assert_allclose(np.linalg.det(mk), (-b)**5, rtol=1e-10)
        np.testing.assert_allclose(np.trace(mk), np.trace(m), rtol=1e-12)


def test_gate_rejects_lower_period_even_with_zero_trace():
    z = np.r_[np.zeros(5), 0., .17]
    assert max(abs(h.constraints(z))) == 0
    assert not h.gates(z)['passed']


def test_serialized_precision_coordinates_reconstruct_nonzero_parameter():
    b = .16
    s = 1-b
    xs = np.roots([1, -s, -b/2])
    a = s*s+b/2
    row = h.high_precision(np.r_[xs, a, b], 60)
    assert row['passed']
    assert abs(float(row['a'])-a) < 1e-14
    saved = np.array([*map(float, row['states']), float(row['a']), float(row['b'])])
    assert max(abs(h.constraints(saved))) < 1e-14
    assert hg.replay(h.gates(saved))['passed']


def test_segment_crossing_and_gap_are_distinguished():
    a,b,c,d = [np.array(v,dtype=float) for v in [(0,0),(1,1),(0,1),(1,0)]]
    np.testing.assert_allclose(hg.segment_intersection(a,b,c,d), [.5,.5])
    assert hg.segment_intersection(a,b,a+[0,1],b+[0,1]) is None


def test_orbit_trace_gradient_against_fixed_parameter_resolve():
    b = .16
    xs = np.roots([1, -(1-b), -b/2])
    z = np.r_[xs,(1-b)**2+b/2,b]
    g = hg.total_trace_gradient(z)
    numerical=[]
    for k in (-2,-1):
        values=[]
        for sign in (-1,1):
            v=z.copy()
            v[k]+=sign*1e-5
            sol=hg.root(lambda x:h.constraints(np.r_[x,v[-2:]])[:-1],xs,tol=1e-12)
            assert max(abs(h.constraints(np.r_[sol.x,v[-2:]])[:-1])) < 1e-12
            values.append(h.constraints(np.r_[sol.x,v[-2:]])[-1])
        numerical.append((values[1]-values[0])/2e-5)
    np.testing.assert_allclose(g,numerical,rtol=1e-8,atol=1e-8)
