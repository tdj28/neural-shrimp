import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('henon_audit', Path(__file__).parents[1]/'tools/audit_henon_skeleton.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


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
