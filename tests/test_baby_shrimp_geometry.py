"""Known spectra and composed operators guard the geometry worker's checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_geometry as geometry


def test_spectrum_retains_conjugate_pair_and_dominance_exchange():
    matrix = np.array([[.01, -.03, 0], [.03, .01, 0], [0, 0, -.02]])
    values, vectors, residual = geometry.spectrum(matrix)
    assert np.allclose(values[:2].real, .01)
    assert np.allclose(np.sort(values[:2].imag), [-.03, .03])
    assert values[2] == pytest.approx(-.02)
    assert residual.max() < 1e-12
    assert np.allclose(matrix @ vectors, vectors * values)
    matrix[2, 2] = -.04
    assert geometry.spectrum(matrix)[0][0] == pytest.approx(-.04)


def test_composed_gate_rejects_substituting_a_one_step_jacobian():
    torch = pytest.importorskip("torch")
    first = torch.tensor([[.7, 4.], [0., .2]], dtype=torch.float64)
    second = torch.tensor([[.3, 0.], [-2., .8]], dtype=torch.float64)

    class Orbit:
        n = 2

        def jvp_rows(self, point, tangent):
            return tangent @ (first if point[0, 0] == 0 else second).T

    points = [torch.zeros((1, 2), dtype=torch.float64), torch.ones((1, 2), dtype=torch.float64)]
    matrix = (second @ first).numpy()
    values, vectors, residual = geometry.spectrum(matrix)
    assert geometry.composed_check(Orbit(), points, matrix, vectors)[3].max() < 1e-12
    assert geometry.composed_check(Orbit(), points, first.numpy(), vectors)[3].max() > .1


def test_plan_bands_cover_the_declared_grid_once():
    plan = json.loads((Path(__file__).resolve().parents[1] / "docs/baby-shrimp-zoom/geometry-plan.json").read_text())
    rows = [r for b in range(2) for r in range(*geometry.validate_plan(plan, b))]
    assert rows == list(range(21))
    plan["newton_tolerance"] = 1e-5
    with pytest.raises(ValueError, match="gates"):
        geometry.validate_plan(plan, 0)


def test_nonfinite_monodromy_is_not_classified():
    with pytest.raises(ValueError, match="finite"):
        geometry.spectrum(np.array([[np.nan]]))


def test_watched_spectrum_keeps_conjugate_companions_and_excludes_rounding_tail():
    assert geometry.qualified_spectral_count(np.array([.006, -5e-6, 3e-12])) == 2
    assert geometry.qualified_spectral_count(np.array([.1, 1e-6+2e-6j, 1e-6-2e-6j, 1e-14])) == 3
    assert geometry.qualified_spectral_count(np.array([.01, -.008, .003, 1e-12])) == 3


def test_orbit_propagation_does_not_qualify_failed_spectrum():
    row = {"row": 11, "column": 19, "passed": False,
           "orbit_passed": True, "same_cyclic_text": True}
    assert not geometry.propagation_passed(row, {})
    assert geometry.propagation_passed(row, {"propagation_gate": "primitive_orbit"})
    status = geometry.completion_status([row], 0, 310)
    assert status["orbit_solved"] == 1
    assert status["solved"] == 0
    assert not status["complete"]
    assert status["reason"] == "no qualified spectral cell"
    row["same_cyclic_text"] = False
    assert not geometry.propagation_passed(row, {"propagation_gate": "primitive_orbit"})


def test_empty_or_incomplete_continuation_is_not_complete():
    assert not geometry.completion_status([], 0, 310)["complete"]
    row = {"row": 0, "column": 0, "passed": True, "orbit_passed": True}
    assert not geometry.completion_status([row], 1, 310)["complete"]
    status = geometry.completion_status([row, dict(row)], 0, 310)
    assert status["complete"] and status["solved"] == 1
