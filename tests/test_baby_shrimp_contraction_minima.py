"""Analytic and orchestration controls for BSC5; no model or GPU calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_contraction_minima as probe


def plan():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "docs/baby-shrimp-zoom/contraction-minima-plan.json").read_text())


def samples(function, center, step):
    return [[function(np.asarray(center) + step*np.array([x, y]))
             for x in (-1, 0, 1)] for y in (-1, 0, 1)]


def test_mixed_quadratic_jet_recovers_derivatives_at_its_actual_center():
    H = np.array([[-.04, .013], [.013, .022]])
    gradient = np.array([.27, -.19])
    center = np.array([.23, -.41])
    def field(q):
        return 1.3 + gradient @ q + .5*q @ H @ q
    for h in (.1, .03):
        jet = probe.field_jet(samples(field, center, h), h)
        assert jet["value"] == pytest.approx(field(center))
        np.testing.assert_allclose(jet["gradient"], gradient+H@center, atol=1e-13)
        np.testing.assert_allclose(jet["hessian"], H, atol=1e-12)


def test_cubic_gradient_discrepancy_has_second_order_scale():
    # A solver residual is not an accuracy estimate: the true center gradient
    # vanishes while the central-difference gradient retains a known h^2 bias.
    def field(q):
        x, y = q
        return -.02*x*x + .01*y*y + 7*x**3 - 3*y**3
    errors = []
    for h in (.01, .003):
        jet = probe.field_jet(samples(field, [0., 0.], h), h)
        np.testing.assert_allclose(jet["gradient"], h*h*np.array([7., -3.]), atol=1e-17)
        errors.append(np.linalg.norm(jet["gradient"]))
    assert errors[0]/errors[1] == pytest.approx((.01/.003)**2)


@pytest.mark.parametrize("step", [0., -1., float("nan"), float("inf")])
def test_invalid_stencil_scale_fails_loudly(step):
    with pytest.raises(ValueError):
        probe.field_jet(np.eye(3), step)


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.full((3, 3), np.nan),
                                    np.full((3, 3), np.inf)])
def test_invalid_stencil_values_fail_loudly(values):
    with pytest.raises(ValueError):
        probe.field_jet(values, .001)


@pytest.mark.parametrize("angle", [0., .37, 1.2])
def test_four_null_arms_are_unit_opposite_rays_of_two_distinct_lines(angle):
    Q = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    H = Q @ np.diag([-.043, .013]) @ Q.T
    arms = probe.null_arms(H)
    assert len(arms) == len({a["arm_id"] for a in arms}) == 4
    assert arms == probe.null_arms(H)
    for a in arms:
        t, n = np.array(a["tangent"]), np.array(a["normal"])
        assert np.linalg.norm(t) == pytest.approx(1.)
        assert np.linalg.norm(n) == pytest.approx(1.)
        assert t @ n == pytest.approx(0., abs=1e-15)
        assert t @ H @ t == pytest.approx(0., abs=1e-15)
    np.testing.assert_allclose(arms[0]["tangent"], -np.array(arms[1]["tangent"]))
    np.testing.assert_allclose(arms[2]["tangent"], -np.array(arms[3]["tangent"]))
    assert abs(np.linalg.det(np.array([arms[0]["tangent"], arms[2]["tangent"]]))) > .5


@pytest.mark.parametrize("H", [np.eye(2), -np.eye(2), np.diag([-1., 0.]),
                               [[-1., 1.], [0., 1.]], np.diag([-1e-12, 1.]),
                               np.full((2, 2), np.nan)])
def test_null_arms_refuse_non_saddles_and_unresolved_frames(H):
    with pytest.raises(ValueError):
        probe.null_arms(H)


def predictor_fixture():
    previous = dict(kind="continued saddle", parameter=.006, center=[2., -3.], tau=4e-5)
    current = dict(kind="continued saddle", parameter=.007, center=[3., -5.], tau=-5e-5)
    return {"receipt": {"accepted_centers": [
        dict(kind="continued saddle", parameter=.005, center=[0., 0.], tau=.1),
        previous,
        dict(kind="other", parameter=.0069, center=[1e4, 1e4], tau=.8),
        current,
        dict(kind="continued saddle", parameter=.008, center=[8., 9.], tau=-.2),
    ]}}, {"center": current}


def test_predictor_uses_nearest_lower_corrected_saddle_not_other_or_future_rows():
    bundle, seed = predictor_fixture()
    result = probe.predictor_derivatives(bundle, seed)
    assert result["lower"]["parameter"] == .006
    assert result["upper"]["parameter"] == .007
    assert result["height_secant"] == pytest.approx(-.09)
    np.testing.assert_allclose(result["center_secant"], [1000., -2000.])
    assert "predictor only" in result["scope"]


def test_predictor_refuses_missing_lower_sample():
    bundle, seed = predictor_fixture()
    bundle["receipt"]["accepted_centers"] = [seed["center"]]
    with pytest.raises(ValueError, match="missing lower"):
        probe.predictor_derivatives(bundle, seed)


@pytest.mark.parametrize("tau", [4e-5, 5e-5, float("nan"), float("inf")])
def test_predictor_refuses_nonnegative_or_nonfinite_height_secant(tau):
    bundle, seed = predictor_fixture()
    seed["center"]["tau"] = tau
    with pytest.raises(ValueError, match="invalid measured predictor"):
        probe.predictor_derivatives(bundle, seed)


def test_joint_refinement_solves_a_translating_quadratic_saddle():
    H = np.array([[-.04, .01], [.01, .02]])
    true_center = np.array([.23, -.41])
    offset = np.array([2e-5, -3e-5])
    beta, h0, hp = .00644853, 1.6e-10, -.089
    cp = np.array([12000., -70000.])
    old_center = true_center+offset
    jet = dict(center=old_center.tolist(), beta=beta, fields={"tau": dict(
        value=float(h0+.5*offset@H@offset), gradient=(H@offset).tolist(), hessian=H.tolist())})
    result = probe.refine_step(jet, dict(height_secant=hp, center_secant=cp.tolist()), 1e-7)
    db = result["new_beta"]-beta
    new_true_center = true_center + cp*db
    np.testing.assert_allclose(result["step_center"], -offset, atol=1e-19)
    assert result["stationary_height_predictor"] == pytest.approx(h0, rel=1e-13)
    assert result["step_beta"] == pytest.approx(-h0/hp)
    np.testing.assert_allclose(result["new_center"], new_true_center, atol=5e-14)
    assert abs(h0+hp*db) < 1e-18


@pytest.mark.parametrize("height,gradient", [(1e-4, [0., 0.]), (0., [.01, 0.])])
def test_joint_refinement_refuses_beta_or_center_trust_escape(height, gradient):
    jet = dict(center=[0., 0.], beta=.006, fields={"tau": dict(
        value=height, gradient=gradient, hessian=[[-.04, 0.], [0., .02]])})
    with pytest.raises(ValueError, match="trust region"):
        probe.refine_step(jet, dict(height_secant=-.09, center_secant=[0., 0.]), 1e-7)


class QuadraticEvaluator:
    def __init__(self):
        self.rows = []

    def __call__(self, coordinate, beta, cyclic=False):
        x, y = np.asarray(coordinate)-[.2, -.3]
        tau, delta = 1e-10-.02*x*x+.01*y*y+.005*x*y, -1e-8+1e-10*x
        rho = float((np.sqrt(tau*tau-4*delta)+abs(tau))/2)
        row = dict(index=len(self.rows), coordinate=np.asarray(coordinate).tolist(), beta=beta,
                   tau=float(tau), delta=float(delta), radius=rho, qualified=True)
        self.rows.append(row)
        return row


def test_local_jet_records_real_same_center_coordinates_and_both_smooth_fields():
    ev = QuadraticEvaluator()
    center = np.array([.2, -.3])
    jet = probe.local_jet(ev, center, .006, .001)
    np.testing.assert_allclose(jet["fields"]["tau"]["hessian"], [[-.04, .005], [.005, .02]], atol=1e-13)
    np.testing.assert_allclose(jet["fields"]["delta"]["gradient"], [1e-10, 0.], atol=1e-18)
    assert jet["correction_norm"] < 1e-12
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            row = ev.rows[jet["point_indices"][iy][ix]]
            np.testing.assert_array_equal(row["coordinate"], center+.001*np.array([dx, dy]))
            assert row["beta"] == .006
    q = np.array([.0004, -.0007])
    measured = ev(center+q, .006)
    assert probe.predict_field(jet, "tau", q) == pytest.approx(measured["tau"], abs=1e-18)
    assert probe.predict_field(jet, "delta", q) == pytest.approx(measured["delta"], abs=1e-20)


def test_local_jet_stops_on_unqualified_orbit():
    ev = QuadraticEvaluator()
    def bad(*args, **kwargs):
        row = ev(*args, **kwargs)
        row["qualified"] = False
        return row
    with pytest.raises(ValueError, match="unqualified"):
        probe.local_jet(bad, [.2, -.3], .006, .001)
    assert len(ev.rows) == 1


@pytest.mark.parametrize("value", [True, 1.5, 0, float("inf"), float("nan")])
def test_plan_requires_integer_bounded_evaluation_budget(value):
    p = plan()
    p["maximum_evaluations"] = value
    with pytest.raises(ValueError):
        probe.validate_plan(p)


def test_declared_section_budget_can_retain_two_competing_cusp_minima():
    from contraction_section_minima import search_section_minima
    cfg = plan()["contraction_minima"]
    def run(budget):
        calls = []
        def objective(s):
            calls.append(s)
            return {"radius": 8e-5+.004*abs(abs(s)-.0002), "index": len(calls)-1}
        return search_section_minima(objective, cfg["section_mesh"],
            coordinate_tolerance=cfg["coordinate_tolerance"], max_evaluations=budget,
            radius_floor=cfg["radius_floor_minimum"], signal_multiple=cfg["signal_multiple"],
            validation_offsets=cfg["validation_offsets"])
    short = run(24)
    assert not short["complete"] and short["status"] == "budget_exhausted"
    assert len(short["candidates"]) == 2
    enough = run(cfg["section_maximum_evaluations"])
    assert enough["complete"] and enough["status"] == "converged"
    assert len(enough["minima"]) == 2
    assert [c["best_sample"]["s"] for c in enough["minima"]] == pytest.approx(
        [-.0002, .0002], abs=cfg["coordinate_tolerance"])


def test_expanded_spectrum_disagreement_preserves_arrays_before_rejection(tmp_path, monkeypatch):
    import baby_shrimp_memory as memory
    directory = tmp_path/"point-0000"
    directory.mkdir()
    np.save(directory/"points.npy", np.ones((18, 896)))
    point = dict(index=0, coordinate=[0., 0.], beta=.006, phases=[dict(
        phase=p, multipliers=[[1e-4, 0.], [-1e-4, 0.]], radius=1e-4,
        tau=0., delta=-1e-8) for p in (0, 6, 12)])
    values = np.array([2e-4, -2e-4]+[1e-9]*10)
    fake = dict(values=values, vectors=np.zeros((896, 12)), direct=np.zeros((896, 12)),
                relative_residuals=np.zeros(12), q=np.zeros((896, 12)), w=np.zeros((896, 12)), history=[])
    monkeypatch.setattr(memory, "qualified_ritz", lambda *a, **k: fake)
    ev = SimpleNamespace(out=tmp_path, origin=np.zeros(2), scales=np.ones(2),
        cell=lambda *a: SimpleNamespace(shape=(896,)),
        torch=SimpleNamespace(as_tensor=lambda a, **k: np.asarray(a), float64=np.float64),
        loop=SimpleNamespace(device="cpu"), started=time.monotonic(), args=SimpleNamespace(seconds=600),
        apply=lambda *a: None)
    with pytest.raises(ValueError, match="disagrees|tracked regime"):
        probe.expanded_check(ev, point, 12)
    saved = tmp_path/"expanded-spectrum"
    assert (saved/"phase-0.npz").exists()
    with np.load(saved/"phase-0.npz") as arrays:
        np.testing.assert_array_equal(arrays["values"], values)
    partial = json.loads((saved/"check.json").read_text())
    assert partial["complete"] is False and partial["rows"][0]["phase"] == 0


@pytest.mark.parametrize("section_status", ["budget_exhausted", "evaluation_failed", "nonconverged"])
@pytest.mark.parametrize("expanded_trace", [0., 2e-12])
def test_main_cannot_report_complete_after_failed_admission_or_section(
        tmp_path, monkeypatch, section_status, expanded_trace):
    """Run the orchestration with synthetic callbacks, without loading a model."""
    import baby_shrimp_memory_orbit_alignment as alignment
    import baby_shrimp_memory_resume as resume
    import baby_shrimp_saddle_memory as saddle
    import contraction_section_minima as section
    root, data = tmp_path/"repo", tmp_path/"data"
    root.mkdir(); data.mkdir()
    p = plan()
    p["prior_audit"] = dict(path="prior.json", bytes=2, sha256="unused")
    (root/"plan.json").write_text(json.dumps(p))
    (data/"prior.json").write_text("{}")
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "DATA", data)
    monkeypatch.setattr(probe, "digest", lambda p: hashlib.sha256(b"").hexdigest())
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **k: b"")
    bundle = dict(scientific_metadata=p)
    monkeypatch.setattr(alignment, "load_receipt", lambda *a: bundle)
    seed = dict(center=dict(parameter=.006, center=[0., 0.], tau=0.),
                point=dict(coordinate=[0., 0.], beta=.006), orbit=np.ones((18, 896)), bases={})
    monkeypatch.setattr(probe, "seed_records", lambda *a: [seed])
    monkeypatch.setattr(probe, "predictor_derivatives", lambda *a: {})
    monkeypatch.setattr(probe, "replay_seed", lambda *a: {})
    monkeypatch.setattr(resume, "seed_evaluator", lambda *a: None)
    jet = dict(center=[0., 0.], beta=.006, correction_norm=0., fields={
        "tau": dict(value=0., gradient=[0., 0.], hessian=[[-.04, 0.], [0., .02]])})
    monkeypatch.setattr(probe, "local_jet", lambda *a: copy.deepcopy(jet))
    monkeypatch.setattr(probe, "expanded_check", lambda *a: dict(
        variations={"radius": 0.}, rows=[dict(phase=i, tau=expanded_trace) for i in (0, 6, 12)]))
    class FakeEvaluator:
        def __init__(self, args, plan, out, emit):
            self.out, self.attempts = out, 0
        def baseline(self):
            return {"passed": True}
        def __call__(self, coordinate, beta, cyclic=False):
            dest = self.out/"point-0000"
            dest.mkdir(exist_ok=True)
            np.save(dest/"points.npy", np.ones((18, 896)))
            for phase in (0, 6, 12):
                np.savez(dest/f"phase-{phase}.npz", q=np.zeros((896, 6)))
            return dict(index=0, qualified=True, coordinate=np.asarray(coordinate).tolist(), beta=beta,
                        tau=0., delta=-1e-8, radius=1e-4,
                        phases=[dict(phase=i, tau=0.) for i in (0, 6, 12)])
    monkeypatch.setattr(saddle, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(section, "search_section_minima", lambda *a, **k: dict(
        status=section_status, complete=False, minima=[], candidates=[]))
    out = data/"explore"/"test"
    monkeypatch.setattr(sys, "argv", ["probe", "--plan", "plan.json", "--out", str(out),
                                     "--source-commit", "a"*40])
    try:
        probe.main()
    except (ValueError, RuntimeError):
        pass  # An explicit stop is also valid, provided the partial evidence remains.
    progress = json.loads((out/"probe-progress.json").read_text())
    if expanded_trace > p["contraction_minima"]["height_tolerance"]:
        assert not progress["center_qualified"] and not progress["sections"]
        assert progress["expanded_spectrum"]["rows"][0]["tau"] == expanded_trace
    else:
        assert progress["sections"], "must reach the synthetic section, not fail earlier"
    assert progress["complete"] is False
    assert all(s["complete"] is False for s in progress["sections"])
    assert json.loads((out/"manifest.json").read_text())["complete"] is False
