"""New synthetic saved-schema controls only; all fixtures/reports are retained.

Run with TASK37_SAVED_CONTROL_ROOT set to a new directory. No model/runner imports.
"""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import eig
import audit_saved_numerics as checker


ROOT = Path(os.environ["TASK37_SAVED_CONTROL_ROOT"])
ROOT.mkdir(parents=True, exist_ok=False)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def geom(a, b=None):
    scale = np.linalg.norm(a, axis=1).mean()
    out = dict(proper_divisor_phase_separations={str(q): float(max(np.linalg.norm(a-np.roll(a, -q, axis=0), axis=1))/scale)
                                               for q in (1, 2, 3)}, primitive=True)
    if b is not None:
        normb = np.linalg.norm(b, axis=1).mean(); den = (scale+normb)/2
        distances = np.asarray([np.linalg.norm(a-np.roll(b, -s, axis=0), axis=1) for s in range(6)])
        maxima = distances.max(axis=1); best = int(np.argmin(maxima)); rel = float(maxima[best]/den)
        out.update(best_shift=best, all_shift_max_distances=maxima.tolist(),
                   phase_distances_at_best_shift=distances[best].tolist(), largest_distance_after_best_shift=float(maxima[best]),
                   mean_state_norm_a=float(scale), mean_state_norm_b=float(normb), normalization_mean_state_norm=float(den),
                   relative_largest_distance=rel,
                   comparison_if_both_closed="same_within_tolerance" if rel <= 1e-10 else "distinct_numerically")
    return out


def fixture(root, shifted=False):
    root.mkdir()
    temperature = float(10**-.154104)
    dump(root/"model-ready.json", dict(temperature_binary64=temperature))
    a = np.eye(6); b = np.roll(a, 3, axis=0) if shifted else a+2
    summaries = []
    for index, phases in enumerate((a, b)):
        p = root/f"seed{index}"; p.mkdir(); x = phases[0].copy()
        counts = dict(composed_calls=0, jvp_columns=0, jvp_calls=0, gmres_unconverged=0, linear_solves=0)
        events = []; sequence = 0
        def emit(kind, **fields):
            events.append(dict(event=kind, counts=counts.copy(), elapsed_seconds=0., **fields))
        m = np.diag([.8, .8, .7, .6, .5, .4]); m[0, 1] = -.2; m[1, 0] = .2
        def call(state, direction=None):
            nonlocal sequence
            number = sequence; sequence += 1
            operation = "composed_map" if direction is None else "composed_jvp" if direction.ndim == 1 else "composed_jvp_batch"
            columns = 0 if direction is None else 1 if direction.ndim == 1 else len(direction)
            counts["composed_calls"] += int(direction is None); counts["jvp_calls"] += int(direction is not None)
            counts["jvp_columns"] += columns
            stem = str(p/f"operator-{number:06d}")
            np.save(stem+"-state.npy", state)
            if direction is not None: np.save(stem+"-direction.npy", direction)
            emit("operator_start", identifier=number, operation=operation, columns=columns)
            center = phases[np.argmin(np.linalg.norm(phases-state, axis=1))]; delta = state-center
            result = center+m@delta+delta**3 if direction is None else direction@(m+np.diag(3*delta**2)).T
            np.save(stem+"-result.npy", result); emit("operator_end", identifier=number, operation=operation)
            return result
        emit("start"); call(x)
        solve = dict(converged=True, relative_residual=0., updates=0, reason="converged", counts=counts.copy())
        dump(p/"solve-result.json", solve); np.save(p/"solution-or-last-state.npy", x)
        np.save(p/"phases.npy", phases)
        tokens = []
        for j in range(6):
            emit("phase_logits_start", phase=j)
            logits = np.roll(np.arange(7, dtype=float)/7, j)
            np.save(p/f"phase-{j}-logits.npy", logits)
            np.save(p/f"phase-{j}-image.npy", phases[(j+1)%6]); emit("phase_logits_end", phase=j)
            tokens.append(dict(phase=j, token_id=int(np.argmax(logits)), text="synthetic",
                               probability=float(1/np.exp((logits-max(logits))/temperature).sum()), logit_gap=float(1/7)))
        for state in phases: call(state)
        matrix = call(x, np.eye(6)).T
        np.save(p/"monodromy.npy", matrix); np.save(p/"monodromy-columns-0000-0006.npy", matrix)
        values, left, modes = eig(m, left=True, right=True); order = np.argsort(-abs(values))
        values = values[order]; modes = modes[:, order]; left = left[:, order]
        np.save(p/"eigenvectors.npy", modes)
        scale = max(1., np.sqrt(np.linalg.norm(m, 1)*np.linalg.norm(m, np.inf)))
        residuals = np.linalg.norm(m@modes-modes*values, axis=0)/scale; records = []
        for j in range(6):
            v = modes[:, j]; ad = call(x, v.real)
            if np.linalg.norm(v.imag) > 0: ad = ad.astype(complex)+1j*call(x, v.imag)
            records.append(dict(real=float(values[j].real), imag=float(values[j].imag), modulus=float(abs(values[j])),
                                argument=float(np.angle(values[j])), eigenvalue_condition=float(1/abs(np.vdot(left[:, j], v))),
                                dense_relative_residual=float(residuals[j]), scalar_jvp_relative_residual=float(np.linalg.norm(ad-values[j]*v)/scale)))
        spec = dict(spectral_radius=float(abs(values[0])), next_largest_modulus=float(abs(values[1])), eigen_gate=True,
                    stability_if_orbit_and_fd_gates_pass="numerically_attracting", stability_margin=1e-7, largest=records,
                    all_multipliers=[[float(v.real), float(v.imag)] for v in values],
                    residual_normalization="max(1,sqrt(norm1(M)*norminf(M)))",
                    crossing_type="not_classified_no_bisected_crossing", max_dense_relative_residual=float(max(residuals)))
        dump(p/"spectrum.json", spec)
        rng = np.random.default_rng(370001); random_errors = []
        for _ in range(3):
            d = rng.normal(size=6); d /= np.linalg.norm(d); actual = call(x, d)
            random_errors.append(float(np.linalg.norm(actual-m@d)/max(np.linalg.norm(actual), 1e-14)))
        fd = []
        for name, d in (("random0", rng.normal(size=6)), ("random1", rng.normal(size=6)),
                        ("critical_real", modes[:, 0].real.copy()), ("critical_imag", modes[:, 0].imag.copy())):
            if np.linalg.norm(d) <= 1e-12: continue
            d /= np.linalg.norm(d); ad = call(x, d); rows = []
            for rel in (1e-3, 5e-4, 2.5e-4, 1.25e-4):
                h = rel*np.linalg.norm(x); plus = call(x+h*d); minus = call(x-h*d)
                error = np.linalg.norm((plus-minus)/(2*h)-ad)
                rows.append(dict(relative_step=rel, h=float(h), absolute_error=float(error),
                                 relative_error=float(error/max(np.linalg.norm(ad), 1e-14)),
                                 cancellation_floor=float(32*np.finfo(float).eps*max(np.linalg.norm(plus), np.linalg.norm(minus))/h)))
            ratios = [rows[j]["absolute_error"]/max(rows[j+1]["absolute_error"], 1e-300) for j in range(3)]
            at_floor = all(r["absolute_error"] <= r["cancellation_floor"] for r in rows)
            quadratic = any(2.5 <= r <= 6 for r in ratios)
            gate = dict(passed=bool(rows[-1]["relative_error"] <= 1e-5 and (quadratic or at_floor)),
                        quadratic_interval_seen=quadratic, all_at_cancellation_floor=at_floor,
                        error_ratios=ratios, steps=rows, ad_direction_norm=float(np.linalg.norm(ad)))
            emit("derivative_gate", result=gate); fd.append(dict(component=name, **gate))
        nearest = int(np.argmin(abs(values+1)))
        summary = dict(seed=index, newton=solve, qualified=True, certificate_passed=False,
                       nearest_minus_one=dict(sorted_index=nearest, real=float(values[nearest].real), imag=float(values[nearest].imag),
                                              distance=float(abs(values[nearest]+1)), rounded_R_is_not_a_bisected_crossing=True),
                       phase_closure_relative=0., primitive=geom(phases), one_step_phase_calls=6,
                       tokens=tokens,
                       all_phase_return_residuals_relative=[0.]*6, spectrum=spec, composed_fd=fd,
                       random_dense_scalar_jvp_errors=random_errors, nonlinear_random_quadratic_interval=True, counts=counts.copy())
        dump(p/"result.json", summary); summaries.append(summary)
        end = dict(completed_without_uncaught_exception=True, error_type=None, error=None, counts=counts.copy())
        dump(p/"attempt-end.json", end); emit("attempt_end", result=end)
        (p/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    dump(root/"result.json", dict(status="completed", orbits=summaries, both_orbits_numerically_qualified=True,
                                  alignment=geom(a, b), body_points_eligible=not shifted, body_points_run=False, certificate_passed=False))


class SavedControls(unittest.TestCase):
    def run_fixture(self, mutation=None, shifted=False):
        directory = ROOT/self._testMethodName; fixture(directory, shifted)
        if mutation: mutation(directory)
        report = checker.audit(directory, expected_dimension=6, expected_vocabulary=7)
        dump(ROOT/(self._testMethodName+"-report.json"), report)
        self.assertFalse(report["scientific_verdict_admitted"])
        return report

    def test_complete_saved_arithmetic_never_admits_science(self):
        report = self.run_fixture()
        self.assertTrue(report["saved_numerical_checks_passed"], report)
        self.assertEqual(report["alignment_from_arrays"]["comparison_if_both_closed"], "distinct_numerically")
        self.assertEqual(len(report["seeds"][0]["all_dense_eigenpair_residuals"]), 6)

    def test_cyclic_alignment_uses_all_six_phases(self):
        report = self.run_fixture(shifted=True)
        self.assertTrue(report["saved_numerical_checks_passed"], report)
        self.assertEqual(report["alignment_from_arrays"]["best_shift"], 3)
        self.assertEqual(report["alignment_from_arrays"]["relative_largest_distance"], 0.)

    def test_matrix_corruption_refused(self):
        def mutate(p):
            m = np.load(p/"seed0/monodromy.npy"); m[0, 0] += .1; np.save(p/"seed0/monodromy.npy", m)
        self.assertFalse(self.run_fixture(mutate)["saved_numerical_checks_passed"])

    def test_missing_operator_result_refused(self):
        self.assertFalse(self.run_fixture(lambda p: (p/"seed0/operator-000001-result.npy").unlink())["saved_numerical_checks_passed"])

    def test_phase_chain_corruption_refused(self):
        self.assertFalse(self.run_fixture(lambda p: np.save(p/"seed0/phase-2-image.npy", np.zeros(6)))["saved_numerical_checks_passed"])

    def test_fd_claim_corruption_refused(self):
        def mutate(p):
            path = p/"seed0/result.json"; data = json.loads(path.read_text())
            data["composed_fd"][0]["steps"][-1]["relative_error"] += .01; dump(path, data)
        self.assertFalse(self.run_fixture(mutate)["saved_numerical_checks_passed"])

    def test_repeated_saved_eigenvalue_refused(self):
        def mutate(p):
            path = p/"seed0/spectrum.json"; data = json.loads(path.read_text())
            data["all_multipliers"][-1] = data["all_multipliers"][0]; dump(path, data)
            result = p/"seed0/result.json"; summary = json.loads(result.read_text()); summary["spectrum"] = data; dump(result, summary)
        self.assertFalse(self.run_fixture(mutate)["saved_numerical_checks_passed"])

    def test_token_rank_corruption_refused(self):
        def mutate(p):
            path = p/"seed0/result.json"; data = json.loads(path.read_text())
            data["tokens"][0]["token_id"] = 0; dump(path, data)
        self.assertFalse(self.run_fixture(mutate)["saved_numerical_checks_passed"])

    def test_retained_fd_array_corruption_refused(self):
        def mutate(p):
            events = [json.loads(s) for s in (p/"seed0/events.jsonl").read_text().splitlines()]
            gate = next(j for j, e in enumerate(events) if e["event"] == "derivative_gate")
            ident = next(e["identifier"] for e in reversed(events[:gate]) if e["event"] == "operator_end")
            path = p/f"seed0/operator-{ident:06d}-result.npy"; value = np.load(path); value[0] += .01; np.save(path, value)
        self.assertFalse(self.run_fixture(mutate)["saved_numerical_checks_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
