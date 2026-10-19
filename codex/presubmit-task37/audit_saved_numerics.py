"""Read-only replay of the p6_kite retained-array schema; never admits science.

No model, producer, runner or periodic_audit imports. JSON is checked against
arrays, not used as evidence that the underlying operator/model was authentic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def consistent(actual, claimed, label="summary"):
    """Compare recomputed fields, allowing only floating arithmetic roundoff."""
    if isinstance(actual, dict):
        require(isinstance(claimed, dict), label + ": object required")
        for key, value in actual.items():
            require(key in claimed, label + ": missing " + key)
            consistent(value, claimed[key], label + "." + key)
    elif isinstance(actual, (list, tuple)):
        require(isinstance(claimed, list) and len(actual) == len(claimed), label + ": length")
        for index, value in enumerate(actual):
            consistent(value, claimed[index], f"{label}[{index}]")
    elif isinstance(actual, (bool, str)) or actual is None:
        require(type(actual) is type(claimed) and actual == claimed, label + ": mismatch")
    else:
        require(type(claimed) in (int, float) and np.isfinite(claimed) and
                np.isclose(actual, claimed, rtol=1e-10, atol=1e-14), label + ": mismatch")


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def data(self, name):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "input escapes supplied root")
        content = path.read_bytes()
        self.files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        return content

    def json(self, name):
        return json.loads(self.data(name), parse_constant=lambda s: (_ for _ in ()).throw(Invalid(s)))

    def array(self, name, shape=None, complex_ok=False):
        import io
        value = np.load(io.BytesIO(self.data(name)), allow_pickle=False)
        require(value.dtype == np.float64 or (complex_ok and value.dtype == np.complex128),
                name + ": wrong effective saved dtype")
        require(np.isfinite(value).all() and (shape is None or value.shape == shape),
                name + ": nonfinite or wrong shape")
        return value


def geometry(a, b=None):
    scale = float(np.mean(np.linalg.norm(a, axis=1)))
    require(scale > 0, "zero mean phase norm")
    separations = {str(q): float(np.max(np.linalg.norm(a - np.roll(a, -q, axis=0), axis=1)) / scale)
                   for q in (1, 2, 3)}
    out = {"proper_divisor_phase_separations": separations,
           "primitive": all(v > 1e-7 for v in separations.values())}
    if b is not None:
        scale_b = float(np.mean(np.linalg.norm(b, axis=1)))
        denominator = (scale + scale_b) / 2
        distances = [np.linalg.norm(a - np.roll(b, -s, axis=0), axis=1) for s in range(6)]
        maxima = [float(v.max()) for v in distances]
        shift = int(np.argmin(maxima)); relative = maxima[shift] / denominator
        out.update(best_shift=shift, all_shift_max_distances=maxima,
                   phase_distances_at_best_shift=distances[shift].tolist(),
                   largest_distance_after_best_shift=maxima[shift], mean_state_norm_a=scale,
                   mean_state_norm_b=scale_b, normalization_mean_state_norm=denominator,
                   relative_largest_distance=relative,
                   comparison_if_both_closed=("same_within_tolerance" if relative <= 1e-10 else
                                              "distinct_numerically" if relative > 1e-7 else "unresolved"))
    return out


def journal(reader, prefix, n):
    events = [json.loads(line) for line in reader.data(prefix + "events.jsonl").splitlines()]
    require(events and events[0]["event"] == "start" and events[-1]["event"] == "attempt_end",
            "incomplete event journal")
    calls, pending = [], None
    counts = dict(composed_calls=0, jvp_columns=0, jvp_calls=0)
    for position, event in enumerate(events):
        kind = event["event"]
        require(kind != "operator_failed", "failed operator retained: attempt cannot pass")
        if kind == "operator_start":
            require(pending is None and event["identifier"] == len(calls), "operator sequence")
            ident = event["identifier"]; stem = prefix + f"operator-{ident:06d}"
            operation = event["operation"]
            require(operation in ("composed_map", "composed_jvp", "composed_jvp_batch"), "unknown operator")
            x = reader.array(stem + "-state.npy", (n,))
            direction = None if operation == "composed_map" else reader.array(stem + "-direction.npy")
            shape = (n,) if operation != "composed_jvp_batch" else (len(direction), n)
            require(direction is None or direction.shape == shape, "operator direction shape")
            columns = 0 if direction is None else (1 if direction.ndim == 1 else len(direction))
            require(event["columns"] == columns, "operator column count")
            counts["composed_calls"] += int(direction is None)
            counts["jvp_columns"] += columns; counts["jvp_calls"] += int(direction is not None)
            pending = dict(id=ident, operation=operation, x=x, direction=direction,
                           y=reader.array(stem + "-result.npy", shape), position=position)
        elif kind == "operator_end":
            require(pending is not None and event["identifier"] == pending["id"] and
                    event["operation"] == pending["operation"], "unpaired operator end")
            calls.append(pending); pending = None
        consistent(counts, event["counts"], "event operator counts")
    require(pending is None, "unfinished operator")
    end = reader.json(prefix + "attempt-end.json")
    consistent(counts, end["counts"], "attempt counts")
    require(end["completed_without_uncaught_exception"] is True, "uncaught attempt failure")
    return events, calls, counts


def replay_fd(calls, x, claimed):
    require(len(calls) == 9 and calls[0]["operation"] == "composed_jvp", "FD call block")
    adcall = calls[0]; d = adcall["direction"]; ad = adcall["y"]
    require(np.array_equal(adcall["x"], x) and abs(np.linalg.norm(d) - 1) < 1e-12, "FD base/tangent")
    rows = []
    for index, rel in enumerate((1e-3, 5e-4, 2.5e-4, 1.25e-4)):
        h = rel * np.linalg.norm(x); plus, minus = calls[1+2*index:3+2*index]
        require(plus["operation"] == minus["operation"] == "composed_map" and
                np.array_equal(plus["x"], x+h*d) and np.array_equal(minus["x"], x-h*d), "FD offsets")
        error = float(np.linalg.norm((plus["y"]-minus["y"])/(2*h)-ad))
        floor = float(32*np.finfo(float).eps*max(np.linalg.norm(plus["y"]), np.linalg.norm(minus["y"]))/h)
        rows.append(dict(relative_step=rel, h=float(h), absolute_error=error,
                         relative_error=error/max(float(np.linalg.norm(ad)), 1e-14), cancellation_floor=floor))
    ratios = [rows[i]["absolute_error"]/max(rows[i+1]["absolute_error"], 1e-300) for i in range(3)]
    floor = all(row["absolute_error"] <= row["cancellation_floor"] for row in rows)
    quadratic = any(2.5 <= v <= 6 for v in ratios)
    result = dict(passed=bool(rows[-1]["relative_error"] <= 1e-5 and (quadratic or floor)),
                  quadratic_interval_seen=quadratic, all_at_cancellation_floor=floor,
                  error_ratios=ratios, steps=rows, ad_direction_norm=float(np.linalg.norm(ad)))
    consistent(result, claimed, "FD event")
    return result


def audit_seed(reader, index, n, vocabulary):
    p = f"seed{index}/"; summary = reader.json(p+"result.json")
    solve = reader.json(p+"solve-result.json")
    require(solve["converged"] is True, "Newton did not converge")
    consistent(solve, summary["newton"], "Newton summary")
    x = reader.array(p+"solution-or-last-state.npy", (n,))
    phases = reader.array(p+"phases.npy", (6, n))
    require(np.array_equal(x, phases[0]) and np.all(np.linalg.norm(phases, axis=1) > 0), "phase origin/norm")
    images = np.stack([reader.array(p+f"phase-{j}-image.npy", (n,)) for j in range(6)])
    require(np.array_equal(images[:-1], phases[1:]), "one-step phase chain")
    closure = float(np.linalg.norm(images[-1]-x)/np.linalg.norm(x))
    primitive = geometry(phases)
    consistent(dict(phase_closure_relative=closure, primitive=primitive, one_step_phase_calls=6), summary)
    temperature = reader.json("model-ready.json")["temperature_binary64"]
    require(type(temperature) in (int, float) and np.isfinite(temperature) and temperature > 0,
            "invalid saved readout temperature")
    require(len(summary["tokens"]) == 6, "six token records required")
    tokens = []
    for phase in range(6):
        logits = reader.array(p+f"phase-{phase}-logits.npy", (vocabulary,))
        top = int(np.argmax(logits)); highest = np.partition(logits, -2)[-2:]
        probability = float(1/np.exp((logits-logits[top])/temperature).sum())
        row = dict(phase=phase, token_id=top, probability=probability, logit_gap=float(highest[1]-highest[0]))
        consistent(row, summary["tokens"][phase], "saved-logit readout"); tokens.append(row)
    events, calls, counts = journal(reader, p, n)
    phase_start = min(j for j, e in enumerate(events) if e["event"] == "phase_logits_start")
    terminal = [c for c in calls if c["position"] < phase_start and c["operation"] == "composed_map"
                and np.array_equal(c["x"], x)]
    require(terminal, "no retained Newton evaluation at final state")
    newton_residual = float(np.linalg.norm(terminal[-1]["y"]-x)/np.linalg.norm(x))
    consistent(newton_residual, solve["relative_residual"], "terminal Newton residual")
    require(newton_residual <= 1e-11, "terminal Newton residual failed")
    require([e["phase"] for e in events if e["event"] == "phase_logits_end"] == list(range(6)), "phase event sequence")
    phase_end = max(j for j, e in enumerate(events) if e["event"] == "phase_logits_end")
    after = [c for c in calls if c["position"] > phase_end]
    require(len(after) >= 6, "missing composed phase returns")
    returns = []
    for j, c in enumerate(after[:6]):
        require(c["operation"] == "composed_map" and np.array_equal(c["x"], phases[j]), "phase-return argument")
        returns.append(float(np.linalg.norm(c["y"]-phases[j])/np.linalg.norm(phases[j])))
    consistent(returns, summary["all_phase_return_residuals_relative"], "phase returns")
    # Dense columns must be the actual consecutive standard-basis operator calls.
    m = reader.array(p+"monodromy.npy", (n, n)); cursor = 6
    for start in range(0, n, 8):
        stop = min(start+8, n); c = after[cursor]; cursor += 1
        require(c["operation"] == "composed_jvp_batch" and np.array_equal(c["x"], x) and
                np.array_equal(c["direction"], np.eye(n)[start:stop]), "dense basis block")
        chunk = reader.array(p+f"monodromy-columns-{start:04d}-{stop:04d}.npy", (n, stop-start))
        require(np.array_equal(chunk, c["y"].T) and np.array_equal(chunk, m[:, start:stop]), "dense saved columns")
    spec = reader.json(p+"spectrum.json"); consistent(spec, summary["spectrum"], "spectrum copies")
    values = np.asarray([complex(*v) for v in spec["all_multipliers"]])
    require(values.shape == (n,) and np.isfinite(values).all(), "multiplier shape/finite")
    modes = reader.array(p+"eigenvectors.npy", (n, n), complex_ok=True)
    require(np.allclose(np.linalg.norm(modes, axis=0), 1, rtol=0, atol=1e-12), "eigenvector normalization")
    scale = max(1., float(np.sqrt(np.linalg.norm(m, 1)*np.linalg.norm(m, np.inf))))
    residuals = np.linalg.norm(m@modes-modes*values, axis=0)/scale
    # Fresh eigenvalues plus one-to-one assignment catch repeated/omitted saved modes.
    fresh = np.linalg.eigvals(m); cost = abs(values[:, None]-fresh[None, :])
    ii, jj = linear_sum_assignment(cost); spectrum_error = float(cost[ii, jj].max()/scale)
    require(spectrum_error <= 1e-10, "saved spectrum differs from fresh full spectrum")
    require(np.all(np.diff(abs(values)) <= 1e-12*scale), "multipliers not modulus sorted")
    nearest = int(np.argmin(abs(values+1)))
    consistent(dict(sorted_index=nearest, real=float(values[nearest].real), imag=float(values[nearest].imag),
                    distance=float(abs(values[nearest]+1)), rounded_R_is_not_a_bisected_crossing=True),
               summary["nearest_minus_one"], "nearest minus one")
    scalar = []
    for j in range(min(8, n)):
        v = modes[:, j]; result = np.zeros(n, dtype=np.complex128)
        for part, coefficient in ((v.real, 1), (v.imag, 1j)):
            if coefficient == 1j and not np.any(part):
                continue
            c = after[cursor]; cursor += 1
            require(c["operation"] == "composed_jvp" and np.array_equal(c["x"], x) and
                    np.array_equal(c["direction"], part), "eigenmode scalar JVP")
            result += coefficient*c["y"]
        error = float(np.linalg.norm(result-values[j]*v)/scale); scalar.append(error)
        consistent(dict(real=float(values[j].real), imag=float(values[j].imag), modulus=float(abs(values[j])),
                        argument=float(np.angle(values[j])), dense_relative_residual=float(residuals[j]),
                        scalar_jvp_relative_residual=error), spec["largest"][j], "largest multiplier")
    rho = float(max(abs(values)))
    eigen_gate = bool(max(residuals) <= 1e-10 and max(scalar) <= 1e-10)
    stability = "numerically_attracting" if rho < 1-1e-7 else "numerically_unstable" if rho > 1+1e-7 else "unresolved_near_unit_circle"
    consistent(dict(spectral_radius=rho, next_largest_modulus=float(abs(values[1])), eigen_gate=eigen_gate,
                    max_dense_relative_residual=float(max(residuals)), stability_margin=1e-7,
                    stability_if_orbit_and_fd_gates_pass=stability,
                    crossing_type="not_classified_no_bisected_crossing"), spec)
    random_errors = []; rng = np.random.default_rng(370001)
    for _ in range(3):
        d = rng.normal(size=n); d /= np.linalg.norm(d); c = after[cursor]; cursor += 1
        require(c["operation"] == "composed_jvp" and np.array_equal(c["x"], x) and
                np.array_equal(c["direction"], d), "random scalar JVP argument")
        random_errors.append(float(np.linalg.norm(c["y"]-m@d)/max(np.linalg.norm(c["y"]), 1e-14)))
    consistent(random_errors, summary["random_dense_scalar_jvp_errors"], "random dense/scalar")
    expected = [("random0", rng.normal(size=n)), ("random1", rng.normal(size=n)),
                ("critical_real", modes[:, 0].real), ("critical_imag", modes[:, 0].imag)]
    expected = [(name, d/np.linalg.norm(d)) for name, d in expected if np.linalg.norm(d) > 1e-12]
    gates = [(pos, e) for pos, e in enumerate(events) if e["event"] == "derivative_gate"]
    require(len(gates) == len(expected) == len(summary["composed_fd"]), "FD component count")
    fd = []
    for (name, d), (pos, event), claimed in zip(expected, gates, summary["composed_fd"]):
        block = after[cursor:cursor+9]; cursor += 9
        require(len(block) == 9 and block[-1]["position"] < pos and
                (cursor == len(after) or pos < after[cursor]["position"]), "FD event association")
        require(np.array_equal(block[0]["direction"], d), "FD component direction")
        row = dict(component=name, **replay_fd(block, x, event["result"]))
        consistent(row, claimed, "FD summary"); fd.append(row)
    require(cursor == len(after), "unaccounted post-phase operator calls")
    quadratic = any(v["quadratic_interval_seen"] for v in fd if v["component"].startswith("random"))
    passed = bool(closure <= 1e-11 and max(returns) <= 1e-11 and primitive["primitive"] and
                  eigen_gate and max(random_errors) <= 1e-10 and quadratic and all(v["passed"] for v in fd))
    consistent(dict(qualified=passed, nonlinear_random_quadratic_interval=quadratic, certificate_passed=False), summary)
    consistent(counts, summary["counts"], "final counts")
    return dict(saved_numerical_checks_passed=passed, phase_closure_relative=closure,
                all_phase_return_residuals_relative=returns, primitive=primitive,
                all_dense_eigenpair_residuals=residuals.tolist(), fresh_spectrum_relative_error=spectrum_error,
                spectral_radius=rho, conditional_stability=stability, random_dense_scalar_errors=random_errors,
                composed_fd=fd, operator_counts=counts, tokens_from_saved_logits=tokens,
                readout_temperature_from_saved_metadata=temperature), phases, summary


def audit(root, expected_dimension=896, expected_vocabulary=151936):
    reader = Reader(root)
    report = dict(schema="task37-saved-numerics-v1", expected_dimension=expected_dimension,
                  expected_vocabulary=expected_vocabulary,
                  scientific_verdict_admitted=False, certificate_passed=False, seeds=[],
                  authentication_required=["original supervisor exit and attempt identity", "source commit and source pins",
                                           "input/seed and model/tokenizer inventory hashes", "run card and admission hashes",
                                           "complete raw-file inventory bound to original supervisor"],
                  not_recomputed=["left-eigenvector condition estimates", "tokenizer text and embedding readout",
                                  "Newton/GMRES iteration history, linear-solves and unconverged-GMRES counters",
                                  "model/operator authenticity", "effective internal arithmetic precision"])
    phases = []; summaries = []
    for index in range(2):
        try:
            row, orbit, summary = audit_seed(reader, index, expected_dimension, expected_vocabulary)
            phases.append(orbit); summaries.append(summary)
            report["seeds"].append(dict(seed=index, **row))
        except (Invalid, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            report["seeds"].append(dict(seed=index, saved_numerical_checks_passed=False,
                                        error_type=type(error).__name__, error=str(error)))
    report["saved_numerical_checks_passed"] = all(r["saved_numerical_checks_passed"] for r in report["seeds"])
    try:
        root_summary = reader.json("result.json")
        if len(phases) == 2:
            report["alignment_from_arrays"] = geometry(*phases)
            consistent(summaries, root_summary["orbits"], "root orbit copies")
        both = report["saved_numerical_checks_passed"]
        consistent(dict(status="completed", both_orbits_numerically_qualified=both,
                        certificate_passed=False, body_points_run=False,
                        body_points_eligible=bool(both and report["alignment_from_arrays"]["comparison_if_both_closed"] == "distinct_numerically")), root_summary)
        consistent(report["alignment_from_arrays"] if both else None, root_summary["alignment"], "root alignment")
    except (Invalid, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        report["saved_numerical_checks_passed"] = False
        report["root_summary_error"] = f"{type(error).__name__}: {error}"
    report["read_file_inventory"] = dict(sorted(reader.files.items()))
    report["inventory_scope"] = "Files read by this checker only; not an authenticated original or complete attempt inventory."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", required=True, help="new output path; existing files are refused")
    args = parser.parse_args()
    result = audit(args.root)
    with Path(args.report).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    raise SystemExit(0 if result["saved_numerical_checks_passed"] else 1)


if __name__ == "__main__":
    main()
