"""Baseline total memory sensitivity, including the saved P18 orbit's motion.

One phase only in this pilot. Nested forward AD is checked against decreasing
finite differences of the one-step JVP along the periodic orbit tangent.
Unsupported second derivatives fail with preserved evidence, never a fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"

import numpy as np

from baby_shrimp_saddle_memory import Evaluator, ROOT, DATA, save, sha
from baby_shrimp_orbits import primitive_gate
from baby_shrimp_geometry import canonical
from periodic_memory_sensitivity import periodic_memory_trace_sensitivity, memory_product_derivatives
import llm_feigenbaum as lfb


FD_STEPS = (1e-3, 3e-4, 1e-4)


def normalized_eta_step(state, eta, relative_step):
    """Return h with ||h eta||/||state||=relative_step; None for eta=0."""
    state, eta = np.asarray(state), np.asarray(eta)
    if (state.shape != eta.shape or not state.size or not np.isfinite(state).all() or
            not np.isfinite(eta).all() or not np.isfinite(relative_step) or relative_step <= 0):
        raise ValueError("finite matching states/tangents and a positive relative step are required")
    norm_eta, norm_state = float(np.linalg.norm(eta)), float(np.linalg.norm(state))
    if norm_eta == 0:
        return None
    if norm_state == 0:
        raise ValueError("a nonzero state is required to normalize a nonzero tangent step")
    step = relative_step * norm_state / norm_eta
    if not np.isfinite(step) or step <= 0:
        raise ValueError("the normalized Hessian finite-difference step is invalid")
    return step


def nested_hessian_rows(jvp_rows, point, eta, directions):
    """D_x[J(x)V][eta] by forward-over-forward AD, preserving row batches."""
    import torch
    if point.shape != eta.shape or directions.shape[1:] != point.shape[1:]:
        raise ValueError("nested AD state and tangent shapes differ")
    _, result = torch.func.jvp(lambda x: jvp_rows(x, directions), (point,), (eta,))
    if result.shape != directions.shape or not bool(torch.isfinite(result).all()):
        raise ValueError("nested AD returned an invalid Hessian-direction product")
    return result


def finite_difference_hessian_rows(jvp_rows, point, eta, directions, step):
    """Central difference of J(x)V, using an already normalized scalar step."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("a finite positive Hessian difference step is required")
    return (jvp_rows(point+step*eta, directions)-jvp_rows(point-step*eta, directions))/(2*step)


def observed_orders(errors, steps=FD_STEPS):
    errors, steps = np.asarray(errors, float), np.asarray(steps, float)
    if errors.shape != steps.shape or len(errors) < 2 or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("finite nonnegative errors matching the finite-difference steps are required")
    if not np.isfinite(steps).all() or np.any(steps <= 0) or np.any(steps[:-1] <= steps[1:]):
        raise ValueError("finite positive decreasing steps are required")
    return [float(np.log(a/b)/np.log(h/k)) if a > 0 and b > 0 else None
            for a, b, h, k in zip(errors[:-1], errors[1:], steps[:-1], steps[1:])]


def _compact(result):
    orbit = {key: value for key, value in result["orbit_tangent"].items()
             if key not in ("tangents", "forcing", "zero_start_return")}
    pair = {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result["pair"].items() if key not in ("left", "right")}
    return {"passed": result["passed"], "orbit_tangent": orbit, "pair": pair,
            "operator_relative_errors": result["operator_relative_errors"].tolist(),
            "operator_tolerance": result["operator_tolerance"],
            "trace_derivatives": result["trace_derivatives"],
            "multiplier_derivatives": {key: value.tolist() for key, value in result["multiplier_derivatives"].items()},
            "scope": result["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="docs/baby-shrimp-zoom/saddle-memory-plan.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if not 0 < args.seconds <= 900:
        parser.error("this pilot is bounded by at most 900 seconds")
    args.out = args.out.resolve()
    if not args.out.is_relative_to(DATA.resolve()):
        parser.error("raw output must be under the external neural-shrimp data directory")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads((ROOT/args.plan).read_text())

    def check_time():
        if time.monotonic()-started > args.seconds:
            raise TimeoutError("bounded total-sensitivity pilot expired")

    def emit(stage, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
               "seconds": time.monotonic()-started, **fields}
        with (args.out/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    sources = ["tools/baby_shrimp_memory_total.py", "tools/periodic_memory_sensitivity.py",
               "tools/trace_pair_sensitivity.py", "tools/baby_shrimp_saddle_memory.py",
               "tools/baby_shrimp_memory.py", "tools/baby_shrimp_orbits.py", "tools/baby_shrimp_geometry.py",
               "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
               "tools/explore_llm_phase_map.py", "tools/llm_feigenbaum.py"]
    save(args.out/"protocol.json", {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {path: sha(ROOT/path) for path in sources},
        "plan_sha256": sha(ROOT/args.plan), "phase": 6, "period": 18,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hessian_difference_steps": FD_STEPS,
        "step_definition": "h_i=relative_step*norm(x_i)/norm(eta_i); zero eta gives an exact zero Hessian contribution",
        "gates": {"best_trace_and_column_relative_error": 1e-5,
                  "required_orders": "both consecutive orders of maximum total-column error in [1.5,2.5]",
                  "orbit_connection_tolerance": 1e-11},
        "scope": "One-phase total fixed-alpha/T memory derivative. Not a corrected-saddle height or native mode-coupling certificate."})
    try:
        ev = Evaluator(args, plan, args.out, emit)
        orb = ev.cell(ev.origin, 0)
        raw = np.load(ev.anchor/"points.npy", allow_pickle=False)
        if raw.shape != (18, orb.n) or not np.isfinite(raw).all():
            raise ValueError("the saved orbit has invalid dimensions or values")
        rolled = np.roll(raw, -6, axis=0)
        points = [ev.torch.as_tensor(row.reshape((1,)+orb.shape), dtype=ev.torch.float64,
                                    device=ev.loop.device) for row in rolled]
        check_time()
        final = lfb._flat(orb.base.f(points[-1]))
        states = np.vstack([rolled, final])
        fresh = orb.points(points[0], 18)
        returns = [float((point-points[0]).norm()/points[0].norm()) for point in fresh[1:]]
        primitive = primitive_gate(returns, 18)
        fresh_states = np.stack([lfb._flat(point) for point in fresh])
        connection = np.linalg.norm(fresh_states-states, axis=1)/np.maximum(np.linalg.norm(states, axis=1), 1e-300)
        tokens = [int(orb.probabilities(point).argmax()) for point in points]
        same_text = canonical(tokens) == canonical([271, 785, 3405, 271, 785, 13885]*3)
        qualification = {"primitive": primitive, "maximum_replayed_state_relative_difference": float(connection.max()),
                         "tokens": tokens, "same_cyclic_text": same_text}
        np.savez_compressed(args.out/"orbit-inputs.npz", states=states, replayed_states=fresh_states,
                            replayed_relative_differences=connection)
        save(args.out/"orbit-qualification.json", qualification)
        if not (primitive["closed"] and primitive["no_lower_period"] and same_text and connection.max() < 1e-11):
            raise ValueError("saved orbit failed independent state, primitive-period or text-family replay")
        with np.load(ev.anchor/"phase-6.npz", allow_pickle=False) as saved:
            matrix = saved["matrix"]

        def apply_j(index, columns):
            check_time()
            answer = lfb._cols(orb.base.jvp_rows(points[index], lfb._rows(columns, points[index])))
            check_time()
            return answer

        def apply_ad_h(index, eta, columns):
            check_time()
            point = points[index]
            tangent = ev.torch.as_tensor(eta.reshape(point.shape), dtype=point.dtype, device=point.device)
            emit("hessian_ad_start", phase_index=index)
            answer = lfb._cols(nested_hessian_rows(orb.base.jvp_rows, point, tangent, lfb._rows(columns, point)))
            emit("hessian_ad_complete", phase_index=index)
            check_time()
            return answer

        js = [lambda columns, index=index: apply_j(index, columns) for index in range(18)]
        hs = [lambda eta, columns, index=index: apply_ad_h(index, eta, columns) for index in range(18)]
        emit("total_ad_start")
        ad = periodic_memory_trace_sensitivity(matrix, states, js, hs)
        np.savez_compressed(args.out/"ad-columns.npz", **ad["derivative_columns"],
                            tangents=ad["orbit_tangent"]["tangents"], forcing=ad["orbit_tangent"]["forcing"],
                            zero_start_return=ad["orbit_tangent"]["zero_start_return"],
                            left=ad["pair"]["left"], right=ad["pair"]["right"], multipliers=ad["pair"]["multipliers"])
        compact = _compact(ad)
        save(args.out/"ad-result.json", compact)
        emit("total_ad_complete", trace_derivatives=ad["trace_derivatives"])
        fd_rows = []
        for relative in FD_STEPS:
            actual_steps = [normalized_eta_step(states[index], ad["orbit_tangent"]["tangents"][index], relative)
                            for index in range(18)]

            def apply_fd_h(index, eta, columns):
                check_time()
                if actual_steps[index] is None:
                    return np.zeros_like(columns)
                point = points[index]
                tangent = ev.torch.as_tensor(eta.reshape(point.shape), dtype=point.dtype, device=point.device)
                answer = lfb._cols(finite_difference_hessian_rows(
                    orb.base.jvp_rows, point, tangent, lfb._rows(columns, point), actual_steps[index]))
                check_time()
                return answer

            fhs = [lambda eta, columns, index=index: apply_fd_h(index, eta, columns) for index in range(18)]
            emit("total_fd_start", relative_step=relative)
            columns = memory_product_derivatives(js, fhs, ad["orbit_tangent"]["tangents"], ad["pair"]["right"])
            derivatives = {key: np.einsum("ij,ij->j", ad["pair"]["left"], columns[key])
                           for key in ("explicit", "orbit_mediated", "total")}
            traces = {key: float(value.sum()) for key, value in derivatives.items()}
            column_error = np.linalg.norm(columns["total"]-ad["derivative_columns"]["total"], axis=0)/np.maximum(
                np.linalg.norm(ad["derivative_columns"]["total"], axis=0), 1e-300)
            absolute_trace_error = abs(traces["total"]-ad["trace_derivatives"]["total"])
            trace_error = absolute_trace_error/max(abs(ad["trace_derivatives"]["total"]), 1e-300)
            row = {"relative_step": relative, "actual_parameter_steps": actual_steps,
                   "trace_derivatives": traces, "multiplier_derivatives": {key: value.tolist() for key, value in derivatives.items()},
                   "total_column_relative_errors": column_error.tolist(),
                   "total_trace_absolute_error": absolute_trace_error, "total_trace_relative_error": float(trace_error)}
            np.savez_compressed(args.out/f"fd-columns-{relative:.0e}.npz", **columns,
                                actual_parameter_steps=np.array([np.nan if step is None else step for step in actual_steps]))
            fd_rows.append(row)
            save(args.out/"fd-partial.json", fd_rows)
            emit("total_fd_complete", **row)
        column_errors = [max(row["total_column_relative_errors"]) for row in fd_rows]
        trace_errors = [row["total_trace_relative_error"] for row in fd_rows]
        column_orders, trace_orders = observed_orders(column_errors), observed_orders(trace_errors)
        quadratic = all(order is not None and 1.5 <= order <= 2.5 for order in column_orders)
        accurate = min(max(a, b) for a, b in zip(column_errors, trace_errors)) < 1e-5
        result = {"complete": True, "passed": bool(quadratic and accurate), "phase": 6,
                  "ad": compact, "finite_differences": fd_rows,
                  "total_column_error_orders": column_orders, "total_trace_error_orders": trace_orders,
                  "quadratic_regime_found": quadratic, "joint_trace_and_column_accuracy_passed": accurate,
                  "seconds": time.monotonic()-started,
                  "scope": "One-phase total memory response at the saved alpha/T. Stationarity, other phases and native internal coupling are not established by this pilot."}
        save(args.out/"result.json", result)
        if not result["passed"]:
            raise ValueError("total Hessian sensitivity did not pass the declared finite-difference precision gate")
    except BaseException as error:
        save(args.out/"failure.json", {"type": type(error).__name__, "error": str(error),
                                     "traceback": traceback.format_exc(), "seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
