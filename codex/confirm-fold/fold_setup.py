"""Task5 independent fold setup. Never runs passage-test trajectories.

Use only after a committed setup run card. Parameter A is physical alpha;
parameter B is log10(T), including its chain rule. No producer tools imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import resource
import re
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codex.crosscheck.numerics import cycle_points, fold_refine, orbit_diagnostics
from codex.crosscheck.qwen_map import ManualQwenSoftMap, PINNED_SHA256, PREFIX_IDS, sha256


@dataclass(frozen=True)
class Target:
    label: str
    period: int
    coordinate: str
    fixed_coordinate: str
    fixed_value: float
    initial_parameter: float
    bounds: tuple[float, float]


TARGETS = {
    "A": Target("A", 2, "alpha", "log10T", -.140, .0670, (.0660, .0690)),
    "B": Target("B", 8, "log10T", "alpha", .0717, -.1033, (-.108, -.100)),
}
OWNED_OUTPUT = Path("/data2/neural-shrimp-data/codex/task5-fresh-prediction")


def tensor(value):
    result = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64, device="cpu")
    if result.ndim != 1 or not torch.isfinite(result).all():
        raise ValueError("finite full one-dimensional state/mode required")
    return result


def array(value):
    result = value.detach().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite derivative or model result")
    return result


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def json_save(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(plain(payload), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Operations:
    """Coordinate-aware callbacks; no detached cache in parameter AD paths."""
    def __init__(self, model, target, *, chunk=32, save=None, check=None, bounds=None):
        self.model, self.target, self.chunk = model, target, int(chunk)
        self.bounds = tuple(target.bounds if bounds is None else bounds)
        if self.chunk < 1 or len(self.bounds) != 2 or not np.isfinite(self.bounds).all() or self.bounds[0] >= self.bounds[1]:
            raise ValueError("positive chunk and finite ordered parameter bounds required")
        self.save = save or (lambda stage, payload: None)
        self.check = check or (lambda: None)
        self.counts, self.stages = Counter(), Counter()
        self.cached_alpha, self.cache = None, None

    def emit(self, stage, payload):
        self.stages[stage] += 1
        self.save(f"{stage}-{self.stages[stage]:03d}", dict(payload, operation_counts=dict(self.counts)))

    def coordinates(self, parameter):
        # Preserve scalar tensors: d/dlog10T must include ln(10)*T.
        if self.target.coordinate == "alpha":
            return parameter, self.target.fixed_value
        if self.target.coordinate == "log10T":
            return self.target.fixed_value, parameter
        raise ValueError("unsupported active coordinate")

    def guard(self, parameter, operation):
        self.check()
        if not np.isfinite(parameter) or not self.bounds[0] <= parameter <= self.bounds[1]:
            raise ValueError("active parameter outside declared target bounds")
        self.counts[operation] += 1

    def fixed(self, parameter):
        alpha, temperature = self.coordinates(float(parameter))
        if self.cache is None or alpha != self.cached_alpha:
            self.cache = self.model.prefix_cache(alpha)
            self.cached_alpha = alpha
            self.counts["prefix_cache"] += 1
        return alpha, temperature, self.cache

    def torch_map(self, state, parameter):
        alpha, temperature = self.coordinates(parameter)
        return self.model.period_map(state, alpha, temperature, self.target.period)

    def step(self, state, parameter):
        self.guard(parameter, "one_step")
        alpha, temperature, cache = self.fixed(parameter)
        result = array(self.model.forward(tensor(state), alpha, temperature, cache=cache))
        self.check()
        return result

    def jvp(self, state, direction, parameter):
        self.guard(parameter, "one_step_jvp")
        alpha, temperature, cache = self.fixed(parameter)
        result = array(self.model.jvp(tensor(state), tensor(direction), alpha, temperature, period=1, cache=cache))
        self.check()
        return result

    def period_map(self, state, parameter):
        self.guard(parameter, "period_map")
        alpha, temperature, cache = self.fixed(parameter)
        result = array(self.model.period_map(tensor(state), alpha, temperature, self.target.period, cache=cache))
        self.check()
        return result

    def dense(self, state, parameter):
        self.guard(parameter, "dense_monodromy")
        self.emit("monodromy-start", dict(state=np.asarray(state), parameter=parameter))
        alpha, temperature, cache = self.fixed(parameter)
        result = array(self.model.jacobian(tensor(state), alpha, temperature, period=self.target.period,
                                           chunk_size=self.chunk, cache=cache))
        if result.shape != (len(state), len(state)):
            raise ValueError("wrong full-state monodromy dimension")
        self.check()
        self.emit("monodromy", dict(state=np.asarray(state), parameter=parameter, matrix=result))
        return result

    def parameter_derivative(self, state, parameter):
        self.guard(parameter, "parameter_ad")
        pt = torch.tensor(float(parameter), dtype=torch.float64)
        result = array(torch.func.jvp(lambda p: self.torch_map(tensor(state), p),
                                      (pt,), (torch.ones_like(pt),))[1])
        self.check()
        self.emit("parameter-ad", dict(parameter=parameter, derivative=result))
        return result

    def critical_grad(self, state, parameter, left, right):
        self.guard(parameter, "critical_gradient_ad")
        xt, lt, rt = tensor(state), tensor(left), tensor(right)
        pt = torch.tensor(float(parameter), dtype=torch.float64)
        def scalar(x, p):
            action = torch.func.jvp(lambda z: self.torch_map(z, p), (x,), (rt,))[1]
            return torch.dot(lt, action)
        gx, gp = torch.func.grad(scalar, argnums=(0, 1))(xt, pt)
        result = array(gx), float(gp.detach())
        if not np.isfinite(result[1]):
            raise ValueError("nonfinite mixed parameter/state derivative")
        self.check()
        self.emit("critical-gradient", dict(parameter=parameter, state_gradient=result[0], parameter_gradient=result[1]))
        return result

    def second(self, state, right, parameter):
        self.guard(parameter, "nested_state_ad")
        xt, rt = tensor(state), tensor(right)
        def first(x):
            return torch.func.jvp(lambda z: self.torch_map(z, float(parameter)), (x,), (rt,))[1]
        result = array(torch.func.jvp(first, (xt,), (rt,))[1])
        self.check()
        self.emit("second-ad", dict(parameter=parameter, derivative=result))
        return result


def classify_fold(matrix, *, critical_tol=1e-8, imaginary_tol=1e-10,
                  complementary_limit=.95, isolation=1e-3, eigen_tol=1e-8):
    """Require a real +1 and separated attracting complement before any NF."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("finite square monodromy required")
    values, modes = np.linalg.eig(matrix)
    k = int(np.argmin(abs(values - 1)))
    critical, others = values[k], np.delete(values, k)
    radius = float(np.max(abs(others))) if others.size else 0.
    if abs(critical.imag) > imaginary_tol or abs(critical.real - 1) > critical_tol:
        raise ValueError("fold gate: no real +1 crossing")
    if radius >= complementary_limit:
        raise ValueError("fold gate: complementary modulus near or above one")
    if others.size and np.min(abs(others - critical)) <= isolation:
        raise ValueError("fold gate: critical mode not isolated")
    r = modes[:, k].real.copy()
    r /= np.linalg.norm(r)
    if r[np.argmax(abs(r))] < 0:
        r *= -1
    lv, lmodes = np.linalg.eig(matrix.T)
    j = int(np.argmin(abs(lv - critical)))
    l = lmodes[:, j].real.copy()
    pairing = float(l @ r)
    if abs(pairing) < 1e-12:
        raise ValueError("fold gate: unresolved left/right pairing")
    l /= pairing
    errors = np.linalg.norm(matrix @ modes - modes * values[None, :], axis=0)
    critical_error = max(np.linalg.norm(matrix @ r - r), np.linalg.norm(matrix.T @ l - l) / np.linalg.norm(l))
    if np.max(errors) >= eigen_tol or critical_error >= eigen_tol:
        raise ValueError("fold gate: eigenpair residual failed")
    return dict(critical_index=k, critical_multiplier=critical, critical_modulus=float(abs(critical)),
                critical_argument_turns=float(np.angle(critical)/(2*np.pi)), next_modulus=radius,
                eigenvalues=values, eigenvectors=modes, left_eigenvalues=lv, left_eigenvectors=lmodes,
                right=r, left=l, eigenpair_residuals=errors, critical_eigen_residual=float(critical_error),
                left_right_pairing=float(l @ r), left_right_cosine=float(1 / np.linalg.norm(l)))


def fd_agrees(table, field, exact, rtol, floor):
    """Each finest FD estimate must agree: averaging cannot hide cancellation."""
    if len(table) < 2 or not np.isfinite(exact):
        return False
    return bool(all(np.isfinite(row[field]) and np.sign(row[field]) == np.sign(exact)
                    and abs(row[field]-exact) <= rtol*max(abs(exact),floor)
                    for row in table[-2:]))


def source_commit_from_environment():
    """The launcher snapshots source without .git and supplies its pinned HEAD."""
    revision = os.environ.get("TASK5_SOURCE_COMMIT", "")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("TASK5_SOURCE_COMMIT must contain the launcher's full 40-hex source commit")
    return revision.lower()


def prepare_fold(model, seed, target, *, parameter=None, bounds=None, chunk=32, save=None, check=None,
                 max_newton=6, max_backtracks=2, complementary_limit=.95, closure_tol=1e-10,
                 primitive_tol=1e-6, derivative_rtol=1e-3, coefficient_floor=1e-10):
    """Locate/qualify a fold without inspecting any passage outcome."""
    seed = array(tensor(seed))
    parameter = target.initial_parameter if parameter is None else float(parameter)
    ops = Operations(model, target, chunk=chunk, save=save, check=check, bounds=bounds)
    ops.emit("initial", dict(seed=seed, parameter=parameter, target=asdict(target), bounds=ops.bounds))
    # Existing Task2 solver names this scalar alpha; here it is target.coordinate.
    solver = fold_refine(ops.period_map, ops.dense, ops.parameter_derivative, ops.critical_grad,
                         seed, parameter, max_newton=max_newton, max_backtracks=max_backtracks)
    ops.emit("corrector", solver)
    result = dict(schema="codex-task5-fold-setup-v1", passed=False, target=asdict(target),
                  active_parameter_units=target.coordinate, parameter_bounds=ops.bounds, solver=solver,
                  evidence="rung2 numerical setup; no interval certificate; no test passages",
                  tolerances=dict(closure=closure_tol, primitive=primitive_tol, complementary_limit=complementary_limit,
                                  derivative_rtol=derivative_rtol, coefficient_floor=coefficient_floor))
    def finish(stage):
        result.update(stage=stage, operation_counts=dict(ops.counts))
        ops.emit("setup-result", result)
        return result
    if not solver["converged"]:
        return finish("fold-corrector-failed")
    x, p, matrix = solver["state"], solver["alpha"], solver["matrix"]
    result.update(parameter_c=p, alpha=ops.coordinates(p)[0], log10T=ops.coordinates(p)[1], x_c=x, matrix=matrix)
    try:
        spectrum = classify_fold(matrix, complementary_limit=complementary_limit)
    except ValueError as error:
        result["gate_failure"] = str(error)
        return finish("crossing-classification-failed")
    result["spectrum"] = spectrum
    fixed = lambda state: ops.step(state, p)
    diagnostics = orbit_diagnostics(fixed, lambda state,v: ops.jvp(state,v,p), x, target.period,
                                    matrix=matrix, closure_tol=closure_tol, separation_tol=primitive_tol)
    diagnostics["raw_strict_radius_below_one"] = diagnostics.pop("spectral_contraction_passed")
    diagnostics["numerically_attracting"] = None
    points = np.stack(cycle_points(fixed, x, target.period))
    phase_returns = [float(np.linalg.norm(ops.period_map(q,p)-q)/max(np.linalg.norm(q),1e-30)) for q in points[:-1]]
    gates = dict(closure=diagnostics["closure_passed"], primitive=diagnostics["primitive_separation_passed"],
                 eigenpairs=diagnostics["eigenpairs_passed"], all_phase_closure=bool(max(phase_returns)<closure_tol))
    result.update(orbitpoints=points, orbit_diagnostics=diagnostics, phase_returns=phase_returns, gates=gates)
    ops.emit("prerequisites", dict(gates=gates, diagnostics=diagnostics, spectrum=spectrum, points=points))
    if not all(gates.values()):
        return finish("orbit-gates-failed")
    r, l = spectrum["right"], spectrum["left"]
    gp, hrr = ops.parameter_derivative(x,p), ops.second(x,r,p)
    raw_a, raw_b = float(l @ gp), float(.5*l @ hrr)
    atable, btable, jtable = [], [], []
    scale = max(float(np.linalg.norm(x)), 1.)
    for h in (1e-5,3e-6,1e-6):
        fd = (ops.period_map(x,p+h)-ops.period_map(x,p-h))/(2*h)
        atable.append(dict(h=h, a=float(l @ fd), vector_relative_error=float(np.linalg.norm(fd-gp)/max(np.linalg.norm(gp),1e-30))))
    for relative_h in (1e-3,3e-4,1e-4):
        h = relative_h*scale
        alpha,t = ops.coordinates(p)
        def action(state):
            ops.guard(p,"composed_state_jvp")
            value = array(model.jvp(tensor(state),tensor(r),alpha,t,period=target.period,cache=ops.fixed(p)[2]))
            ops.check()
            return value
        plus, minus = action(x+h*r), action(x-h*r)
        btable.append(dict(relative_h=relative_h, h=h, b=float(l @ (plus-minus)/(4*h))))
        fd = (ops.period_map(x+h*r,p)-ops.period_map(x-h*r,p))/(2*h)
        jtable.append(dict(relative_h=relative_h, h=h, relative_error=float(np.linalg.norm(fd-matrix@r)/max(np.linalg.norm(matrix@r),1e-30))))
    afd = float(np.mean([row["a"] for row in atable[-2:]]))
    bfd = float(np.mean([row["b"] for row in btable[-2:]]))
    gates.update(nonzero_a=bool(abs(raw_a)>coefficient_floor), nonzero_b=bool(abs(raw_b)>coefficient_floor),
                 parameter_ad_fd=fd_agrees(atable,"a",raw_a,derivative_rtol,coefficient_floor),
                 state_ad_fd=fd_agrees(btable,"b",raw_b,derivative_rtol,coefficient_floor),
                 composed_precision=bool(jtable[-1]["relative_error"]<1e-5 and
                                         (jtable[-1]["relative_error"]<1e-7 or jtable[-1]["relative_error"]<.3*jtable[0]["relative_error"])))
    orientation = float(np.sign(raw_b))
    side = float(np.sign(raw_a*raw_b))
    result.update(raw_a=raw_a, raw_b=raw_b, raw_ab=raw_a*raw_b, raw_r=r, raw_l=l,
                  r=orientation*r, l=orientation*l, a=orientation*raw_a, b=abs(raw_b),
                  a_positive=abs(raw_a), b_positive=abs(raw_b), ghost_parameter_direction=side,
                  orientation=orientation, parameter_derivative=gp, directional_second=hrr,
                  parameter_fd=atable, state_fd=btable, composed_precision=jtable,
                  derivative_discrepancy=dict(a_fd_minus_ad=afd-raw_a,b_fd_minus_ad=bfd-raw_b),
                  coordinate_convention="u=l dot (x-x_c); ||r||=1; l dot r=1; b>0; p=p_c+sign(raw_a*raw_b)*epsilon",
                  U=.1/abs(raw_b) if raw_b else None)
    result["passed"] = bool(all(gates.values()))
    return finish("qualified" if result["passed"] else "derivative-gates-failed")


def load_seed(path, *, key=None, index=None):
    with Path(path).open("rb") as stream:
        value = np.load(stream, allow_pickle=False)
        if isinstance(value,np.lib.npyio.NpzFile):
            try:
                if key is None:
                    raise ValueError("NPZ seed requires explicit --seed-key")
                data = np.asarray(value[key]).copy()
            finally:
                value.close()
        else:
            if key is not None:
                raise ValueError("--seed-key applies only to NPZ")
            data = value
    if index is not None:
        if data.ndim < 2 or not 0 <= index < len(data):
            raise ValueError("seed index requires an existing row of a multi-state array")
        data = data[index]
    data = np.asarray(data).squeeze()
    return array(tensor(data))


def persist(output, name, payload):
    """JSON metadata references full binary arrays, including failed stages."""
    arrays = {}
    def split(value, key):
        if isinstance(value,np.ndarray):
            arrays[key] = value
            return dict(file=name+".npz", key=key, shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value,dict):
            return {str(k):split(v,str(k) if not key else key+"/"+str(k)) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [split(v,key+"/"+str(i)) for i,v in enumerate(value)]
        return plain(value)
    metadata = split(payload, "")
    if arrays:
        np.savez_compressed(output/(name+".npz"), **arrays)
        metadata["array_file_sha256"] = sha256(output/(name+".npz"))
    json_save(output/(name+".json"), metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",choices=sorted(TARGETS),required=True)
    parser.add_argument("--seed",type=Path,required=True)
    parser.add_argument("--seed-key")
    parser.add_argument("--seed-index",type=int)
    parser.add_argument("--parameter",type=float)
    parser.add_argument("--parameter-min",type=float)
    parser.add_argument("--parameter-max",type=float)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--chunk",type=int,default=32)
    parser.add_argument("--wall-seconds",type=float,default=3600)
    parser.add_argument("--max-rss-gb",type=float,default=16)
    parser.add_argument("--max-newton",type=int,default=6)
    parser.add_argument("--max-backtracks",type=int,default=2)
    args = parser.parse_args()
    if not 1<=args.threads<=12 or args.chunk<1 or not np.isfinite([args.wall_seconds,args.max_rss_gb]).all() or min(args.wall_seconds,args.max_rss_gb)<=0:
        raise ValueError("invalid CPU resource limits")
    target = TARGETS[args.target]
    bounds = (target.bounds[0] if args.parameter_min is None else args.parameter_min,
              target.bounds[1] if args.parameter_max is None else args.parameter_max)
    output = args.output.resolve()
    if not output.is_relative_to(OWNED_OUTPUT) or output==OWNED_OUTPUT:
        raise ValueError("output must be a fresh child of the task-owned output directory")
    output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    def check():
        if time.monotonic()-start>args.wall_seconds:
            raise TimeoutError("setup wall-time cap exceeded")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.max_rss_gb*1024**2:
            raise MemoryError("setup peak-RSS cap exceeded")
    def event(kind,**fields):
        row = dict(event=kind,elapsed_seconds=time.monotonic()-start,**plain(fields))
        with (output/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        print(json.dumps(row,allow_nan=False),flush=True)
    def save(stage,payload):
        persist(output,stage,payload)
        event("stage_saved",stage=stage,operation_counts=payload.get("operation_counts",{}))
    try:
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS"):
            os.environ.setdefault(name,str(args.threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
        if os.getpriority(os.PRIO_PROCESS,0)<10:
            os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        from threadpoolctl import threadpool_limits
        seed=load_seed(args.seed,key=args.seed_key,index=args.seed_index)
        metadata=dict(args=vars(args),target=asdict(target),seed_sha256=sha256(args.seed),
                      source_commit=source_commit_from_environment(),
                      source_sha256=sha256(__file__),pinned_assets=PINNED_SHA256,prefix_ids=PREFIX_IDS,
                      torch_version=torch.__version__,numpy_version=np.__version__,pid=os.getpid(),
                      thread_environment={key:os.environ.get(key) for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
                      device="cpu",dtype="float64",resource_checks="before/after callbacks; external watchdog needed for a single long native operation")
        json_save(output/"inputs.json",metadata)
        event("setup_started",target=args.target,threads=args.threads)
        with threadpool_limits(limits=args.threads):
            check()
            model=ManualQwenSoftMap.from_pinned_files()
            check()
            event("model_loaded",peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result=prepare_fold(model,seed,target,parameter=args.parameter,bounds=bounds,chunk=args.chunk,
                                save=save,check=check,max_newton=args.max_newton,max_backtracks=args.max_backtracks)
        result["provenance"]=metadata
        persist(output,"setup",result)
        json_save(output/"status.json",dict(status="complete" if result["passed"] else "gate_failed",stage=result["stage"],
                                           elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if not result["passed"]:
            raise SystemExit(2)
    except BaseException as error:
        if not (output/"status.json").exists():
            json_save(output/"status.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=="__main__":
    main()
