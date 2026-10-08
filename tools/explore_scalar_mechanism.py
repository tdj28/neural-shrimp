"""Bounded local scalar-factorization mechanism pilot; exploratory, no held-out seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp.criticality import finite_difference_forms, ns_coefficient
from neural_shrimp.minimal import follow_orbit, iterate
from neural_shrimp.orbits import certify_orbit
from neural_shrimp.scalar_mechanism import ScalarFactorization, planar_stability


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def run_case(model, ratio, out, start_steps, probe_steps, tail_steps):
    flip = model.flip_rate(ratio)
    eta0 = 1.02*flip
    step = lambda x: model.step(x, eta0, ratio)
    values, vectors = np.linalg.eig(np.diag([1.0, ratio])@model.hessian(model.minimum))
    direction = vectors[:, np.argmax(values.real)].real
    initial = model.minimum + 1e-3*direction/np.linalg.norm(direction)
    x = iterate(step, initial, start_steps)
    result = {'target': model.target, 'ridge': model.ridge, 'ratio': ratio,
              'eta_flip': flip, 'eta_start': eta0, 'initial_state': initial.tolist(),
              'evidence_rung': 0, 'label': 'exploratory'}
    if x is None:
        return {**result, 'outcome': 'start_diverged'}
    family = lambda s: (lambda p: model.step(p, eta0*10**s, ratio),
                        lambda p: model.jacobian(p, eta0*10**s, ratio))
    follow = follow_orbit(family, x, 2, s_max=0.7, h0=0.002, max_steps=1500)
    result.update(outcome=follow['type'], boundary_s=follow['s_boundary'], bracket=follow['bracket'])
    path = follow['path']
    np.savez(out / 'continuation.npz', s=np.array([p[0] for p in path]),
             states=np.array([p[2] for p in path]), multipliers=np.array([p[1] for p in path]))
    result['path_points'] = len(path)
    if not path:
        return result
    result['eta_boundary'] = None if follow['s_boundary'] is None else eta0*10**follow['s_boundary']
    result['last_multipliers'] = [[m.real, m.imag] for m in path[-1][1]]
    discs, commutators = [], []
    for s, _, p in path:
        eta = eta0*10**s
        y = model.step(p, eta, ratio)
        A = model.jacobian(y, eta, ratio)@model.jacobian(p, eta, ratio)
        discs.append(planar_stability(A)['discriminant'])
        scale = np.sqrt(eta*np.array([1, ratio]))
        H0, H1 = (scale[:, None]*model.hessian(z)*scale for z in (p, y))
        commutators.append(np.linalg.norm(H0@H1-H1@H0))
    result['minimum_discriminant'] = min(discs)
    result['max_commutator_norm'] = max(commutators)
    if follow['type'] != 'neimark_sacker':
        return result
    def equations(z):
        p, eta = z[:2], z[2]
        A = model.two_step_forms(p, eta, ratio)[0]
        return np.r_[model.step(model.step(p, eta, ratio), eta, ratio)-p, np.linalg.det(A)-1]
    refined = root(equations, np.r_[path[-1][2], result['eta_boundary']], tol=1e-11)
    p, eta = refined.x[:2], refined.x[2]
    residual = float(np.max(np.abs(equations(refined.x))))
    result['ns_root'] = {'solver_success': bool(refined.success), 'residual': residual,
                         'eta': eta, 'point': p.tolist()}
    if residual > 1e-9 or eta <= 0:
        result['outcome'] = 'ns_refinement_unresolved'
        return result
    A, B, C = model.two_step_forms(p, eta, ratio)
    result['ns_root'].update(planar_stability(A))
    result['ns_root'].pop('attracting')  # Roundoff at the crossing does not decide stability.
    scale = np.sqrt(eta*np.array([1.0, ratio]))
    S0, S1 = [np.eye(2)-scale[:, None]*model.hessian(z)*scale
              for z in (p, model.step(p, eta, ratio))]
    result['ns_root']['symmetric_step_eigenvalues'] = [np.linalg.eigvalsh(S).tolist() for S in (S0, S1)]
    result['ns_root']['symmetric_step_commutator_norm'] = float(np.linalg.norm(S0@S1-S1@S0))
    second = model.step(p, eta, ratio)
    result['ns_root']['cycle_separation'] = float(np.linalg.norm(second-p))
    result['ns_root']['mean_difference_residual'] = float(np.max(np.abs(model.cycle_equations((p+second)/2, (p-second)/2, eta, ratio))))
    coefficient = ns_coefficient(A, B, C)
    result['coefficient_analytic'] = clean(coefficient)
    def jvp(points, vectors):
        return np.array([model.two_step_forms(z, eta, ratio)[0]@v for z, v in zip(points, vectors)])
    result['coefficient_finite_difference'] = []
    for factor in (1.0, 0.5):
        Bfd, Cfd = finite_difference_forms(jvp, p, step_b=1e-4*factor, step_c=1e-3*factor)
        result['coefficient_finite_difference'].append(clean(ns_coefficient(A, Bfd, Cfd)))
    # Continue the same orbit at symmetric small detunings before estimating crossing slope.
    result['transversality'] = []
    for h in (1e-5, 5e-6):
        mods = []
        for sign in (-1, 1):
            rate = eta*(1+sign*h)
            cert = certify_orbit(lambda z: model.step(z, rate, ratio),
                                 lambda z: model.jacobian(z, rate, ratio), p, 2, candidate_tolerance=0.05)
            if not cert['verified']:
                raise RuntimeError('transversality continuation failed')
            z = np.asarray(cert['orbit'][0])
            mods.append(np.sqrt(np.linalg.det(model.two_step_forms(z, rate, ratio)[0])))
        result['transversality'].append({'relative_h': h, 'd_modulus_d_eta': (mods[1]-mods[0])/(2*eta*h)})
    # Dynamic amplitude/rotation probe only for the previously known ridge=.2, ratio=sqrt(8) case.
    if abs(model.ridge-0.2) < 1e-12 and abs(ratio-np.sqrt(8)) < 1e-12:
        result['circle_probes'] = []
        for delta in (0.0001, 0.0004, 0.0016):
            rate = eta*(1+delta)
            step = lambda z: model.step(z, rate, ratio)
            z = iterate(step, p+np.array([1e-4, -1e-4]), probe_steps)
            if z is None:
                result['circle_probes'].append({'delta': delta, 'outcome': 'diverged'})
                continue
            tail = []
            for _ in range(tail_steps):
                z = step(step(z))
                tail.append(z.copy())
            tail = np.array(tail)
            np.save(out / f'tail-{delta:.4f}.npy', tail)
            centered = tail-tail.mean(0)
            angle = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
            result['circle_probes'].append({'delta': delta, 'outcome': 'bounded',
                'rms_amplitude': float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
                'rotation': float(abs(angle[-1]-angle[0])/(2*np.pi*(len(angle)-1))),
                'max_distance_from_parent': float(np.max(np.linalg.norm(tail-p, axis=1)))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['unit', 'pilot'], default='pilot')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [(lam, ratio) for lam in (0.05, 0.1, 0.2, 0.3) for ratio in (1.0, 2.0, np.sqrt(8), 4.0)]
    cases.append((0.2, 1/np.sqrt(8)))
    if args.profile == 'unit':
        cases = [(0.2, np.sqrt(8))]
    plan = {'target': 1.2, 'cases': cases, 'start_steps': 10000, 'probe_steps': 60000,
            'tail_steps': 8192, 'profile': args.profile, 'exploratory': True,
            'max_s': 0.7, 'max_continuation_steps': 1500,
            'selection': 'ridge and ratio grid; dynamic probes only on previously known NS case'}
    write(args.output/'plan.json', clean(plan))
    metadata = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                'plan_sha256': hashlib.sha256((args.output/'plan.json').read_bytes()).hexdigest()}
    write(args.output/'metadata.json', metadata)
    started = time.monotonic()
    results = []
    for index, (lam, ratio) in enumerate(cases):
        directory = args.output/f'case-{index:02d}'
        directory.mkdir()
        start = time.monotonic()
        try:
            result = run_case(ScalarFactorization(ridge=lam), ratio, directory,
                              plan['start_steps'], plan['probe_steps'], plan['tail_steps'])
        except Exception as exc:
            result = {'ridge': lam, 'ratio': ratio, 'outcome': 'failed', 'error': repr(exc)}
        result['elapsed_seconds'] = time.monotonic()-start
        result = clean(result)
        write(directory/'result.json', result)
        results.append(result)
        print(json.dumps({'case': index, 'ridge': lam, 'ratio': ratio, 'outcome': result['outcome'],
                          'seconds': result['elapsed_seconds']}), flush=True)
    write(args.output/'results.json', {'results': results, 'elapsed_seconds': time.monotonic()-started,
                                      'evidence_rung': 0, 'label': 'exploratory'})
    files = [{'path': str(p.relative_to(args.output)), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(args.output.rglob('*')) if p.is_file()]
    write(args.output/'manifest.json', {'metadata': metadata, 'files': files})
    return int(any(row['outcome']=='failed' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
