"""Exploratory reproduction of the period-five Hénon skeleton in Façanha et al. (2013).

Multiple shooting, analytic polynomial derivatives, and pseudo-arclength
continuation of full-cycle closure plus tr(M)=0. No pixel classification.
Raw failures and every continued row are retained. This is a numerical
reproduction, not an interval proof or a classification of the network.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import root


def monodromy(xs, b):
    m = np.eye(2, dtype=np.result_type(xs, b))
    for x in xs:
        m = np.array([[-2*x, b], [1, 0]]) @ m
    return m


def constraints(z):
    xs, a, b = z[:-2], z[-2], z[-1]
    return np.r_[np.roll(xs, -1) - a + xs*xs - b*np.roll(xs, 1),
                 np.trace(monodromy(xs, b))]


def jacobian(z):
    xs, b = z[:-2], z[-1]
    p = len(xs)
    j = np.zeros((p+1, p+2))
    for i in range(p):
        j[i, i] += 2*xs[i]
        j[i, (i+1) % p] += 1
        j[i, (i-1) % p] -= b
        j[i, -2] = -1
        j[i, -1] = -xs[(i-1) % p]
    # Product rule, including each state's derivative and b in every factor.
    mats = [np.array([[-2*x, b], [1., 0.]]) for x in xs]
    prefix = [np.eye(2)]
    for m in mats:
        prefix.append(m @ prefix[-1])
    suffix = [None] * (p+1)
    suffix[p] = np.eye(2)
    for i in reversed(range(p)):
        suffix[i] = suffix[i+1] @ mats[i]
    for i in range(p):
        j[-1, i] = np.trace(suffix[i+1] @ np.array([[-2., 0.], [0., 0.]]) @ prefix[i])
        j[-1, -1] += np.trace(suffix[i+1] @ np.array([[0., 1.], [0., 0.]]) @ prefix[i])
    return j


def gates(z):
    xs, a, b = z[:-2], float(z[-2]), float(z[-1])
    p = len(xs)
    f = constraints(z)
    m = monodromy(xs, b)
    eig = np.linalg.eigvals(m)
    expected = (-b)**p
    det_error = abs(np.linalg.det(m)-expected)/max(abs(expected), 1e-30)
    lower = {str(q): float(np.max(np.abs(xs-np.roll(xs, q))))
             for q in range(1, p) if p % q == 0}
    cyclic_trace = [float(np.trace(monodromy(np.roll(xs, i), b))) for i in range(p)]
    row = dict(a=a, b=b, states=xs.tolist(), closure_max=float(max(abs(f[:-1]))),
               trace=float(f[-1]), determinant=float(np.linalg.det(m)),
               determinant_exact=expected, determinant_relative_error=float(det_error),
               multipliers=[[float(v.real), float(v.imag)] for v in eig],
               spectral_radius=float(max(abs(eig))), primitive_distances=lower,
               cyclic_trace_max=float(max(abs(np.array(cyclic_trace)))),
               monodromy=m.tolist())
    row['passed'] = bool(np.isfinite(z).all() and row['closure_max'] < 2e-11
                         and row['cyclic_trace_max'] < 2e-9 and det_error < 1e-7
                         and min(lower.values(), default=1.) > 1e-7
                         and row['spectral_radius'] < 1)
    return row


def fixed_b_seed(xs, a, b):
    z0 = np.r_[xs, a, b]
    sol = root(lambda z: np.r_[constraints(z), z[-1]-b], z0,
               jac=lambda z: np.vstack([jacobian(z), np.eye(len(z))[-1]]), tol=1e-11)
    row = gates(sol.x)
    row.update(solver_success=bool(sol.success), message=str(sol.message),
               nfev=int(sol.nfev), fixed_b_error=float(abs(sol.x[-1]-b)))
    row['passed'] &= row['fixed_b_error'] < 1e-12
    return sol.x, row


def tangent(z, scale, previous=None):
    _, _, vh = np.linalg.svd(jacobian(z)*scale[None, :], full_matrices=True)
    t = vh[-1]
    if previous is not None and t @ previous < 0:
        t = -t
    return t


def trace(seed, direction, max_steps, ds):
    scale = np.r_[np.ones(len(seed)-2), .1, .1]
    z = seed.copy()
    t = tangent(z, scale)*direction
    rows, failures = [gates(z)], []
    start = z.copy()
    reason = 'step_limit'
    for index in range(max_steps):
        q = z/scale
        prediction = q+ds*t
        sol = root(lambda v: np.r_[constraints(v*scale), (v-prediction)@t], prediction,
                   jac=lambda v: np.vstack([jacobian(v*scale)*scale[None, :], t]), tol=1e-11)
        row = gates(sol.x*scale)
        correction = float(np.linalg.norm(sol.x-prediction))
        row.update(step=index+1, step_size=ds, correction=correction,
                   nfev=int(sol.nfev), solver_success=bool(sol.success))
        if not row['passed'] or correction > .6*ds:
            failures.append(row)
            ds *= .5
            if ds < 1e-6:
                reason = 'minimum_step_reached'
                break
            continue
        z = sol.x*scale
        rows.append(row)
        t = tangent(z, scale, t)
        if not (1.40 <= z[-2] <= 1.65 and .10 <= z[-1] <= .24):
            reason = 'left_expanded_figure_window'
            break
        if index > 50 and np.linalg.norm((z-start)/scale) < ds*.8:
            reason = 'returned_near_seed_in_full_state'
            break
        if correction < .05*ds:
            ds = min(ds*1.15, .02)
    return dict(rows=rows, failed_correctors=failures, stop_reason=reason)


def distance_to_curves(z, curves):
    best = np.inf
    for curve in curves:
        for row in curve['rows']:
            if abs(row['a']-z[-2]) > .004 or abs(row['b']-z[-1]) > .004:
                continue
            p = len(z)-2
            delta = min(np.linalg.norm(np.roll(z[:-2], k)-row['states']) for k in range(p))
            best = min(best, np.hypot(delta, np.linalg.norm((z[-2:]-[row['a'], row['b']])/.1)))
    return best


def high_precision(z, dps):
    import mpmath as mp
    with mp.workdps(dps):
        p = len(z)-2
        zz = [mp.mpf(str(v)) for v in z]
        b = zz[-1]
        def residual(*v):
            xs, a = v[:-1], v[-1]
            m = mp.eye(2)
            for x in xs:
                m = mp.matrix([[-2*x, b], [1, 0]])*m
            return [xs[(i+1)%p]-a+xs[i]**2-b*xs[(i-1)%p] for i in range(p)]+[m[0,0]+m[1,1]]
        v = list(mp.findroot(residual, tuple(zz[:-1]), tol=mp.mpf(10)**(-(dps-15)), maxsteps=30))
        residual_max = max(abs(q) for q in residual(*v))
        m = mp.eye(2)
        for x in list(v)[:-1]:
            m = mp.matrix([[-2*x, b], [1, 0]])*m
        D = mp.det(m)
        serialized = [mp.mpf(mp.nstr(q, dps)) for q in v]
        serialized_residual = max(abs(q) for q in residual(*serialized))
        return dict(dps=dps, states=[mp.nstr(q, dps) for q in v[:-1]],
                    a=mp.nstr(v[-1], dps), b=mp.nstr(b, dps),
                    residual_max=mp.nstr(residual_max, 12),
                    determinant=mp.nstr(D, dps),
                    determinant_identity_error=mp.nstr(abs(D-(-b)**p), 12),
                    serialized_residual_max=mp.nstr(serialized_residual, 12),
                    radius_at_trace_zero=mp.nstr(mp.sqrt(abs(D)), dps),
                    change_from_float64=float(max(abs(v[i]-zz[i]) for i in range(p+1))),
                    passed=bool(residual_max < mp.mpf('1e-40') and serialized_residual < mp.mpf('1e-40')
                                and abs(D-(-b)**p) < mp.mpf('1e-40')))


def plot(curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout='constrained')
    for i, curve in enumerate(curves):
        rows = curve['rows']
        ax.plot([r['a'] for r in rows], [r['b'] for r in rows], color='#2463a2', lw=1.4)
    ax.set(xlim=(1.43, 1.60), ylim=(.12, .22), xlabel='$a$', ylabel='$b$',
           title='Original Hénon example: continued period-5 trace-zero curves')
    ax.text(.02, .02, 'Every point: full periodic orbit + trace zero\nAnalytic determinant: $D=(-b)^5<0$; neither multiplier is zero',
            transform=ax.transAxes, fontsize=9, bbox=dict(fc='white', ec='none', alpha=.9))
    ax.grid(alpha=.15)
    fig.savefig(out/'henon-period5-skeleton.png', dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-steps', type=int, default=2500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = Path(__file__).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parents[1], text=True).strip()
    blob = subprocess.check_output(['git', 'show', commit+':tools/audit_henon_skeleton.py'], cwd=source.parents[1])
    if blob != source.read_bytes():
        raise ValueError('commit source before this exploratory run')
    attempts, seeds, curves = [], [], []
    p = 5
    for b in (.14, .15, .16, .165, .17, .18, .20):
        for a in (1.46, 1.48, 1.50, 1.52, 1.54):
            x, y = 0., 0.
            history = []
            for i in range(2500):
                x, y = a-x*x+b*y, x
                if not np.isfinite(x) or abs(x) > 1e6:
                    break
                if i >= 2500-p:
                    history.append(x)
            if len(history) != p:
                attempts.append(dict(a=a, b=b, passed=False, reason='seed iteration escaped'))
                continue
            with np.errstate(over='ignore', invalid='ignore'):
                try:
                    z, row = fixed_b_seed(np.array(history), a, b)
                except (ValueError, np.linalg.LinAlgError) as error:
                    attempts.append(dict(a=a, b=b, passed=False, reason=str(error)))
                    continue
            row['initial_a'], row['initial_b'] = a, b
            attempts.append(row)
            if row['passed'] and 1.4 < z[-2] < 1.65 and distance_to_curves(z, curves) > .035:
                seeds.append(z)
                for direction in (-1, 1):
                    curve = trace(z, direction, args.max_steps, .01)
                    curve.update(seed=len(seeds)-1, direction=direction)
                    curves.append(curve)
                (args.out/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')
                print(json.dumps(dict(seed=len(seeds), a=float(z[-2]), b=float(z[-1]),
                                      curves=len(curves))), flush=True)
            (args.out/'seed-attempts.json').write_text(json.dumps(attempts, indent=2)+'\n')
    if not curves:
        raise ValueError('no admissible period-five seed found')
    precision = []
    for z in seeds:
        precision.append([high_precision(z, 60), high_precision(z, 90)])
    (args.out/'precision.json').write_text(json.dumps(precision, indent=2)+'\n')
    plot(curves, args.out)
    files = {f.name:dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
             for f in args.out.iterdir() if f.is_file()}
    rows = [r for c in curves for r in c['rows']]
    receipt = dict(status='exploratory', source_commit=commit, source_sha256=hashlib.sha256(blob).hexdigest(),
                   finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
                   map='x_next=a-x*x+b*y; y_next=x', period=p, threads={k:os.environ.get(k) for k in
                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                   source_reference='10.1016/j.physleta.2013.03.025; Fig.3(c), Fig.4, pp.1266-1267',
                   seed_attempts=len(attempts), accepted_seeds=len(seeds), curves=len(curves), points=len(rows),
                   all_rows_pass=all(r['passed'] for r in rows),
                   max_closure=max(r['closure_max'] for r in rows),
                   max_cyclic_trace=max(r['cyclic_trace_max'] for r in rows),
                   max_det_relative_error=max(r['determinant_relative_error'] for r in rows),
                   precision_pass=all(r['passed'] for pair in precision for r in pair), files=files,
                   limits='Finite curve portions; visual review and orbit-sheet crossing audit required. No interval certification.')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not receipt['all_rows_pass'] or not receipt['precision_pass']:
        raise ValueError('a final gate failed; retain raw output')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
