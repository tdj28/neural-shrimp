"""Audit saved Hénon Type A curves and distinguish orbit-sheet intersections.

Post-reproduction extension: retain every row, refine the projected crossing,
locate the loop's left turn, and resolve a cut to the separate outer arm.
All numerical results are exploratory and lack interval certification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
from scipy.optimize import root
import audit_henon_skeleton as h


def replay(row):
    xs, a, b = np.array(row['states']), row['a'], row['b']
    p = len(xs)
    start = np.array([xs[0], xs[-1]])
    state, matrix = start.copy(), np.eye(2)
    observed, first_returns = [], []
    for k in range(p):
        observed.append(state[0])
        x, y = state
        matrix = np.array([[-2*x, b], [1., 0.]]) @ matrix
        state = np.array([a-x*x+b*y, x])
        first_returns.append(float(np.linalg.norm(state-start)))
    out = dict(closure=float(np.linalg.norm(state-start)),
               state_disagreement=float(max(abs(np.array(observed)-xs))),
               trace=float(np.trace(matrix)), determinant=float(np.linalg.det(matrix)),
               radius=float(max(abs(np.linalg.eigvals(matrix)))),
               primitive_min=min(first_returns[q-1] for q in range(1, p) if p % q == 0))
    out['passed'] = bool(out['closure'] < 1e-11 and out['state_disagreement'] < 1e-11
                         and abs(out['trace']) < 1e-9 and out['primitive_min'] > 1e-7
                         and out['radius'] < 1)
    return out


def segment_intersection(p, q, r, s):
    mat = np.column_stack([q-p, r-s])
    if abs(np.linalg.det(mat)) < 1e-20:
        return None
    uv = np.linalg.solve(mat, r-p)
    return uv if np.all(uv >= 0) and np.all(uv <= 1) else None


def joint(z):
    p = (len(z)-2)//2
    return np.r_[h.constraints(np.r_[z[:p], z[-2:]]),
                 h.constraints(np.r_[z[p:2*p], z[-2:]])]


def mp_equations(v, p, b=None):
    if b is None:
        xs, a, b = list(v[:-2]), v[-2], v[-1]
    else:
        xs, a = list(v[:-1]), v[-1]
    m = mp.eye(2)
    for x in xs:
        m = mp.matrix([[-2*x, b], [1, 0]]) * m
    return [xs[(i+1)%p]-a+xs[i]**2-b*xs[(i-1)%p] for i in range(p)]+[m[0,0]+m[1,1]]


def precision_crossing(z, dps):
    with mp.workdps(dps):
        p = (len(z)-2)//2
        start = [mp.mpf(str(v)) for v in z]
        def fun(*v):
            return (mp_equations(list(v[:p])+list(v[-2:]), p)
                    +mp_equations(list(v[p:2*p])+list(v[-2:]), p))
        v = list(mp.findroot(fun, tuple(start), tol=mp.mpf(10)**(-(dps-15)), maxsteps=40))
        residual = max(abs(q) for q in fun(*v))
        distance = min(mp.sqrt(sum((v[i]-v[p+(i+k)%p])**2 for i in range(p))) for k in range(p))
        return dict(dps=dps, a=mp.nstr(v[-2], dps), b=mp.nstr(v[-1], dps),
                    orbit_a=[mp.nstr(q, dps) for q in v[:p]],
                    orbit_b=[mp.nstr(q, dps) for q in v[p:2*p]],
                    closure_and_trace_max=mp.nstr(residual, 12),
                    minimum_cyclic_orbit_distance=mp.nstr(distance, dps),
                    passed=bool(residual < mp.mpf('1e-40') and distance > mp.mpf('1e-5')))


def total_trace_gradient(z):
    p = len(z)-2
    j = h.jacobian(z)
    dx = np.linalg.solve(j[:p, :p], -j[:p, -2:])
    return j[-1, :p] @ dx + j[-1, -2:]


def tip_equations(z):
    return np.r_[h.constraints(z), total_trace_gradient(z)[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    for name in ('audit_henon_skeleton.py', 'audit_henon_skeleton_geometry.py'):
        blob = subprocess.check_output(['git','show',commit+':tools/'+name])
        if blob != Path(__file__).with_name(name).read_bytes():
            raise ValueError('commit source before this extension')
    rec = json.loads((args.input/'receipt.json').read_text())
    for name, expected in rec['files'].items():
        if hashlib.sha256((args.input/name).read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('input hash mismatch')
    curves = json.loads((args.input/'curves.json').read_text())
    checked = [[replay(r) for r in c['rows']] for c in curves]
    (args.out/'row-replay.json').write_text(json.dumps(checked, indent=2)+'\n')
    if not all(r['passed'] for c in checked for r in c):
        raise ValueError('independent orbit replay failed')
    candidates = []
    for ci, c in enumerate(curves):
        cr = c['rows']
        for cj in range(ci, len(curves)):
            rr = curves[cj]['rows']
            for i in range(len(cr)-1):
                a, b = [np.array([r['a'], r['b']]) for r in cr[i:i+2]]
                for j in range(len(rr)-1):
                    if ci == cj and abs(i-j) < 2:
                        continue
                    d, e = [np.array([r['a'], r['b']]) for r in rr[j:j+2]]
                    uv = segment_intersection(a, b, d, e)
                    if uv is None:
                        continue
                    x = np.array(cr[i]['states'])*(1-uv[0])+np.array(cr[i+1]['states'])*uv[0]
                    y = np.array(rr[j]['states'])*(1-uv[1])+np.array(rr[j+1]['states'])*uv[1]
                    if min(np.linalg.norm(x-np.roll(y, k)) for k in range(len(x))) < 1e-3:
                        continue
                    sol = root(joint, np.r_[x,y,a+(b-a)*uv[0]], tol=1e-11)
                    item = dict(curves=[ci,cj], segments=[i,j], result=sol.x.tolist(),
                                residual_max=float(max(abs(joint(sol.x)))), solver_success=bool(sol.success))
                    if item['residual_max'] < 1e-10:
                        item['precision'] = [precision_crossing(sol.x, dps) for dps in (60,90)]
                    candidates.append(item)
    (args.out/'projected-crossings.json').write_text(json.dumps(candidates, indent=2)+'\n')
    if not candidates or not all('precision' in c and all(p['passed'] for p in c['precision']) for c in candidates):
        raise ValueError('projected crossing not qualified; do not default to another point')
    # The inner component is the distinct seed component to the right of the
    # outer arm. Its leftmost saved point initializes a tangent-turn solve.
    components = sorted(set(c['seed'] for c in curves))
    if len(components) != 2:
        raise ValueError('expected two saved components for this bounded reproduction')
    groups = [[r for c in curves if c['seed']==s for r in c['rows']] for s in components]
    ordered = sorted(groups, key=lambda rs:min(r['a'] for r in rs))
    tip0 = min(ordered[1], key=lambda r:r['a'])
    sol = root(tip_equations, np.r_[tip0['states'],tip0['a'],tip0['b']], tol=1e-11)
    tip = h.gates(sol.x)
    tip.update(total_trace_gradient=total_trace_gradient(sol.x).tolist(),
               turn_residual=float(max(abs(tip_equations(sol.x)))))
    if not tip['passed'] or tip['turn_residual'] > 1e-9:
        raise ValueError('loop-turn refinement failed')
    bcut = tip['b']
    outer0 = min(ordered[0], key=lambda r:abs(r['b']-bcut))
    zouter, outer = h.fixed_b_seed(np.array(outer0['states']), outer0['a'], bcut)
    if not outer['passed']:
        raise ValueError('outer-arm cut failed')
    cuts = []
    for dps in (60,90):
        inner = h.high_precision(sol.x, dps)
        outside = h.high_precision(zouter, dps)
        with mp.workdps(dps):
            gap = mp.mpf(inner['a'])-mp.mpf(outside['a'])
            cuts.append(dict(dps=dps, inner=inner, outer=outside, gap_alpha=mp.nstr(gap,dps),
                             passed=bool(inner['passed'] and outside['passed'] and gap > 0)))
    geometry = dict(loop_turn=tip, outer_cut=outer, same_b_cuts=cuts,
                    caveat='The measured cut gap is not a global minimum separation. The projected loop crossing joins different full-state cycles.')
    (args.out/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    t=np.linspace(-.35,.35,501)
    axes[0].plot(t,t*t,label='$b=a^2$',color='#126d98')
    axes[0].plot(t*t,t,label='$a=b^2$',color='#d17515')
    axes[0].plot(0,0,'ko',ms=4)
    axes[0].set(xlim=(-.3,.3),ylim=(-.3,.3),xlabel='$a$',ylabel='$b$',title='Gallas (1994): exact crossing')
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    for ax in axes[1:]:
        for c in curves:
            ax.plot([r['a'] for r in c['rows']],[r['b'] for r in c['rows']],color='#126d98',lw=1.5)
        ax.set(xlabel='$a$',ylabel='$b$')
    crossing=candidates[0]['precision'][-1]
    ca,cb=float(crossing['a']),float(crossing['b'])
    axes[1].set(xlim=(1.43,1.6),ylim=(.12,.22),title='Façanha et al. (2013): Type A')
    axes[1].plot(ca,cb,'o',color='#d17515',ms=5)
    axes[1].annotate('Different coexisting cycles',xy=(ca,cb),xytext=(1.475,.192),fontsize=8,
                     arrowprops=dict(arrowstyle='->',color='#666'))
    axes[2].set(xlim=(1.435,1.495),ylim=(.156,.18),title='The same Type A head, enlarged')
    axes[2].plot([outer['a'],tip['a']],[bcut,bcut],'o-',color='#d17515',ms=4,lw=1)
    axes[2].text(.04,.05,'Two separated branches\nCut gap in $a$: '+f"{float(cuts[-1]['gap_alpha']):.6f}",
                 transform=axes[2].transAxes,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.14)
    fig.suptitle('The historical examples survive closer inspection',fontsize=15)
    fig.savefig(args.out/'historical-shrimp-audit.png',dpi=180)
    plt.close(fig)
    files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
           for p in args.out.iterdir() if p.is_file()}
    report=dict(status='exploratory post-reproduction audit',source_commit=commit,
                source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('audit_henon_skeleton.py','audit_henon_skeleton_geometry.py')},
                input_receipt=str(args.input/'receipt.json'),
                input_receipt_sha256=hashlib.sha256((args.input/'receipt.json').read_bytes()).hexdigest(),
                finished_utc=datetime.now(timezone.utc).isoformat(),
                rows_replayed=sum(map(len,checked)),all_replay_pass=True,
                max_replayed_closure=max(r['closure'] for c in checked for r in c),
                projected_crossings=len(candidates),
                distinct_orbit_distance=crossing['minimum_cyclic_orbit_distance'],
                projected_crossing_a=crossing['a'],projected_crossing_b=crossing['b'],
                loop_tip_a=tip['a'],loop_tip_b=bcut,gap_on_tip_horizontal_cut=cuts[-1]['gap_alpha'],
                all_geometry_gates_pass=all(c['passed'] for c in cuts),files=files,
                conclusion='The reproduced period-5 Hénon curves agree with Type A: a projected loop and a separate outer arm. Its projected crossing involves distinct cycles. This supplies no evidence of historical misclassification; the canonical Gallas scalar crossing is algebraically exact.')
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
