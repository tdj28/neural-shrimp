"""One exploratory local fold test; floating-point evidence, never a proof.

Select the lowest-index saved +1 ray of S2-28011-zoom:587 before new outcomes.
Follow 24 fixed corrector steps, then solve R=0, (M-I)v=0, ||v||=1.
Finite-difference convergence of transversality/nonlinearity is reported.
No holdouts, random targets, parameter search, or interval claim.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp.gdmaps import GridMap, TwoLayerNet
from neural_shrimp.orbits import certify_orbit
DATA = Path('/data2/neural-shrimp-data')
SOURCE_FILES = ['tools/explore_window_fold.py', 'src/neural_shrimp/gdmaps.py', 'src/neural_shrimp/orbits.py']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def fold_diagnostics(evaluate, x, s):
    residual, monodromy, _ = evaluate(x, s)
    n = len(x)
    a_matrix = monodromy - np.eye(n)
    u, singular, vh = np.linalg.svd(a_matrix)
    v, w = vh[-1].copy(), u[:, -1].copy()
    if v[np.argmax(abs(v))] < 0:
        v = -v
    overlap = float(w @ v)
    eigenvalues = np.linalg.eigvals(monodromy)
    selected = int(np.argmin(abs(eigenvalues - 1)))
    other = np.delete(eigenvalues, selected)
    basic = {'state': x.tolist(), 's': float(s), 'residual_inf': float(np.max(abs(residual))),
             'singular_values': singular.tolist(), 'right_null': v.tolist(),
             'left_unit_null': w.tolist(), 'unit_left_right_overlap': overlap,
             'multipliers': [[float(z.real), float(z.imag)] for z in eigenvalues],
             'unit_multiplier_error': float(abs(eigenvalues[selected] - 1)),
             'next_unit_distance': float(np.min(abs(other - 1))),
             'other_radius': float(np.max(abs(other))),
             'nondegenerate_fold_numerically_supported': False}
    if abs(overlap) < 1e-6:
        return {**basic, 'reason': 'near_orthogonal_left_right_null_vectors'}
    w /= overlap
    rows = []
    for exponent in (14, 16, 18, 20, 22):
        dx = 2.0**(-exponent) * max(1., np.max(abs(x)))
        ds = 2.0**(-exponent) * max(1., abs(s))
        plus, _, _ = evaluate(x, s + ds)
        minus, _, _ = evaluate(x, s - ds)
        parameter = (plus - minus) / (2 * ds)
        _, mp, _ = evaluate(x + dx * v, s)
        _, mm, _ = evaluate(x - dx * v, s)
        quadratic = ((mp - mm) @ v) / (2 * dx)
        a, c = float(w @ parameter), float(w @ quadratic)
        rows.append({'exponent': exponent, 'dx': float(dx), 'ds': float(ds), 'a': a, 'c': c,
                     's_second': -c / a if a != 0 else None})
    def converged(name):
        p, q = rows[-2][name], rows[-1][name]
        return p * q > 0 and abs(q) > 1e-7 and abs(p-q) / max(abs(p), abs(q)) < 1e-3
    gates = {'closure': basic['residual_inf'] / max(1., np.max(abs(x))) < 1e-8,
             'one_dimensional_kernel': singular[-1] < 1e-7 and singular[-2] > 1e-5,
             'simple_unit_mode': basic['unit_multiplier_error'] < 1e-6 and basic['next_unit_distance'] > 1e-4,
             'a_nonzero_converged': bool(converged('a')), 'c_nonzero_converged': bool(converged('c'))}
    return {**basic, 'left_null_normalized': w.tolist(), 'finite_difference_series': rows,
            'gates': {k: bool(v) for k,v in gates.items()},
            'nondegenerate_fold_numerically_supported': bool(all(gates.values())),
            'reason': 'diagnostics_complete'}

def self_test():
    def fold(x, s):
        return np.array([s+x[0]**2, -.5*x[1]]), np.diag([1+2*x[0], .5]), [x.copy()]
    good = fold_diagnostics(fold, np.zeros(2), 0.)
    assert good['nondegenerate_fold_numerically_supported']
    assert abs(good['finite_difference_series'][-1]['a']-1) < 1e-10
    assert abs(good['finite_difference_series'][-1]['c']-2) < 1e-8
    def degenerate(x, s):
        return np.array([s+x[0]**3, -.5*x[1]]), np.diag([1+3*x[0]**2, .5]), [x.copy()]
    assert not fold_diagnostics(degenerate, np.zeros(2), 0.)['nondegenerate_fold_numerically_supported']
    def defective(x, s):
        return np.array([x[1]+s, 0.]), np.array([[1.,1.],[0.,1.]]), [x.copy()]
    assert not fold_diagnostics(defective, np.zeros(2), 0.)['nondegenerate_fold_numerically_supported']
    def wrong(x,s):
        return np.array([-2*x[0]+s,-.5*x[1]]), np.diag([-1.,.5]), [x.copy()]
    assert not fold_diagnostics(wrong, np.zeros(2), 0.)['nondegenerate_fold_numerically_supported']
    print('Four synthetic fold/non-fold controls passed.')

def run(out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    runtime = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'source_sha256': {p:digest(ROOT/p) for p in SOURCE_FILES},
               'pid': os.getpid(), 'start_utc': datetime.now(timezone.utc).isoformat(),
               'cpu_threads': 1, 'nice': os.getpriority(os.PRIO_PROCESS,0), 'internal_seconds': 100,
               'map_jacobian_pair_cap': 40000, 'evidence_rung': 0, 'exploratory': True}
    write(out/'runtime-start.json', runtime)
    count = 0
    evaluations = 0
    report = {'accepted': False}
    try:
        input_path = DATA/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
        cert_path = DATA/'raw/EXP-028/attempt-002/stage3/certificates.json'
        rays_path = DATA/'explore/a2-survey/rays.json'
        expected = [(input_path,'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81'),
                    (cert_path,'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8'),
                    (rays_path,'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4')]
        for p, h in expected:
            if digest(p) != h:
                raise ValueError(f'changed historical input: {p.name}')
        inputs = json.loads(input_path.read_text())
        record = json.loads(cert_path.read_text())[111]
        window = json.loads(rays_path.read_text())['S2-28011-zoom:587']
        assert record['map_id'] == 'S2-28011-zoom' and record['component'] == 587 and record['period'] == 18
        index, ray = next((i,r) for i,r in enumerate(window['rays']) if r['type']=='fold')
        assert index == 0 and window['period'] == 18
        spec = inputs['network'].copy()
        for k in ('in_scale','out_scale','loss_scale','ridge'):
            spec[k] = float(Fraction(spec[k]))
        net = TwoLayerNet(**spec)
        xx = np.array([[float(Fraction(v)) for v in row] for row in inputs['x']])
        yy = np.array([float(Fraction(v)) for v in inputs['y']])
        center = np.array(window['center'])
        direction = np.array(window['scale']) * np.array(ray['direction'])
        theta = np.array(record['certificate']['orbit'][0])
        assert net.n_parameters == len(theta) == 13
        def family(s):
            p = center + float(s)*direction
            rh, ro = 10.0**float(p[0]), 10.0**float(p[1])
            mf, jf = GridMap(net,xx,yy,[rh],[ro]).cell_map(0)
            def tick():
                nonlocal count
                count += 1
                if count > 80000 or time.monotonic()-start > 100:
                    raise TimeoutError('bounded map/Jacobian or wall limit reached')
            def f(t):
                tick()
                return mf(t)
            def j(t):
                tick()
                return jf(t)
            return f,j
        def evaluate(t,s):
            f,j = family(s)
            z = t.copy()
            m = np.eye(13)
            orbit = [z.copy()]
            for _ in range(18):
                m = j(z) @ m
                z = f(z)
                orbit.append(z.copy())
            return z-t,m,orbit
        write(out/'inputs.json', {'historical_sources': [{'path':str(p),'sha256':h} for p,h in expected],
                                  'window':window['key'], 'row_index':111, 'ray_index':index,
                                  'center':center.tolist(), 'direction':direction.tolist(), 'ray':ray,
                                  'network':inputs['network'], 'theta0':theta.tolist(),
                                  'selection':'lowest-index historical +1 ray of preselected anatomy candidate'})
        path = []
        write(out/'path.json', path)
        for s in np.linspace(0., .98*ray['s_boundary'], 25):
            f,j = family(s)
            c = certify_orbit(f,j,theta,18,newton_tolerance=1e-9,candidate_tolerance=.05)
            path.append({'s':float(s),'certificate':c})
            write(out/'path.json',path)
            if not c['verified'] or not c['attracting']:
                raise ValueError('fixed corrector path failed; no alternate path attempted')
            theta = np.array(c['orbit'][0])
        s0 = float(path[-1]['s'])
        _,m,_ = evaluate(theta,s0)
        _,_,vh = np.linalg.svd(m-np.eye(13))
        v0 = vh[-1]
        with (out/'augmented-evaluations.jsonl').open('w') as handle:
            def augmented(z):
                nonlocal evaluations
                x,s,v = z[:13],z[13],z[14:]
                residual,m,_ = evaluate(x,s)
                ans = np.r_[residual,(m-np.eye(13))@v,(v@v-1)/2]
                evaluations += 1
                handle.write(json.dumps({'z':z.tolist(),'residual':ans.tolist()},allow_nan=False)+'\n')
                handle.flush()
                return ans
            solved = root(augmented,np.r_[theta,s0,v0],method='hybr',options={'xtol':1e-9,'maxfev':600})
            final_residual = augmented(solved.x)
        x,s = solved.x[:13],float(solved.x[13])
        diag = fold_diagnostics(evaluate,x,s)
        _,_,orbit = evaluate(x,s)
        phase1 = fold_diagnostics(evaluate,orbit[1],s)
        scale = max(1.,np.max(abs(x)))
        divisors = {str(d):float(np.max(abs(orbit[d]-x))/scale) for d in (1,2,3,6,9)}
        local = abs(s-ray['s_boundary']) < .01 and np.max(abs(x-theta))/scale < .05
        primitive = all(q>1e-6 for q in divisors.values())
        report = {'schema':'neural-shrimp-window-fold-exploration-v1', 'evidence_rung':0,
                  'research_status':'exploratory numerical local fold test; no interval proof',
                  'target':'S2-28011-zoom:587', 'period':18, 'ray_index':index,
                  'solver':{'success':bool(solved.success),'message':str(solved.message),'nfev':int(solved.nfev),
                            'augmented_residual_inf':float(np.max(abs(final_residual)))},
                  'diagnostics':diag,'next_phase_diagnostics':phase1,'proper_divisor_separations':divisors,'local_to_selected_ray':bool(local),
                  'parameter_log10':(center+s*direction).tolist(),
                  'accepted':bool(diag['nondegenerate_fold_numerically_supported']
                                  and phase1['nondegenerate_fold_numerically_supported'] and primitive and local
                                  and np.max(abs(final_residual))<1e-8),
                  'not_established':['interval bifurcation proof','continuous branch enclosure','cusp','whole shrimp','ancestry','utility']}
    except Exception as error:
        report.update({'error':type(error).__name__+': '+str(error)})
    finally:
        write(out/'outcome.json',report)
        runtime.update({'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'map_or_jacobian_calls':count,'augmented_evaluations':evaluations})
        write(out/'runtime-end.json',runtime)
        inventory = {p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        write(out/'manifest.json',inventory)
    print(json.dumps({'out':str(out),'accepted':report['accepted'],'elapsed_seconds':runtime['elapsed_seconds'],
                      'error':report.get('error'),'manifest_sha256':digest(out/'manifest.json')}))
    return 0 if report['accepted'] else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.out:
        sys.exit(run(args.out))
    else:
        parser.error('specify --self-test or --out')
