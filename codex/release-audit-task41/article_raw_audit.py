"""Task41 saved numerical checks, source frozen before execution.

Only small saved JSON/NumPy records; no producer/model imports or original
trajectory, derivative, eigenspectrum, or old control reruns. The trace-plane
algorithm is our independent Task40 implementation reused on the primary arrays
for the newly changed trace claims. Dense matrices are never diagonalized.
Proposed envelope: single CPU, nice10, 4GiB AS, 180CPU/240wall seconds.
"""
from collections import deque, Counter
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

ROOT = Path('/data2/neural-shrimp')
DATA = Path('/data2/neural-shrimp-data')
BASE = DATA / 'explore/llm-soft-loop'
POD = BASE / 'pod64'
OUT = ROOT / 'codex/release-audit-task41/article-raw-evidence.json'
ARTICLE = 'b15a593676fa5ff3614b6bc1922fee514c08109b'
BASELINE = 'e300c8eda07361fa5ad7054bf540a3951226c6b4'
PINS = {}
np = None


def read(path):
    path = Path(path)
    if not (path.is_relative_to(DATA / 'explore') or
            path.is_relative_to(DATA / 'codex/llm-loop-gain-crosscheck/run-002') or
            path.is_relative_to(ROOT / 'tools') or path.is_relative_to(ROOT / 'docs/e2e')):
        raise ValueError('undeclared input root')
    before = path.stat()
    if before.st_size > 64*1024**2:
        raise ValueError('64MiB per-file cap')
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError('input changed during read')
    pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if str(path) in PINS and PINS[str(path)] != pin:
        raise ValueError('input changed between reads')
    PINS[str(path)] = pin
    return raw


def pairs(items):
    d = {}
    for k, v in items:
        if k in d:
            raise ValueError('duplicate JSON key')
        d[k] = v
    return d


def decode(raw):
    # Preserve historical nonfinite diagnostics explicitly; gate fields used.
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda s: 'NONFINITE:'+s)


def js(path):
    return decode(read(path))


def nz(path):
    with np.load(io.BytesIO(read(path)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def bounds(x):
    x = list(x)
    if not x or not all(finite(v) for v in x):
        raise ValueError('empty/nonfinite numerical collection')
    return [min(x), max(x)]


def norm_sample():
    rows, seen, duplicated = [], set(), 0
    for run in ('cells2', 'cross-a', 'cross-b'):
        rec = js(POD / f'nb/{run}/nb.json')
        for row in rec['cells']:
            key = (row['source'], round(row['log10_temperature'],9), round(row['alpha'],9))
            if key in seen:
                duplicated += 1
                continue
            seen.add(key)
            rows.append(row)
    result = {}
    for source in sorted({r['source'] for r in rows}):
        subset = [r for r in rows if r['source'] == source]
        valid = [r for r in subset if all(finite(r[k]) for k in ('readout','network','tangent_exponent'))]
        metrics = {}
        for factor in ('readout','network'):
            y = np.array([r[factor] for r in valid])
            metrics[factor] = {}
            for axis in ('alpha','log10_temperature'):
                keys = [round(r[axis],9) for r in valid]
                means = {k:float(y[np.array(keys)==k].mean()) for k in set(keys)}
                fitted = np.array([means[k] for k in keys])
                metrics[factor][axis] = float(np.var(fitted)/np.var(y))
        result[source] = dict(rows=len(subset), finite=len(valid),
            power_converged=sum(max(r['power_rel_change_J'],r['power_rel_change_DF'])<=1e-3 for r in valid),
            variance_fractions=metrics)
    return dict(total=len(rows), duplicate_rows=duplicated, maps=result,
        limits='Power-iteration estimates; reported variance describes all finite sampled estimates, not exact norms.')


def spectra():
    rank = js(DATA / 'explore/e2e/rank-spectra/spectra.json')
    rows = []
    for c in rank['cells']:
        ratios = [s['DF_singular'][1]/s['DF_singular'][0] for s in c['steps']]
        cov = [s['sigma_eigs'][1]/s['sigma_eigs'][0] for s in c['steps']]
        rows.append(dict(cell=c['cell'], states=c['states'], closure=c['closure'], dense_vs_jvp=c['dense_vs_jvp'],
            step_ratio_range=bounds(ratios), covariance_ratio_range=bounds(cov),
            composed_ratio=c['composed_singular'][1]/c['composed_singular'][0],
            periodic_candidate=c['cell'].startswith('p'),
            saved_ratios_match=all(abs(x-s['DF_ratio_2_1'])<1e-14 for x,s in zip(ratios,c['steps']))))
    spec = js(DATA / 'explore/e2e/lyap-spectrum/lyapunov-spectrum.json')
    kyr = []
    if spec['complete'] is not True or spec['args']['record'] != 600 or spec['args']['k'] != 4:
        raise ValueError('incomplete/unexpected spectrum horizon')
    for r in spec['cells']:
        v = r['exponents']
        if len(v)!=4 or not all(finite(x) for x in v) or v!=sorted(v,reverse=True) or not(v[0]>0>v[1] and v[0]+v[1]<0):
            raise ValueError('KY branch unsupported')
        kyr.append(dict(alpha=r['alpha'], log10_temperature=r['log10_temperature'], exponents=v,
                        KY=1+v[0]/abs(v[1])))
    return dict(rank_rows=rows, spectrum_args=spec['args'], KY_rows=kyr,
        positive_range=bounds(r['exponents'][0] for r in kyr),
        negative_range=bounds(v for r in kyr for v in r['exponents'][1:]),
        limits='Ratios recomputed from original stored singular values, not fresh SVD; finite-time 4-vector spectra at three cells.')


def p8():
    base = DATA / 'codex/llm-loop-gain-crosscheck/run-002'
    receipt = js(base/'receipt.json')
    artifacts = {r['path']:r for r in receipt['artifacts']}
    def pinned(rel):
        raw = read(base/rel)
        p = artifacts[rel]
        if len(raw)!=p['bytes'] or hashlib.sha256(raw).hexdigest()!=p['sha256']:
            raise ValueError('original receipt mismatch')
        return raw
    phases, returns = [], []
    for i in range(8):
        rel = f'run/{84+70*i:04d}-phase{i}-phase-summary.json'
        r = decode(pinned(rel))
        phases.append(dict(phase=i, values=r['first_two_step_singular_values'], checks=r['checks'],
                           directions=len(r['directions'])))
        rel = f'run/{575+i:04d}-monodromy-phase{i}.json'
        m = decode(pinned(rel))
        s = np.load(io.BytesIO(pinned('run/'+m['monodromy_singular_values']['path'])),allow_pickle=False)
        if s.shape!=(896,) or not np.isfinite(s).all() or (s<0).any():
            raise ValueError('stored singular spectrum invalid')
        returns.append(dict(phase=i, sigma1=float(s[0]), sigma2=float(s[1])))
    return dict(original_source=receipt['source_commit'], original_exit=receipt['exit_code'],
        original_failure_preserved=True, phases=phases, cyclic_returns=returns,
        step_sigma1_range=bounds(r['values'][0] for r in phases),
        cyclic_sigma1_range=bounds(r['sigma1'] for r in returns),
        limits='Original scientific events authenticated to failed-exit original receipt; no lifecycle replay or fresh dense SVD.')


def context():
    out={}
    for k in (1,8,16):
        r=js(POD/f'big/qwen7b-window-{k}/hidden-chaos.json')
        good=[]; failed=[]
        for i,c in enumerate(r['cells']):
            if not all(finite(v) for v in c['lyapunov']): failed.append(i); continue
            good.append(dict(index=i,temperature=c.get('log10_temperature'),period=c['period'],
                lyapunov=c['lyapunov'], chaotic=c['period']==0 and all(v>1e-3 for v in c['lyapunov'])))
        out[str(k)]=dict(args=r['args'],rows=good,failed_rows=failed,total=len(r['cells']),
                        chaotic=sum(c['chaotic'] for c in good))
    return out


def bridge():
    result={}
    for name in ('exact','m8','m64','m1'):
        base=BASE/f'bridge-{name}-t-0.115'
        m=js(base/'walk.json'); z=nz(base/'walk.npz')
        p=z['probabilities']; tokens=list(z['tokens']); a=z['alphas']
        if p.shape!=(512,96,6) or not np.isfinite(p).all(): raise ValueError('bridge shape/finite')
        it,aa,there,iss=(tokens.index(v) for v in (' it',' a',' there',' is'))
        mask=(p[:,:,iss]>.5)|((p[:,:,it]>p[:,:,aa])&(p[:,:,it]>p[:,:,there])&(p[:,:,it]>.1))
        groups=[]
        for alpha in sorted(set(a.tolist())):
            columns=np.flatnonzero(a==alpha); lens=[]; right=[]
            if len(columns)!=32: raise ValueError('bridge denominator')
            for col in columns:
                n=0
                for active in mask[:,col]:
                    if active:n+=1
                    else:
                        if n>=6:lens.append(n)
                        n=0
                if n>=6:right.append(n)
            groups.append(dict(alpha=alpha,trajectories=len(columns),phases=len(lens),
                mean=sum(lens)/len(lens) if lens else None,excluded_right_censored=right,
                occupancy=float(mask[:,columns].mean())))
        result[name]=dict(args=m['args'],groups=groups)
    return dict(rows=result, convention='Original token mask; runs >=6, retain left-censored and exclude right-censored.')


def reproducibility():
    base=POD/'repro/h2cells'; m=js(base/'repro.json'); z=nz(base/'repro.npz')
    result={}
    for label in ('chaotic','periodic','other'):
        idx=[i for i,v in enumerate(m['labels']) if v==label]
        rows=[m['comparisons']['batch64'][i] for i in idx]
        result[label]=dict(n=len(idx),ever_tokens_differ=sum(r['first_disagreement'] is not None for r in rows),
                           all_distances_finite=bool(np.isfinite(z['batch64_dist'][:,idx]).all()))
    return dict(args=m['args'],counts=result,arrays=list(z),
                limits='Mismatch count uses primary per-cell first-disagreement records; distance arrays independently finite-checked.')


def returnmaps():
    r=js(POD/'sd/returnmaps-compact-cvd-titlefix/returnmaps.json')
    selected=[]
    for a,t in ((.07228,-.14934),(.07269,-.15274)):
        c=min(r['cells'],key=lambda c:(c['alpha']-a)**2+(c['log10_temperature']-t)**2)
        selected.append(c)
    domain=[c for c in r['cells'] if c['domain'] and c['lyapunov']>r['args']['chaos_level'] and c['period']==0]
    return dict(args=r['args'],total=len(r['cells']),domain_chaotic=len(domain),selected=selected,
        under_5percent=sum(c.get('errors_over_minority',float('inf'))<.05 for c in domain),
        limits='Stored per-phase turning diagnostics; no trajectory/SVD/binning rerun and spatial label checked separately against curves.')


def cusps_basin():
    rows=[]
    rels=[f'kite/locate-la-plus-{i}' for i in (14,94,112)]+['p6s/locate-wide2-a-64','p6s/locate-wide2-b-82','p6s/locate-tu-minus-34']
    for rel in rels:
        r=js(POD/rel/'locate.json'); f=r['final']; J=np.array(f['J'])
        curve=js(DATA/'explore'/r['curve']/'curve.json')
        near=[curve['points'][i] for i in (r['left'],r['left']+1)]
        rows.append(dict(path=rel,located=r['located'],checks=r['checks'],period=r['period'],
            alpha=f['alpha'],log10_temperature=f['log10_temperature'],c=f['c'],small_b=f['f'],
            det=float(J[0,0]*J[1,1]-J[0,1]*J[1,0]),singular_values=np.linalg.svd(J,compute_uv=False).tolist(),
            critical=f['critical_multiplier'],critical_error=abs(complex(*f['critical_multiplier'])-1),
            next=f['second_modulus'],bracket_rows=near,
            final_residual_available='residual' in f,final_solver_counts_available='newton' in f))
    basin=js(BASE/'kite/basin-check/basin-check.json')
    saddle=js(POD/'kite/hook-saddle/hook-saddle.json')
    ex=js(BASE/'kite/hook-saddle-exit-v2/hook-saddle-exit.json')
    z=nz(BASE/'kite/hook-saddle-exit-v2/distance-to-saddle.npz')
    if not np.isfinite(z['distance']).all():raise ValueError('nonfinite saddle distances')
    fits={}
    for side in ('above','below'):
        use=[]
        for i,row in enumerate(ex['rows']):
            if row['side']!=side:continue
            d=z['distance'][:,i]; within=np.flatnonzero(d<1e-3)
            if len(within) and d.min()<1e-4 and within[-1]<len(d)-1:
                use.append((float(z['delta'][i]),int(within[-1])))
        coef=np.polyfit(np.log([1/x[0] for x in use]),[x[1] for x in use],1)
        fits[side]=dict(n=len(use),slope=float(coef[0]))
    mu=max(abs(complex(*v)) for v in saddle['saddle']['multipliers'])
    return dict(cusps=rows,basin=basin,saddle=saddle['saddle'],edge=saddle['edge_log10T'],
        predicted_exit_slope=6/math.log(mu),recomputed_exit_fits=fits,
        limits='Saved continuation/Newton/spectra evidence, no fresh composed derivative gate or formal manifold proof.')


def cascade():
    out={}
    for cut,folders in [('-0.148',['crescent-t-0.14798','crescent-deep2','crescent-t-0.14798-768b']),
                        ('-0.156',['crescent-t-0.156','crescent-t-0.156-192','crescent-t-0.156-384'])]:
        bands=[]
        for folder in folders:
            r=js(POD/'superstable'/folder/'superstable.json')
            bands.extend(b for b in r['bands'] if b.get('superstable') is not None and b['period']>=12 and b.get('clean_bracket'))
        bands.sort(key=lambda b:b['period'])
        ratios=[]
        for i in range(1,len(bands)-1):
            triple=bands[i-1:i+2];x=[b['superstable'] for b in triple]
            corners=[(a-b)/(b-c) for a,b,c in itertools.product(*(b['clean_bracket'] for b in triple))]
            lo,hi=min(corners),max(corners)
            ratios.append(dict(table_level=i+1,figure_level=i,ratio=(x[0]-x[1])/(x[1]-x[2]),
                bracket_range=[lo,hi],bar_omitted=cut=='-0.156' and hi-lo>1))
        out[cut]=dict(periods=[b['period'] for b in bands],bands=bands,ratios=ratios)
    flips=[]
    for name in ('crescent-t-0.156-b','crescent-t-0.156-c','crescent-t-0.156-d','crescent-t-0.156-f'):
        r=js(POD/'cascade'/name/'cascade.json')
        chosen=r['flips'][:2] if name.endswith('-b') else r['flips']
        for row in chosen:
            f=row['flip']; flips.append(dict(period=row['period'],location=f['lambda'],
                critical=f['critical_multiplier'],newton=f['newton_residuals'],ritz=f.get('ritz_residual')))
    x=[r['location'] for r in flips]; ratios=[(x[i]-x[i-1])/(x[i+1]-x[i]) for i in range(1,len(x)-1)]
    dev=[v-4.669201609102990 for v in ratios]
    out['minus0156_flips']=dict(rows=flips,ratios=ratios,deviation_factors=[dev[i]/dev[i+1] for i in range(1,len(dev)-1)])
    return out
def grid(folder):
    with np.load(io.BytesIO(read(folder / "grid.npz")), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def combine(folders, ritz=False):
    blocks = []
    for folder in folders:
        z = grid(folder)
        if ritz:
            with np.load(io.BytesIO(read(folder / "ritz.npz")), allow_pickle=False) as v:
                for k in ("alpha", "log10_temperature", "multipliers"):
                    if not np.array_equal(z[k], v[k], equal_nan=True):
                        raise ValueError(f"Ritz/grid axis or spectrum mismatch: {folder}/{k}")
                z["vectors"] = v["vectors"].copy()
        blocks.append(z)
    blocks.sort(key=lambda z: z["log10_temperature"][0])
    a = blocks[0]["alpha"]
    if any(not np.array_equal(b["alpha"], a) for b in blocks):
        raise ValueError("Different alpha axes")
    t = np.concatenate([b["log10_temperature"] for b in blocks])
    if not np.all(np.diff(t) > 0):
        raise ValueError("Duplicate/unordered temperature axis")
    names = ["ok", "mu", "mu_imag", "next_modulus", "newton", "same_tokens"]
    if ritz:
        names += ["multipliers", "vectors"]
    result = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in names}
    return dict(result, alpha=a, log10_temperature=t)


def plane(values, vectors, pair):
    x, y = pair
    if max(abs(values[x].imag), abs(values[y].imag)) > 1e-14:
        if abs(values[x] - values[y].conjugate()) > 1e-9 * max(1, abs(values[x])):
            return None
        real = np.column_stack((vectors[:, x].real, vectors[:, x].imag))
    else:
        columns = []
        for k in pair:
            v = vectors[:, k]
            v = v * np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
            columns.append(v.real)
        real = np.column_stack(columns)
    u, s, _ = np.linalg.svd(real, full_matrices=False)
    if not np.isfinite(s).all() or s[-1] <= 1e-14 * s[0]:
        return None
    return u[:, :2]


def stationary(field, valid, a, t):
    di, dj = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    x, y = di.ravel(), dj.ravel()
    design = np.column_stack((np.ones(25), x, y, x*x, x*y, y*y))
    candidates = []
    for i in range(2, len(t)-2):
        for j in range(2, len(a)-2):
            if not valid[i-2:i+3, j-2:j+3].all():
                continue
            obs = field[i-2:i+3, j-2:j+3].ravel()
            coeff, _, rank, _ = np.linalg.lstsq(design, obs, rcond=None)
            h = np.array([[2*coeff[3], coeff[4]], [coeff[4], 2*coeff[5]]])
            if rank != 6 or abs(np.linalg.det(h)) < 1e-20:
                continue
            offset = np.linalg.solve(h, -coeff[1:3])
            if np.max(abs(offset)) > .75:
                continue
            ev = np.linalg.eigvalsh(h)
            candidates.append({
                "cell": [i, j], "alpha": float(a[j] + offset[1]*(a[1]-a[0])),
                "log10T": float(t[i] + offset[0]*(t[1]-t[0])),
                "value": float(coeff[0] + coeff[1:3]@offset + .5*offset@h@offset),
                "H_cell_eigenvalues": ev.tolist(), "saddle": bool(ev[0] < 0 < ev[1]),
                "rms": float(np.sqrt(np.mean((design@coeff - obs)**2))),
                "offset_linf": float(np.max(abs(offset))),
            })
    selected = []
    for c in sorted(candidates, key=lambda c: c["offset_linf"]):
        if all(abs(c["alpha"]-p["alpha"]) > abs(a[1]-a[0]) or
               abs(c["log10T"]-p["log10T"]) > abs(t[1]-t[0]) for p in selected):
            selected.append(c)
    return {"candidates": len(candidates), "deduplicated": selected}


def edges(field, valid):
    out = []
    for i, j in np.argwhere(valid):
        for di, dj in ((1, 0), (0, 1)):
            ni, nj = i+di, j+dj
            if ni < valid.shape[0] and nj < valid.shape[1] and valid[ni, nj]:
                if np.sign(field[i, j]) != np.sign(field[ni, nj]):
                    out.append((i+di/2, j+dj/2))
    return np.array(out, dtype=float).reshape(-1, 2)


def matches(first, second):
    if len(first) == 0 or len(second) == 0:
        return 0
    return int(np.sum(np.min(np.max(abs(first[:, None] - second[None]), axis=2), axis=1) <= 1))


def delta():
    z = combine([BASE / f"pod64/dss/delta2d-d{k}" for k in range(4)], True)
    values = z["multipliers"][..., 0] + 1j*z["multipliers"][..., 1]
    vectors = z["vectors"]
    good = z["ok"] & np.isfinite(values).all(axis=-1)
    ratio = abs(values[..., 1]) / np.maximum(abs(values[..., 2]), 1e-300)
    seed = tuple(np.unravel_index(np.argmax(np.where(good, ratio, -np.inf)), good.shape))
    pairs = ((0, 1), (0, 2), (1, 2))
    bases = {}
    for i, j in np.argwhere(good):
        bases[i, j] = [(p, plane(values[i, j], vectors[i, j], p)) for p in pairs]
        bases[i, j] = [(p, q) for p, q in bases[i, j] if q is not None]
    # The declared producer traversal selects the same mathematical objective;
    # implementation and orthogonalization are independent.
    picked = {seed: (0, 1)}
    chosen = {seed: dict(bases[seed])[(0, 1)]}
    overlaps = np.full(good.shape, np.nan)
    overlaps[seed] = 1
    queue = deque([seed])
    while queue:
        i, j = queue.popleft()
        for ni, nj in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            key = ni, nj
            if key in chosen or key not in bases or not bases[key]:
                continue
            options = [(float(np.sum((chosen[i, j].T @ q)**2)/2), p, q) for p, q in bases[key]]
            overlap, pair, basis = max(options, key=lambda v: v[0])
            overlaps[key] = overlap
            chosen[key], picked[key] = basis, pair
            queue.append(key)
    tau = np.full(good.shape, np.nan)
    det = tau.copy()
    for key, pair in picked.items():
        vals = values[key][list(pair)]
        tau[key], det[key] = np.sum(vals).real, np.prod(vals).real
    tracked = np.isfinite(tau)
    real = good & (z["mu_imag"] == 0)
    real_edges = edges(z["mu"], real)
    trace_edges = edges(tau, tracked)
    delta_edges = edges(det, tracked)
    complex_cells = np.argwhere(good & (abs(z["mu_imag"]) > 0))
    measured = np.concatenate((real_edges, complex_cells))
    agreement = np.sign(tau) == np.sign(z["mu"])
    return {
        "shape": list(good.shape), "solved": int(good.sum()), "tracked": int(tracked.sum()),
        "seed": list(map(int, seed)), "overlap_below_09": int(np.sum(overlaps < .9)),
        "overlap_min": float(np.nanmin(overlaps)), "sign_agreement": int(np.sum(agreement & tracked)),
        "disagreement_cells": np.argwhere(~agreement & tracked).tolist(),
        "selected_pairs": {str(p): sum(v == p for v in picked.values()) for p in pairs},
        "real_zero_edges": len(real_edges), "trace_zero_edges": len(trace_edges),
        "delta_zero_edges": len(delta_edges), "complex_leading_cells": len(complex_cells),
        "real_near_trace": matches(real_edges, trace_edges),
        "trace_near_measured": matches(trace_edges, measured),
        "real_near_delta": matches(real_edges, delta_edges),
        "delta_near_real": matches(delta_edges, real_edges),
        "trace_range": [float(np.nanmin(tau)), float(np.nanmax(tau))],
        "delta_range": [float(np.nanmin(det)), float(np.nanmax(det))],
        "trace_stationary": stationary(tau, tracked & (overlaps >= .9), z["alpha"], z["log10_temperature"]),
        "delta_stationary": stationary(det, tracked & (overlaps >= .9), z["alpha"], z["log10_temperature"]),
        "leading_real_stationary": stationary(z["mu"], real, z["alpha"], z["log10_temperature"]),
        "newton_max": float(np.max(z["newton"][good])),
        "same_tokens_false_solved": int(np.sum(good & ~z["same_tokens"])),
        "limits": [
            "Finite mesh, selected rank-two Ritz planes, local quadratic least squares, not certified critical points.",
            "No model, original Newton, or composed derivative replay.",
            "Original grid stores no Ritz residuals; saved Ritz values/vectors are audited, not independently generated.",
            "Absence of a fitted stationary point is only for disclosed fit/filter/mesh.",
        ],
    }




def main():
    global np
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1': raise ValueError('single-thread environment required')
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('offline CPU environment required')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10 required')
    import numpy as np
    started=time.monotonic(); results={}; errors={}
    with OUT.open('x') as output:
        for name in ('norm_sample','spectra','p8','context','bridge','reproducibility','returnmaps','cusps_basin','cascade','delta'):
            try:results[name]=globals()[name]()
            except Exception as exc:errors[name]=dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        json.dump(dict(article_commit=ARTICLE,baseline_commit=BASELINE,sections=results,errors=errors,
            inputs=PINS,numpy=np.__version__,elapsed_seconds=time.monotonic()-started,
            model_calls=0,old_control_runs=0,limitations=[
            'Saved numeric summaries recomputed; original trajectories, derivative products and eigen/SVD computations are not regenerated.',
            'Producer records generally record args but lack immutable source-commit receipts; those origins are not upgraded by current source hashes.',
            'Only P8 original event records are additionally joined to their original receipt inventory in this script.',
            'Finite grids, fitted critical points and finite-time spectra remain numerical evidence, not formal global reductions.']),
            output,indent=2,allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(output=str(OUT),completed=list(results),errors=errors,elapsed=time.monotonic()-started)))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
