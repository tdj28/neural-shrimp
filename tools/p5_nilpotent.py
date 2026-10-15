"""Referee check (copied into tools/ on 15 Oct 2026 with its log, docs/review/p5-nilpotent.jsonl) (CPU, float64): does the leading 2x2 block of the period-5 orbit's monodromy become nilpotent
(T = mu1 + mu2 = 0 and D = mu1*mu2 = 0) inside the period-5 window of the two-block plane, as the sign pattern of
T and D on the 31x31 DSS5 grid suggests? Same map and settings as tools/doubly_superstable.py grid (DSS5):
Qwen2.5-0.5B-Instruct, gratitude at block 4 (alpha1), hostility at block 16 (alpha2), log10 T = -0.1, period 5,
Newton tol 1e-12 (tol_for), k = 6 Ritz vectors. Writes only to the referee scratch folder."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/data2/neural-shrimp/tools')
import llm_feigenbaum as lfb            # noqa: E402
import explore_llm_attractors as ela   # noqa: E402
import explore_llm_phase_map as elpm   # noqa: E402
import explore_llm_soft_loop as esl    # noqa: E402

OUT = ela.DATA / 'scratch' / 'physics-review-1015' / 'p5-nilpotent'
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / 'log.jsonl', 'a')
def emit(rec):
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()

P = 5
loop = esl.SoftLoop('qwen0.5b', 4, int(sys.argv[1]) if len(sys.argv) > 1 else 8, 'open', ela.OPENER, 1, 'float64', 'cpu')
v = elpm.load_vectors('llm-phase-map/vectors-l4')['gratitude']
loop.sm.add_steer_layer(16)
w = elpm.load_vectors('llm-phase-map/vectors-l16')['hostility']
def make_cell(a1, a2):
    loop.extra = (16, np.array([a2 * w]))
    return esl.SoftOrbit(loop, a1 * v, 10.0 ** -0.1, chunk=512)

def solve(a1, a2, X, Q):
    t = time.perf_counter()
    cell = make_cell(a1, a2)
    X, Q, hist = lfb.newton(cell, X, P, Q, lfb.tol_for(P, 1e-12), 12)
    ok = hist[-1] < lfb.tol_for(P, 1e-12)
    vals, vecs, res, rh, Q = lfb.ritz(cell, X, P, Q)
    toks = []
    Y = X
    for _ in range(P):
        toks.append(int(cell.probabilities(Y).argmax())); Y = cell.f(Y)
    c1 = float(np.sum(vals).real); c2 = float(sum(vals[i] * vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))).real)
    rec = {'a1': a1, 'a2': a2, 'newton': float(hist[-1]), 'converged': bool(ok), 'tokens': toks,
           'mu': [[float(z.real), float(z.imag)] for z in vals], 'moduli': [float(abs(z)) for z in vals],
           'ritz_res': [float(r) for r in res[:3]], 'T2': float((vals[0] + vals[1]).real), 'D2': float((vals[0] * vals[1]).real),
           'c1': c1, 'c2': c2, 'seconds': round(time.perf_counter() - t, 1)}
    return rec, X, Q

# seed exactly as DSS5: fixed start at the DSS5 seed must reach period 5
tic = time.perf_counter()
a0 = (-0.11873, 0.100693)
cell = make_cell(*a0)
X = cell.iterate(cell.start, 900)
per = cell.detect_period(X, 32, 1e-6)
emit({'seed': a0, 'period_detected': int(per), 'seconds': round(time.perf_counter() - tic, 1)})
if per != P:
    raise SystemExit('seed is not period 5')
Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), 6)))[0]
rec, X, Q = solve(*a0, X, Q); rec['tag'] = 'seed'; emit(rec)

targets = {'P1': (-0.118494, 0.101735), 'P2': (-0.119004, 0.102388)}
which = sys.argv[2] if len(sys.argv) > 2 else 'P1'
tgt = targets[which]
# march from the seed to the target in steps no larger than the DSS5 grid spacing
n = int(np.ceil(max(abs(tgt[0] - a0[0]) / 8.3e-5, abs(tgt[1] - a0[1]) / 1.5e-4))) + 1
for s in np.linspace(0, 1, n + 1)[1:]:
    a = (a0[0] + s * (tgt[0] - a0[0]), a0[1] + s * (tgt[1] - a0[1]))
    rec, X, Q = solve(*a, X, Q); rec['tag'] = f'march-{which}'; emit(rec)
    if not rec['converged']:
        raise SystemExit('march failed')
# 2D Newton on (c1, c2) = (sum of Ritz values, sum of pairwise products): smooth surrogates of (T, D)
a = np.array(tgt, dtype=float)
h = np.array([4e-6, 8e-6])
for it in range(6):
    r0, X, Q = solve(a[0], a[1], X, Q); r0['tag'] = f'newton2d-{which}-{it}'; emit(r0)
    if max(r0['moduli']) < 3e-3 or not r0['converged']:
        break
    F0 = np.array([r0['c1'], r0['c2']])
    J = np.zeros((2, 2))
    for k in range(2):
        b = a.copy(); b[k] += h[k]
        rk, Xk, Qk = solve(b[0], b[1], X.clone(), Q.copy())
        J[:, k] = (np.array([rk['c1'], rk['c2']]) - F0) / h[k]
    step = np.linalg.solve(J, -F0)
    # damp: never move more than a quarter of a DSS5 cell per iteration
    lim = np.array([8.3e-5, 1.5e-4]) / 4
    scale = min(1.0, float(np.min(lim / np.maximum(np.abs(step), 1e-30))))
    emit({'tag': f'step-{which}-{it}', 'J': J.tolist(), 'step': step.tolist(), 'damping': scale})
    a = a + scale * step
emit({'done': which, 'seconds': round(time.perf_counter() - tic, 1)})
