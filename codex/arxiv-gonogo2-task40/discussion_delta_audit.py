#!/usr/bin/env python3
"""New tiny saved-data reduction only; requires Root checkpoint before execution."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

ROOT = Path('/data2/neural-shrimp')
DATA = Path('/data2/neural-shrimp-data/explore')
OUT = ROOT / 'codex/arxiv-gonogo2-task40/discussion-delta-evidence.json'
PINS = {}

def read(path):
    before = path.stat()
    if before.st_size > 8 * 1024**2:
        raise ValueError('small-input cap')
    raw = path.read_bytes()
    after = path.stat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key(before) != key(after) or len(raw) != before.st_size:
        raise ValueError('input changed')
    PINS[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw

def obj(path):
    def pairs(rows):
        d = {}
        for k, v in rows:
            if k in d:
                raise ValueError('duplicate JSON key')
            d[k] = v
        return d
    return json.loads(read(path), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def main():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('single-thread environment required')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('offline CPU environment required')
    import numpy as np
    start = time.monotonic()
    spectrum = obj(DATA / 'e2e/lyap-spectrum/lyapunov-spectrum.json')
    cells = obj(ROOT / 'docs/e2e/lyap-cells.json')
    if spectrum['complete'] is not True or len(spectrum['cells']) != 3:
        raise ValueError('incomplete spectra')
    if spectrum['args']['record'] != 600 or spectrum['args']['k'] != 4:
        raise ValueError('unexpected spectrum horizon')
    spectra = []
    for r, pair in zip(spectrum['cells'], cells['cells']):
        x = r['exponents']
        if [r['log10_temperature'], r['alpha']] != pair or len(x) != 4:
            raise ValueError('spectrum coordinate/schema mismatch')
        if not all(math.isfinite(v) for v in x) or x != sorted(x, reverse=True):
            raise ValueError('nonfinite or unordered spectrum')
        if not (x[0] > 0 and all(v < 0 for v in x[1:]) and x[0]+x[1] < 0):
            raise ValueError('KY branch not supported')
        spectra.append({**r, 'KY_recomputed': 1+x[0]/abs(x[1])})
    bridge = {}
    for name, m in [('exact', 0), ('m64', 64), ('m8', 8), ('m1', 1)]:
        base = DATA / f'llm-soft-loop/bridge-{name}-t-0.115'
        meta = obj(base / 'walk.json')
        with np.load(io.BytesIO(read(base / 'walk.npz')), allow_pickle=False) as z:
            p, alphas, tokens = z['probabilities'], z['alphas'], list(z['tokens'])
        if p.shape != (512, 96, 6) or len(alphas) != 96 or not np.isfinite(p).all():
            raise ValueError('bridge shape/finite mismatch')
        if meta['args']['sample_m'] != m or meta['args']['replicas'] != 32:
            raise ValueError('bridge metadata mismatch')
        if any(meta['args'][k] != v for k, v in
               [('dtype', 'float32'), ('steps', 512), ('transient', 100)]):
            raise ValueError('bridge horizon/precision mismatch')
        if [r['alpha'] for r in meta['cells']] != alphas.tolist():
            raise ValueError('bridge coordinate mismatch')
        it, a, there, is_ = (tokens.index(t) for t in (' it', ' a', ' there', ' is'))
        mask = (p[:, :, is_] > .5) | ((p[:, :, it] > p[:, :, a]) &
               (p[:, :, it] > p[:, :, there]) & (p[:, :, it] > .1))
        groups = []
        for alpha in sorted(set(alphas.tolist())):
            cols = [j for j, x in enumerate(alphas) if x == alpha]
            if len(cols) != 32:
                raise ValueError('missing replicas')
            lengths, left_censored, right_censored = [], [], []
            for col in cols:
                run = 0
                for t, active in enumerate(mask[:, col]):
                    if active:
                        run += 1
                    else:
                        if run >= 6:
                            lengths.append(run)
                            if t-run == 0:
                                left_censored.append(run)
                        run = 0
                if run >= 6:
                    right_censored.append(run)
            groups.append(dict(alpha=alpha, eps=round(alpha-.0627983, 7),
                trajectories=len(cols), recorded_steps=512*len(cols),
                channel_steps=int(mask[:, cols].sum()), channel_fraction=float(mask[:, cols].mean()),
                phase_count=len(lengths), phase_mean=sum(lengths)/len(lengths) if lengths else None,
                phase_lengths=lengths, included_left_censored=left_censored,
                excluded_right_censored=right_censored))
        bridge[name] = dict(args=meta['args'], groups=groups,
            phase_rule='Token-probability mask; right-terminated runs of >=6 F-steps; left endpoint retained as original plot.',
            original_precision='float32; no new orbit or effective-precision qualification')
    read(ROOT / 'tools/paper_llm_figures.py')
    result = dict(article_commit='bb42a82189da91c37f50804c077832df9e2237ae',
        spectra=dict(args=spectrum['args'], rows=spectra,
            negative_range=[min(v for r in spectra for v in r['exponents'][1:]),
                            max(v for r in spectra for v in r['exponents'][1:])]),
        bridge=bridge, manifest=PINS, elapsed_seconds=time.monotonic()-start,
        model_calls=0, old_controls_repeated=0,
        limits='Saved arithmetic only; source/run/asset history not independently reconstructed.')
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(output=str(OUT), inputs=len(PINS), elapsed=result['elapsed_seconds'])))

if __name__ == '__main__':
    main()
