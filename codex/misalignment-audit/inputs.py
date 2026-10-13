"""Fixed arm membership and read-only schema/input authentication."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

RAW = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/msm')
CAPTURE = ('p1-free','p1-h040','p1-h050','p1-h050s2','p1-h060','p1-h060s2','p1-h070',
           'p1-r1-060','p1-r1-080','p1-r2-080','p1-ck-060','p1-ck-080')
HISTORY = ('p1-ramp-s1','p1-ramp-s2','p1-c040-s2','p1-exit030','p1-exit020','p1-exit000',
           'p1-exitm020','p1-c030-s1','p1-c030-s2','p1-exit040-s1','p1-exit040-s2',
           'p1-exit030-s2','p1-exit000-s2','p1-exitm020-s2')
# Actual historical controller CLI order, needed for its single bootstrap RNG.
CONTROL = ('p1-ctl-none','p1-ctl-trig-cancel','p1-ctl-trig-over','p1-ctl-trig-long',
           'p1-ctl-trig-kind','p1-ctl-const-02','p1-ctl-const-04')
TRAIN = ('p1-det040','p1-det045')
TEST = ('p1-conf040','p1-conf045')
ALL = CAPTURE+HISTORY+CONTROL+TRAIN+TEST


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def files_for(mode):
    arms = CAPTURE+HISTORY+CONTROL if mode == 'descriptive' else TRAIN+(TEST if mode == 'l8' else ())
    files = []
    for arm in arms:
        names = ['sample.json', 'hostility.npz']
        if arm in CONTROL:
            names += ['turns.npz']
        if arm in TRAIN+TEST:
            names += ['turns.npz', 'in-context.npz', 'activations.npy']
        files.extend(RAW/arm/n for n in names)
    return files


def inventory(paths):
    return [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths]


def load_basic(name, controller=False):
    if name not in ALL:
        raise ValueError('unlisted arm')
    path = RAW/name
    meta = json.loads((path/'sample.json').read_text())
    args = meta['args']
    declared=json.loads((Path(__file__).parent/'schemas.json').read_text())['arms'][name]['args']
    if args != declared:
        raise ValueError('actual sample args differ from predeclared metadata: '+name)
    n = 256 if name in TRAIN else 512 if name in TEST else 128
    if args['replicas'] != n or args['turns'] != 40 or args['model'] != 'qwen0.5b':
        raise ValueError('unexpected experiment metadata')
    with np.load(path/'hostility.npz', allow_pickle=False) as z:
        labels = z['labels'].tolist()
        if labels[0] != 'toxic' or len(labels) != 6:
            raise ValueError(f'classifier labels differ: {labels}')
        scores = z['classifier']
    if scores.shape != (n, 40, 6) or not np.isfinite(scores).all():
        raise ValueError('classifier shape/nonfinite')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('classifier probability range')
    out = {'name': name, 'score': scores[:, :, 0].astype(np.float64), 'args': args}
    if controller:
        with np.load(path/'turns.npz', allow_pickle=False) as z:
            out['counter'] = z['counter'].astype(np.float64)
        if not np.isfinite(out['counter']).all():
            raise ValueError('nonfinite counter')
        out['texts'] = meta['texts']
    return out


def load_detection(name):
    if name not in TRAIN+TEST:
        raise ValueError('unlisted detection arm')
    out = load_basic(name)
    path = RAW/name
    n = len(out['score'])
    out['dose'] = .04 if name.endswith('040') else .045
    if out['args']['alphas'] != [out['dose']]:
        raise ValueError('dose mismatch')
    with np.load(path/'turns.npz', allow_pickle=False) as z:
        out['q'], out['q_plus'] = z['probe_q'], z['probe_q_plus']
    with np.load(path/'in-context.npz', allow_pickle=False) as z:
        unit = z['unit']
        out['pre'] = z['pre_reply_18']  # preserve source f16 until table construction
        out['tok'] = z['token_proj_18'].astype(np.float64)
    acts = np.load(path/'activations.npy', mmap_mode='r', allow_pickle=False)
    if (unit.shape != (896,) or acts.shape != (n,40,896)
            or out['pre'].shape != (n,40,896) or out['tok'].shape != (n,40,60)
            or out['q'].shape != (n,40) or out['q_plus'].shape != (n,40)):
        raise ValueError('detection array shape')
    out['axis'] = acts @ unit
    if not all(np.isfinite(out[k]).all() for k in ('q','q_plus','pre','axis')):
        raise ValueError('nonfinite detector input')
    return out
