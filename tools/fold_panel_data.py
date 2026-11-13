"""Read and audit saved fold passages; no model execution or fitted trajectory.

The raw passage files store v = sign(b) * l.T @ (x - x_c), not the
unoriented u used in the branch diagram. Both observation phases are retained.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def coordinates(states, centre, left, right, sign):
    delta = np.asarray(states, dtype=np.float64) - centre
    v = delta @ (sign * left)
    transverse = np.linalg.norm(delta - v[:, None] * (sign * right), axis=1)
    return v, transverse


def channel_mask(v, transverse, width, transverse_limit=0.05):
    if width <= 0 or transverse_limit <= 0:
        raise ValueError("Positive channel limits required")
    if v.shape != transverse.shape or not np.isfinite(v).all() or not np.isfinite(transverse).all():
        raise ValueError("Matching finite coordinate arrays required")
    return (np.abs(v) < width) & (transverse < transverse_limit)


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.where(edges == 1)[0].tolist(), (np.where(edges == -1)[0] - 1).tolist()))


def load_panel(root):
    root = Path(root)
    run = root / 'passage-natural-words-p2-t-0.115'
    original = root / 'passage-natural-p2-t-0.115'
    normal = root / 'normal-form-p2-t-0.115'
    nf = json.loads((normal / 'normal-form.json').read_text())
    vec = np.load(normal / 'vectors.npz')
    xc, left, right = (vec[k] for k in ('x_c', 'l', 'r'))
    assert abs(left @ right - 1) < 1e-12 and nf['a'] * nf['b'] > 0
    sign = float(np.sign(nf['b']))
    meta = json.loads((run / 'passage.json').read_text())
    picks = [i for i, r in enumerate(meta['rows']) if r['eps'] == 1e-5 and r['entrance'] == 'natural-3']
    assert len(picks) == 1
    pick = picks[0]
    with np.load(run / 'top.npz') as z:
        states = z['states'][:, pick]
    with np.load(run / 'phase-check.npz') as z:
        later = z['next_states']
        stored = np.stack([z['u_same_phase'], z['u_one_step_later']])
    assert states.shape == later.shape == (600, len(xc))
    computed = [coordinates(x, xc, left, right, sign) for x in (states, later)]
    v = np.stack([p[0] for p in computed])
    transverse = np.stack([p[1] for p in computed])
    assert np.max(np.abs(v - stored)) < 1e-12
    assert np.array_equal(stored[0], np.load(run / 'u.npy')[:, pick])
    assert np.array_equal(stored[0], np.load(original / 'u.npy')[:600, pick])
    assert np.max(np.abs(transverse[0] - np.load(run / 'transverse.npy')[:, pick])) < 1e-12
    width = 0.1 / abs(nf['b'])
    assert abs(width - meta['U']) < 1e-14
    mask = channel_mask(v, transverse, width)
    files = [normal / 'normal-form.json', normal / 'vectors.npz', run / 'passage.json',
             run / 'top.npz', run / 'phase-check.npz', run / 'u.npy', run / 'transverse.npy', original / 'u.npy']
    receipt = dict(
        source='saved model trajectory and saved one-step phase check; no new simulation',
        horizon=600, row=pick, epsilon=1e-5, width=width, transverse_limit=0.05,
        coordinate='v=sign(b)*l.T@(x-x_c); transverse=norm(x-x_c-v*sign(b)*r)',
        max_projection_discrepancy=float(np.max(np.abs(v - stored))),
        phase_samples=[int(m.sum()) for m in mask], either_phase_samples=int(mask.any(axis=0).sum()),
        phase_intervals=[intervals(m) for m in mask],
        endpoint_intervals_are_censored=True,
        max_transverse_when_coordinate_inside=float(transverse[np.abs(v) < width].max()),
        white_means='Neither observation phase passes the stated local-channel criterion; not a chaos label',
        input_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return dict(v=stored, transverse=transverse, mask=mask, width=width, receipt=receipt)


def draw_passages(ax, data, horizon=600, fontsize=7):
    """Signed local view: separate traces, gaps outside +/-3U, vertical time shading."""
    v, mask = data['v'][:, :horizon], data['mask'][:, :horizon]
    U = data['width']
    for start, end in intervals(mask.any(axis=0)):
        ax.axvspan(max(0, start - .5), end + .5, color='#fde0c5', lw=0, zorder=0)
    for level in (-1, 1):
        ax.axhline(level, color='#a6430a', ls='--', lw=.6)
    ax.axhline(0, color='0.55', lw=.5)
    for p, color, label in [(0, 'black', 'every two updates'), (1, '#0072b2', 'one update later')]:
        local = np.where(np.abs(v[p]) <= 3 * U, v[p] / U, np.nan)
        ax.plot(np.arange(horizon), local, color=color, lw=.85, label=label)
    ax.set(xlim=(0, horizon), ylim=(-3, 3), yticks=[-3, -1, 0, 1, 3])
    ax.set_ylabel(r'Signed position $v/U$', fontsize=fontsize)
    ax.set_xlabel(r'Two-update intervals $n$', fontsize=fontsize)
    ax.legend(loc='upper left', fontsize=fontsize - 1, ncol=2, frameon=True,
              facecolor='white', edgecolor='none', framealpha=.95, handlelength=1.5)
