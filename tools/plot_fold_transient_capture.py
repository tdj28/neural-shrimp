"""Figures: chaotic wandering that ends in the period-3 cycle at the post-fold setting (reads saved data only).

fig-fold-transient-capture: (a) one fresh trajectory, its distance to the slow channel and to the cycle, with the
  loop's most probable words; (b) all 48 fresh trajectories, update by update.
fig-fold-twins-capture (with --twins): (a) copies of one start nudged by 1e-12, 1e-10 and 1e-8 drift apart at the
  rate measured with the network's derivatives; (b) every capture closes in at the cycle's own rate.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ORANGE, RED, BLUE, INK = '#c8641e', '#a43b43', '#1f6785', '#222222'
NEAR = 0.03       # relative distance of this or the previous state to the channel center: a slow passage
ON = 1e-3         # relative distance to the cycle: captured
EXAMPLE = 13      # FGR5 row shown in fig-fold-transient-capture (a)
GROWTH = 0.0734   # mean of the 26 complete FGR3 tangent-growth windows, per update (fold-return-autonomous.json)


def regimes(d3, dc, valid):
    """0 burst, 1 slow passage, 2 on the cycle, 3 after the run stopped or ended."""
    pair = np.fmin(dc, np.concatenate([dc[:, :1], dc[:, :-1]], axis=1))
    R = np.where(pair < NEAR, 1, 0)
    for i in range(len(d3)):
        hit = np.flatnonzero(d3[i, :valid[i]+1] < ON)
        if len(hit):
            R[i, hit[0]:valid[i]+1] = 2
        R[i, valid[i]+1:] = 3
    return R, pair


def words(tk, ids):
    return ''.join(tk.decode([int(t)]) for t in ids).replace('\n', '↵').strip()


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
        fig.savefig(out/f'{name}.{ext}', dpi=200, metadata=meta)
    plt.close(fig)


def twin_data(tw, orbit):
    res = json.loads((tw/'result.json').read_text()); sel = json.loads((tw/'selection.json').read_text())
    labels = sel['labels']; tv = np.array(res['valid_through'])
    shards = sorted((tw/'trajectories').glob('updates-*.npz'), key=lambda q: int(q.stem.split('-')[1]))
    on = np.linalg.norm(orbit, axis=1)
    with np.load(tw/'inputs.npz') as z:
        center = z['center']
    gaps, d3, dc = [], [], []
    for f in shards:
        with np.load(f) as s:
            S = s['states']
        for x in S:
            d3.append((np.linalg.norm(x[:, None, :]-orbit[None], axis=2)/on[None]).min(axis=1))
            dc.append(np.linalg.norm(x-center, axis=1)/np.linalg.norm(center))
            ref = np.array([k-k % 4 for k in range(len(labels))])
            gaps.append(np.linalg.norm(x-x[ref], axis=1)/np.linalg.norm(x[ref], axis=1))
    gaps, d3, dc = np.array(gaps), np.array(d3), np.array(dc); steps = np.arange(1, len(gaps)+1)
    for k in range(len(labels)):
        gaps[steps > min(tv[k], tv[k-k % 4]), k] = np.nan
    # regimes per copy (columns are updates 1..H), same rules as the fresh trajectories
    R, _ = regimes(d3.T, dc.T, tv-1)
    for k in range(len(labels)):
        d3[steps > tv[k], k] = np.nan
    return labels, gaps, d3, steps, tv, res['outcomes'], R


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--series', required=True); p.add_argument('--twins'); p.add_argument('--out', required=True)
    p.add_argument('--start', type=int, default=0, help='which twin start (0-3) to draw')
    p.add_argument('--hf-cache', default='/data2/neural-shrimp-data/models/hf/hub')
    a = p.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=a.hf_cache, local_files_only=True)
    z = np.load(Path(a.series)/'series.npz'); meta = json.loads((Path(a.series)/'meta.json').read_text())
    d3, dc, tok, valid = z['d3'], z['dc'], z['top_token'], z['valid_through']
    R, pair = regimes(d3, dc, valid)
    out = Path(a.out)
    for mobile in (False, True):
        fig, ax = plt.subplots(2 if mobile else 1, 1 if mobile else 2, figsize=(5.6, 9.6) if mobile else (11.5, 4.6),
                               layout='constrained')
        x, w = ax
        i = EXAMPLE; n = int(valid[i]); t = np.arange(n+1)
        x.semilogy(t, d3[i, :n+1], color=BLUE, lw=1.6, label='distance to the 3-step cycle')
        x.semilogy(t, np.clip(pair[i, :n+1], 1e-5, None), color=ORANGE, lw=.7, label='distance to the slow channel')
        cap = int(np.flatnonzero(d3[i, :n+1] < ON)[0])
        seg = np.arange(250, 480); k1 = int(seg[np.argmin(pair[i, seg])])
        x.annotate('“' + words(tk, tok[i, k1-3:k1+7]) + '…”', (k1, pair[i, k1]), xytext=(150, 2e-7), fontsize=8,
                   color=ORANGE, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.6))
        b0 = int(np.flatnonzero(pair[i, :cap] < NEAR)[-1])          # end of the last slow passage
        echo = {tk.convert_tokens_to_ids(tk.tokenize(' it'))[0], tk.convert_tokens_to_ids(tk.tokenize(' is'))[0]}
        while tok[i, b0+1] in echo:                                   # skip the passage's last 'it is'
            b0 += 1
        x.annotate('“' + words(tk, tok[i, b0+1:b0+15]) + '…”', (b0+4, pair[i, b0+4]), xytext=(250, 1e-9),
                   fontsize=8, color=RED, arrowprops=dict(arrowstyle='-', color=RED, lw=.6))
        x.annotate('“' + words(tk, tok[i, n-11:n+1]) + '…”', (n-60, d3[i, n-60]), xytext=(250, 1e-11), fontsize=8,
                   color=BLUE, arrowprops=dict(arrowstyle='-', color=BLUE, lw=.6))
        x.set(xlim=(0, n+30), ylim=(1e-15, 6), xlabel='Network updates', ylabel='Relative distance',
              title='(a) One trajectory: wandering, then the cycle')
        x.legend(fontsize=8, loc='lower left', frameon=False)
        x.spines[['top', 'right']].set_visible(False)
        end = np.array([np.flatnonzero(R[j] >= 2)[0] if (R[j] >= 2).any() else 10**9 for j in range(len(R))])
        order = np.argsort(end, kind='stable')
        cmap = ListedColormap([RED, ORANGE, BLUE, 'white'])
        w.imshow(R[order], aspect='auto', interpolation='nearest', cmap=cmap, vmin=-.5, vmax=3.5,
                 extent=(0, R.shape[1], len(R), 0))
        w.set(xlabel='Network updates', ylabel='Fresh trajectory, sorted by capture',
              title=f'(b) All {len(R)} fresh trajectories')
        w.legend(handles=[Patch(color=ORANGE, label='slow passage (“it is it is…”)'),
                          Patch(color=RED, label='burst (other words)'),
                          Patch(color=BLUE, label='on the cycle (“certainly.Hello…”)')],
                 fontsize=7.5, loc='upper right', frameon=True, framealpha=.95)
        save(fig, out, 'fig-fold-transient-capture'+('-mobile' if mobile else ''))
    if not a.twins:
        return
    with np.load(Path(meta['cycle'])) as c:
        orbit = c['orbit']
    labels, gaps, td3, steps, tv, outcomes, TR = twin_data(Path(a.twins), orbit)
    slope = meta['contraction_per_update']/np.log(10)
    base = 4*a.start
    for mobile in (False, True):
        fig, ax = plt.subplots(3 if mobile else 1, 1 if mobile else 3, figsize=(5.6, 13.5) if mobile else (14, 4.4),
                               layout='constrained')
        x, r, y = ax
        colors = ['#6b4c9a', '#2a8a6e', '#c08a00']
        for c, k in zip(colors, range(base+1, base+4)):
            eps = labels[k]['eps']
            x.semilogy(steps[1::2], gaps[1::2, k], color=c, lw=1,      # one phase of the alternation
                       label=fr'nudged by $10^{{{int(round(np.log10(eps)))}}}$')
            t0 = int(np.log(1/eps)/GROWTH)
            x.semilogy([0, t0], [eps, 1], '--', color=c, lw=.9, alpha=.85)
        x.text(.42, .45, 'dashed: the growth rate measured with the\nnetwork\'s derivatives (doubling every 9.4 updates)',
               transform=x.transAxes, va='top', fontsize=8)
        x.set(xlim=(0, 1000), ylim=(1e-13, 6), xlabel='Network updates', ylabel='Relative distance from the unnudged copy',
              title='(a) Copies of one start drift apart')
        x.legend(fontsize=8, loc='lower right', frameon=False); x.spines[['top', 'right']].set_visible(False)
        rows = TR[base:base+4]
        H = int(np.nanmax(tv[base:base+4]))
        r.imshow(rows[:, :H], aspect='auto', interpolation='nearest', cmap=ListedColormap([RED, ORANGE, BLUE, 'white']),
                 vmin=-.5, vmax=3.5, extent=(0, H, 4, 0))
        r.set_yticks([.5, 1.5, 2.5, 3.5], ['unnudged', r'$10^{-12}$', r'$10^{-10}$', r'$10^{-8}$'])
        r.set(xlabel='Network updates', ylabel='Copy (nudge size)', title='(b) Same start, four different paths to the cycle')
        r.legend(handles=[Patch(color=ORANGE, label='slow passage'), Patch(color=RED, label='burst'),
                          Patch(color=BLUE, label='on the cycle')], fontsize=7.5, loc='upper right', framealpha=.95)
        caps = 0
        for j in range(len(d3)):
            hit = np.flatnonzero(d3[j, :valid[j]+1] < 1e-2)
            if not len(hit):
                continue
            c0 = hit[0]; seg = d3[j, max(0, c0-60):valid[j]+1]
            y.semilogy(np.arange(len(seg))-min(60, c0), seg, color=BLUE, lw=.6, alpha=.45); caps += 1
        tt = np.arange(0, 260)
        y.semilogy(tt, 1e-2*10**(slope*tt), '--', color=INK, lw=1.3,
                   label=fr"the cycle's own rate: $\times${np.exp(meta['contraction_per_update']*3):.3f} every 3 updates")
        y.set(xlim=(-60, 300), ylim=(1e-15, 6), xlabel='Updates after the distance first falls below 0.01',
              ylabel='Relative distance to the 3-step cycle', title=f'(c) All {caps} fresh captures close in at one rate')
        y.legend(fontsize=8, loc='upper right', frameon=False); y.spines[['top', 'right']].set_visible(False)
        save(fig, out, 'fig-fold-twins-capture'+('-mobile' if mobile else ''))


if __name__ == '__main__':
    main()
