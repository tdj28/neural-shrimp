"""Per-update series of saved fold-setting trajectories: distance to the audited period-3 orbit, distance to the
slow channel's centre, and the most probable next token (reads saved full states only; no network calls).

python tools/fold_transient_series.py --run <run dir> --out <audit dir> --backup <backup dir>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json

CYCLE = Path('/data2/neural-shrimp-data/audits/fold-global-return/20261113-cycle-01')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    a = p.parse_args()
    run, out, backup = Path(a.run), Path(a.out), Path(a.backup)
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    result = json.loads((run/'result.json').read_text())
    valid = np.array(result['valid_through'])
    with np.load(CYCLE/'orbit-and-monodromy.npz') as z:
        orbit = z['orbit']; M = z['monodromy']
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    with np.load(run/'inputs.npz') as z:
        centre, left, right = z['centre'], z['left'], z['right']
    U = json.loads((Path(json.loads((run/'plan.json').read_text())['parent_curve'])/'plan.json').read_text())['width']
    shards = sorted((run/'trajectories').glob('updates-*.npz'), key=lambda q: int(q.stem.split('-')[1]))
    with np.load(run/'trajectories'/'initial.npz') as z:
        prev = z['states'].copy()
    B = prev.shape[0]; H = int(valid.max())
    d3 = np.full((B, H+1), np.nan); dc = np.full((B, H+1), np.nan); v = np.full((B, H+1), np.nan)
    tok = np.full((B, H+1), -1, dtype=np.int64); prob = np.full((B, H+1), np.nan)
    on = np.linalg.norm(orbit, axis=1); cn = np.linalg.norm(centre)

    def fill(step, x, ids=None, pr=None):
        d3[:, step] = (np.linalg.norm(x[:, None, :]-orbit[None], axis=2)/on[None]).min(axis=1)
        dc[:, step] = np.linalg.norm(x-centre, axis=1)/cn
        v[:, step] = (x-centre)@left/U
        if ids is not None:
            tok[:, step] = ids[:, 0]; prob[:, step] = pr[:, 0]

    fill(0, prev)
    for f in shards:
        lo, hi = map(int, f.stem.split('-')[1:3])
        with np.load(f) as z:
            S, I, P = z['states'], z['top_ids'], z['top_probabilities']
        for k, step in enumerate(range(lo, hi+1)):
            fill(step, S[k], I[k], P[k])
    for i in range(B):                                  # frozen padding after a row stopped is not data
        d3[i, valid[i]+1:] = np.nan; dc[i, valid[i]+1:] = np.nan; v[i, valid[i]+1:] = np.nan
        tok[i, valid[i]+1:] = -1; prob[i, valid[i]+1:] = np.nan
    np.savez_compressed(out/'series.npz', d3=d3.astype(np.float32), dc=dc.astype(np.float32), v=v.astype(np.float32),
                        top_token=tok, top_probability=prob.astype(np.float32), valid_through=valid)
    meta = {'run': str(run), 'run_manifest_sha256': sha(run/'manifest.json'),
            'cycle': str(CYCLE/'orbit-and-monodromy.npz'), 'cycle_sha256': sha(CYCLE/'orbit-and-monodromy.npz'),
            'spectral_radius': rho, 'contraction_per_update': float(np.log(rho)/3),
            'outcomes': [o['kind'] for o in result['outcomes']], 'valid_through': valid.tolist(),
            'series': 'd3: min relative distance to the three orbit states; dc: relative distance to the channel centre; '
                      'v: channel coordinate in units of U; top_token/top_probability: most probable next token. '
                      'Values after valid_through are NaN (the row had stopped).'}
    write_json(out/'meta.json', meta)
    files = []
    for f in ('series.npz', 'meta.json'):
        backup_file(out/f, out, backup); files.append({'path': f, 'bytes': (out/f).stat().st_size, 'sha256': sha(out/f)})
    write_json(out/'manifest.json', files); backup_file(out/'manifest.json', out, backup)
    print(json.dumps({k: meta[k] for k in ('spectral_radius', 'contraction_per_update')}))


if __name__ == '__main__':
    main()
