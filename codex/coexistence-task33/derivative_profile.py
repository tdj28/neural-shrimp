"""Prepared composed-map AD/FD cost profile; no execution admission or eigenanalysis."""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--card', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); card = json.loads(Path(args.card).read_text())
    if card['alpha'] != .06468 or card['log10T'] != -.148: raise ValueError('wrong exact cell')
    for e in card['inputs']:
        if sha(e['path']) != e['sha256']: raise ValueError('input hash mismatch')
    states = np.load(card['states'], allow_pickle=False)
    if states.shape != (513, 2, 896) or not np.isfinite(states).all(): raise ValueError('invalid complete history')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(card['repository']) / 'codex/crosscheck'))
    import torch
    from qwen_map import ManualQwenSoftMap
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    started = time.monotonic(); model = ManualQwenSoftMap.from_pinned_files()
    cache = model.prefix_cache(card['alpha']); rows = []
    for j, period in enumerate((2, 3)):
        x = torch.from_numpy(states[-period, j].copy())
        basis = torch.eye(896, dtype=torch.float64)[:8]
        v = basis.sum(dim=0) / np.sqrt(8.)
        with torch.no_grad():
            t = time.monotonic()
            columns = model.jvp(x[None, :].expand(8, -1).clone(), basis, card['alpha'], card['log10T'], period, cache)
            batch_seconds = time.monotonic() - t
            t = time.monotonic(); direct = model.jvp(x, v, card['alpha'], card['log10T'], period, cache)
            direct_seconds = time.monotonic() - t
            arrays = {'state': x.numpy(), 'basis_rows': basis.numpy(), 'columns_as_rows': columns.numpy(), 'direction': v.numpy(), 'direct_jvp': direct.numpy()}
            fdrows = []
            for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
                t = time.monotonic()
                plus = model.period_map(x+h*v, card['alpha'], card['log10T'], period, cache)
                minus = model.period_map(x-h*v, card['alpha'], card['log10T'], period, cache)
                fd = (plus-minus)/(2*h)
                arrays['plus_'+str(h)] = plus.numpy(); arrays['minus_'+str(h)] = minus.numpy()
                fdrows.append({'h': h, 'seconds': time.monotonic()-t, 'absolute_inf_error': float((fd-direct).abs().max())})
            if not all(np.isfinite(a).all() for a in arrays.values()): raise ValueError('nonfinite derivative evidence')
            np.savez(out/('p'+str(period)+'.npz'), **arrays)
            rows.append({'period': period, 'state_index': 513-period, 'basis_columns': list(range(8)), 'batch_seconds': batch_seconds, 'single_jvp_seconds': direct_seconds,
              'batch_linear_combination_vs_single_inf': float((columns.sum(dim=0)/np.sqrt(8.)-direct).abs().max()), 'fd_ladder': fdrows})
    for e in card['inputs']:
        if sha(e['path']) != e['sha256']: raise ValueError('post-run input hash mismatch')
    (out/'report.json').write_text(json.dumps({'rows': rows, 'elapsed_seconds': time.monotonic()-started, 'rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'scope': 'Eight native composed derivative columns and fixed directional FD ladder; not a dense monodromy, multiplier, stability or interval certificate.'}, indent=2, allow_nan=False)+'\n')
if __name__ == '__main__': main()
