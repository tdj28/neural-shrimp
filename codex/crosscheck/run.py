"""Bounded independent-map experiment worker; run through launch.py."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from qwen_map import ManualQwenSoftMap, sha256

DATA = Path('/data2/neural-shrimp-data/explore/llm-soft-loop')


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, torch.Tensor):
        return plain(value.detach().numpy())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def profile(model, output, event):
    vectors = np.load(DATA / 'normal-form-p2-t-0.115/vectors.npz')
    x = torch.from_numpy(vectors['x_c'].copy())
    alpha, t = .06279827217667086, -.115
    before = time.monotonic()
    cache = model.prefix_cache(alpha)
    event('prefix_cache', seconds=time.monotonic()-before)
    before = time.monotonic()
    y = model.period_map(x, alpha, t, 2, cache)
    closure = float(torch.linalg.vector_norm(y-x) / torch.linalg.vector_norm(x))
    event('fold_seed_closure', relative_residual=closure, seconds=time.monotonic()-before)
    if closure > 1e-10:
        raise RuntimeError('independent forward disagrees with current-map fold seed')
    cached = model(x, alpha, t, cache)
    uncached = model(x, alpha, t)
    cache_error = float(torch.linalg.vector_norm(cached-uncached))
    if cache_error > 1e-13:
        raise RuntimeError('cache consistency failed')
    derivative_rows = []
    for period in (1, 8):
        xx = x if period == 1 else torch.from_numpy(np.load(DATA / 'orbit64-zr-p8/orbit-state.npy').reshape(-1).copy())
        aa, tt = (alpha, t) if period == 1 else (.0717, -.1205)
        cc = model.prefix_cache(aa)
        # Reproducible non-random direction; includes every coordinate.
        direction = torch.sin(torch.arange(1, model.d+1, dtype=torch.float64))
        direction /= torch.linalg.vector_norm(direction)
        before = time.monotonic()
        jv = model.jvp(xx, direction, aa, tt, period, cc)
        seconds = time.monotonic()-before
        ladder = []
        for relative_h in (1e-3, 3e-4, 1e-4, 3e-5):
            h = relative_h * float(torch.linalg.vector_norm(xx))
            fd = (model.period_map(xx+h*direction, aa, tt, period, cc) - model.period_map(xx-h*direction, aa, tt, period, cc))/(2*h)
            error = float(torch.linalg.vector_norm(fd-jv) / torch.linalg.vector_norm(jv))
            ladder.append(dict(relative_h=relative_h, h=h, relative_error=error))
        row = dict(period=period, jvp_seconds=seconds, jvp_norm=float(torch.linalg.vector_norm(jv)), ladder=ladder)
        derivative_rows.append(row)
        event('derivative_ladder', **row)
        if min(z['relative_error'] for z in ladder) > 1e-5:
            raise RuntimeError('composed derivative precision gate failed')
    timings = []
    for batch in (4, 16):
        points = x[None].expand(batch, -1).clone()
        directions = torch.eye(model.d, dtype=torch.float64)[:batch]
        before = time.monotonic()
        jv = model.jvp(points, directions, alpha, t, 2, cache)
        seconds = time.monotonic()-before
        if not torch.isfinite(jv).all():
            raise RuntimeError('nonfinite batched JVP')
        row = dict(batch=batch, period=2, seconds=seconds,
                   estimated_p2_dense_seconds=seconds*np.ceil(model.d/batch),
                   estimated_p8_dense_seconds=4*seconds*np.ceil(model.d/batch))
        timings.append(row)
        event('batch_profile', **row)
    result = dict(evidence='rung2 exploratory profile', fold_seed_relative_closure=closure,
                  cached_uncached_absolute_error=cache_error, derivatives=derivative_rows,
                  timings=timings, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(output/'profile.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['profile'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    def event(kind, **fields):
        entry = dict(event=kind, elapsed_seconds=time.monotonic()-started, **plain(fields))
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print(json.dumps(entry, allow_nan=False), flush=True)
    try:
        if not 1 <= args.threads <= 8:
            raise ValueError('worker thread limit')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        for row in json.loads(Path(__file__).with_name('reference-inputs.json').read_text()):
            if Path(row['path']).stat().st_size != row['bytes'] or sha256(row['path']) != row['sha256']:
                raise RuntimeError('input hash mismatch: '+row['path'])
        event('inputs_checked', torch_version=torch.__version__, numpy_version=np.__version__,
              threads=torch.get_num_threads(), pid=os.getpid())
        model = ManualQwenSoftMap.from_pinned_files()
        event('model_loaded', peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        profile(model, args.output, event)
        save(args.output/'status.json', dict(status='complete', seconds=time.monotonic()-started))
    except BaseException as error:
        save(args.output/'status.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
