"""Months pilot: does Qwen2.5-0.5B's month geometry (Karkada et al. 2026) show up in the soft-token loop?

static : (A) first-harmonic share of the centered input embeddings of months, weekdays and numbers, against
         permutation and random-token nulls; (B) the same share for hidden states at the month token, layer by layer.
sweep  : the unsteered soft-token loop under a month-reciting prompt (window 2, float64 CPU) across temperature,
         recording states, readout tokens, top-64 probabilities and a twin Lyapunov exponent.
analyze: month itineraries, periods and exponents per temperature, the states on the month circle, and the readout
         covariance trace of the actual competitors against the same probabilities on random tokens.

Exploratory pilot (run card in docs/research-log.md, 19 November 2026). No steering; no claim beyond this prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
DATA = Path('/data2/neural-shrimp-data')
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
NUMBERS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
PREFIX = {'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                       {'role': 'user', 'content': 'Write the names of the months in order, separated by single '
                                                   'spaces, and keep repeating them without stopping.'}]}
CONTEXT = 'The festival is held every year in{w}, when the town fills with visitors.'
TOPK = 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harmonic_share(X, n_period=None):
    """Share of the centered rows' variance in the first cyclic harmonic of their order (rows in that order)."""
    n = len(X); P = n_period or n
    Xc = X-X.mean(0)
    m = np.arange(n)
    B = np.stack([np.cos(2*np.pi*m/P), np.sin(2*np.pi*m/P)], 1)
    B = B-B.mean(0)
    Q, _ = np.linalg.qr(B)
    return float(np.sum((Q.T@Xc)**2)/np.sum(Xc**2))


def closure_stats(X):
    """Is the ordered set closed (last item next to the first) or open? Distances in the full space."""
    n = len(X)
    D = np.linalg.norm(X[:, None]-X[None], axis=2)
    adj = [D[k, k+1] for k in range(n-1)]
    wrap = D[n-1, 0]
    others = [D[i, j] for i in range(n) for j in range(i+2, n) if not (i == 0 and j == n-1)]
    return {'wrap_over_median_adjacent': float(wrap/np.median(adj)),
            'share_of_nonadjacent_pairs_closer_than_wrap': float(np.mean(np.array(others) < wrap))}


def circle_stats(X, rng, perms=20000):
    share = harmonic_share(X)
    null = np.array([harmonic_share(X[rng.permutation(len(X))]) for _ in range(perms)])
    Xc = X-X.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc@vt[:2].T
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    steps = np.diff(np.r_[ang, ang[0]+np.sign(ang[-1]-ang[0] or 1)*2*np.pi])
    monotone = int(max((steps > 0).sum(), (steps < 0).sum()))
    return {'first_harmonic_share': share, 'permutation_p': float((1+np.sum(null >= share))/(1+len(null))),
            'permutation_null_median': float(np.median(null)), 'pc12_share': float((s[:2]**2).sum()/(s**2).sum()),
            'pc12_angular_steps_same_direction': monotone, 'n': len(X), 'pc12_coordinates': xy.tolist()}


def token_ids(tok, words, space=True):
    ids = []
    for w in words:
        t = tok(((' ' if space else '')+w), add_special_tokens=False)['input_ids']
        if len(t) != 1:
            raise ValueError(f'{w!r} is not one token')
        ids.append(t[0])
    return ids


def static(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    cache = str(DATA/'models/hf/hub')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache, local_files_only=True,
                                                 torch_dtype=torch.float32)     # geometry only; float32 suffices
    model.eval()
    E = model.get_input_embeddings().weight.detach().double().numpy()     # bf16-stored weights, exact in float64
    rng = np.random.default_rng(20261119)
    out = {'model_revision': getattr(model.config, '_commit_hash', None), 'sets': {}}
    sets = {'months': token_ids(tok, MONTHS), 'weekdays': token_ids(tok, WEEKDAYS), 'numbers': token_ids(tok, NUMBERS)}
    for name, ids in sets.items():
        out['sets'][name] = {'token_ids': ids, 'embeddings': {**circle_stats(E[ids], rng), **closure_stats(E[ids])}}
    # random sets of 12 single-token words with a leading space, in arbitrary order
    vocab = [i for i, s in enumerate(tok.convert_ids_to_tokens(list(range(E.shape[0]))))
             if s is not None and s.startswith('Ġ') and s[1:].isalpha() and len(s) > 3]
    shares = [harmonic_share(E[rng.choice(vocab, 12, replace=False)]) for _ in range(5000)]
    out['random_12_word_sets'] = {'sets': 5000, 'median_share': float(np.median(shares)),
                                  'p95_share': float(np.percentile(shares, 95)), 'max_share': float(np.max(shares))}
    # hidden states at the word, layer by layer
    for name, words in (('months', MONTHS), ('weekdays', WEEKDAYS), ('numbers', NUMBERS)):
        H = []
        for w in words:
            text = CONTEXT.format(w=' '+w)
            enc = tok(text, return_tensors='pt', add_special_tokens=False)
            ids = enc['input_ids'][0].tolist()
            pos = ids.index(token_ids(tok, [w])[0])
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            H.append(np.stack([h[0, pos].double().numpy() for h in hs]))
        H = np.array(H)                                   # (words, layers+1, d)
        layers = []
        for L in range(H.shape[1]):
            st = {**circle_stats(H[:, L], rng, perms=5000), **closure_stats(H[:, L])}
            st.pop('pc12_coordinates')
            layers.append({'layer': L, **st})
        out['sets'][name]['hidden_states'] = layers
        out['sets'][name]['context'] = CONTEXT
    Path(args.out).mkdir(parents=True, exist_ok=False)
    (Path(args.out)/'static.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {'share': v['embeddings']['first_harmonic_share'], 'p': v['embeddings']['permutation_p']}
                      for k, v in out['sets'].items()}, indent=1))


def sweep(args):
    from explore_llm_soft_loop import SoftLoop
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    prefix_rel = f'months-loop/{out.name}-prefix.json'
    (DATA/'explore'/prefix_rel).write_text(json.dumps(PREFIX))
    loop = SoftLoop('qwen0.5b', 12, args.threads, f'file:{prefix_rel}', '', args.window, 'float64', 'cpu')
    torch = loop.torch
    norms = [m for m in loop.model.modules() if type(m).__name__.endswith('RMSNorm')]
    rotary = [m for m in loop.model.modules() if type(m).__name__.endswith('RotaryEmbedding')]
    if not (norms and rotary and all(m.forward.__func__.__qualname__.startswith('keep_float64_norms.') for m in norms)
            and all(m._inv_freq64.dtype == torch.float64 for m in rotary)):
        raise SystemExit('Effective float64 patch gate failed')
    lt = np.linspace(*args.log_temperature, args.cells)
    n = len(lt); d = loop.E.shape[1]
    X = loop.prepare(np.zeros((2*n, d)))
    T = np.tile(10.0**lt, 2)
    meta = {'prefix': PREFIX, 'prefix_text': loop.prefix_text, 'prefix_ids': loop.prefix_ids.tolist(),
            'window': args.window, 'log10_temperature': lt.tolist(), 'transient': args.transient,
            'record': args.record, 'eps_relative': args.eps, 'threads': args.threads,
            'model_revision': loop.sm.engine.revision, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'tool_sha256': sha(__file__)}
    (out/'meta.json').write_text(json.dumps(meta, indent=1))
    tic = time.monotonic()
    with torch.no_grad():
        for k in range(args.transient):
            X, _ = loop.step(X, T)
            if k % 200 == 199:
                print(json.dumps({'phase': 'transient', 'step': k+1, 'seconds': round(time.monotonic()-tic)}), flush=True)
        rng = np.random.default_rng(7)
        base = X[:n]
        u = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype)
        scale = base.flatten(1).norm(dim=1)*args.eps
        u = u/u.flatten(1).norm(dim=1)[:, None, None]*scale[:, None, None]
        X = torch.cat([base, base+u])
        lg = np.zeros(n)
        states = np.zeros((args.record, n, d)); top_id = np.zeros((args.record, n, TOPK), dtype=np.int64)
        top_p = np.zeros((args.record, n, TOPK)); ent = np.zeros((args.record, n)); trace = np.zeros((args.record, n))
        sq = (loop.E**2).sum(1)
        for k in range(args.record):
            X, p = loop.step(X, T)
            a, b = X[:n], X[n:]
            diff = b-a; dist = diff.flatten(1).norm(dim=1)
            lg += np.log((dist/scale).double().numpy())
            X = torch.cat([a, a+diff*(scale/dist)[:, None, None]])
            pa = p[:n]
            states[k] = a[:, -1, :].double().numpy()
            v, i = torch.topk(pa, TOPK, dim=1)
            top_id[k] = i.numpy(); top_p[k] = v.double().numpy()
            ent[k] = (-(pa*torch.log(pa.clamp_min(1e-300))).sum(1)).double().numpy()
            mean = pa@loop.E
            trace[k] = ((pa@sq)-(mean**2).sum(1)).double().numpy()        # exact tr Cov_p(e) over the vocabulary
            if k % 120 == 119:
                print(json.dumps({'phase': 'record', 'step': k+1, 'seconds': round(time.monotonic()-tic)}), flush=True)
    np.savez_compressed(out/'sweep.npz', log10_temperature=lt, states=states, top_id=top_id, top_p=top_p,
                        entropy=ent, cov_trace=trace, lyapunov=lg/args.record)
    meta['seconds'] = time.monotonic()-tic; meta['sweep_sha256'] = sha(out/'sweep.npz')
    (out/'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({'completed': True, 'seconds': round(meta['seconds'])}), flush=True)


def analyze(args):
    from transformers import AutoTokenizer
    import torch
    run = Path(args.run)
    z = np.load(run/'sweep.npz'); meta = json.loads((run/'meta.json').read_text())
    cache = str(DATA/'models/hf/hub')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache, local_files_only=True)
    from safetensors.torch import load_file
    import glob
    f = glob.glob(cache+'/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors')[0]
    E = load_file(f)['model.embed_tokens.weight'].double().numpy()
    mids = token_ids(tok, MONTHS); mid_set = {t: k for k, t in enumerate(mids)}
    mids0 = token_ids(tok, MONTHS, space=False)
    for k, t in enumerate(mids0):
        mid_set.setdefault(t, k)
    Xm = E[mids]; c = Xm.mean(0)
    _, _, vt = np.linalg.svd(Xm-c, full_matrices=False)
    plane = vt[:2]
    lt = z['log10_temperature']; S = z['states']; tid = z['top_id']; tp = z['top_p']
    R, n, d = S.shape
    rng = np.random.default_rng(1)
    words = [i for i, t in enumerate(tok.convert_ids_to_tokens(list(range(E.shape[0]))))
             if t is not None and t.startswith('Ġ') and t[1:].isalpha() and len(t) > 3 and i not in mid_set]
    cells = []
    for j in range(n):
        lead = tid[:, j, 0]
        months = [mid_set.get(int(t)) for t in lead]
        seq = [m for m in months if m is not None]
        # state period (relative 1e-9): every one of the last L states must equal the state q updates earlier
        # (one comparison is not enough: at low temperature all comma states coincide)
        per = 0
        for q in range(1, min(args.max_period, R//3)+1):
            L = min(R-q, 3*args.max_period)
            a, b = S[R-L:, j], S[R-L-q:R-q, j]
            if np.all(np.linalg.norm(a-b, axis=1) < 1e-9*np.linalg.norm(a, axis=1)):
                per = q; break
        visited = sorted(set(seq))
        # readout covariance of the top-64 competitors against the same probabilities on random ordinary words, at
        # the updates whose most probable token is a month
        act, sh = [], []
        for k in range(R):
            if months[k] is None:
                continue
            p = tp[k, j]/tp[k, j].sum(); ids = tid[k, j]
            Ek = E[ids]; act.append(float(p@(Ek**2).sum(1)-np.sum((p@Ek)**2)))
            Er = E[rng.choice(words, TOPK, replace=False)]
            sh.append(float(p@(Er**2).sum(1)-np.sum((p@Er)**2)))
        act = act or [float('nan')]; sh = sh or [float('nan')]
        lam = float(z['lyapunov'][j])
        month_steps = [k for k in range(R) if months[k] is not None]
        proj = ((S[month_steps, j]-c)@plane.T).tolist() if month_steps else []
        cells.append({'log10_temperature': float(lt[j]), 'lyapunov': lam if np.isfinite(lam) else None,
                      'twin_collapsed': bool(not np.isfinite(lam)), 'state_period': per,
                      'months_visited': [MONTHS[m] for m in visited], 'month_count': len(visited),
                      'month_sequence_tail': [MONTHS[m] for m in seq[-24:]],
                      'text_tail': ''.join(tok.decode([int(t)]) for t in lead[-48:]),
                      'cov_trace_top64_actual_median': float(np.median(act)),
                      'cov_trace_top64_random_median': float(np.median(sh)),
                      'cov_trace_ratio_median': float(np.median(np.array(act)/np.array(sh))),
                      'month_steps': len(month_steps),
                      'entropy_median': float(np.median(z['entropy'][:, j])),
                      'cov_trace_exact_median': float(np.median(z['cov_trace'][:, j])),
                      'month_plane_projection': proj[-48:]})
    adj = [float(np.linalg.norm(E[mids[k]]-E[mids[(k+1) % 12]])) for k in range(12)]
    pairs = [float(np.linalg.norm(E[a]-E[b])) for a, b in rng.choice(words, (5000, 2))]
    res = {'run': str(run), 'sweep_sha256': sha(run/'sweep.npz'), 'cells': cells,
           'adjacent_month_distances': adj, 'random_pair_distance_median': float(np.median(pairs)),
           'month_plane_basis': 'top two principal axes of the 12 centered month embeddings (leading space)'}
    (Path(args.out)).mkdir(parents=True, exist_ok=False)
    (Path(args.out)/'analysis.json').write_text(json.dumps(res, indent=1))
    for cdict in cells:
        lam = 'collapsed' if cdict['twin_collapsed'] else f"{cdict['lyapunov']:+.4f}"
        print(f"{cdict['log10_temperature']:+.3f} lam {lam:>9s} per {cdict['state_period']:3d} "
              f"months {cdict['month_count']:2d} trace ratio {cdict['cov_trace_ratio_median']:.3f} "
              f"(act {cdict['cov_trace_top64_actual_median']:.2e})  {cdict['text_tail'][-44:]!r}")
    print('adjacent month distance median', np.median(adj), ' random word pair median', np.median(pairs))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('static'); s.add_argument('--out', required=True); s.add_argument('--threads', type=int, default=8)
    w = sub.add_parser('sweep'); w.add_argument('--out', required=True); w.add_argument('--threads', type=int, default=16)
    w.add_argument('--window', type=int, default=2); w.add_argument('--cells', type=int, default=48)
    w.add_argument('--log-temperature', type=float, nargs=2, default=[-1.5, 0.3])
    w.add_argument('--transient', type=int, default=1200); w.add_argument('--record', type=int, default=720)
    w.add_argument('--eps', type=float, default=1e-8)
    a = sub.add_parser('analyze'); a.add_argument('--run', required=True); a.add_argument('--out', required=True)
    a.add_argument('--max-period', type=int, default=120)
    args = p.parse_args()
    {'static': static, 'sweep': sweep, 'analyze': analyze}[args.cmd](args)
