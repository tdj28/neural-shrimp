"""Owned fixed-checkpoint frozen-curvature response pilot; consumed data only."""
import os
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for k in THREADS:
    os.environ[k] = '1'
import argparse, hashlib, json, resource, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem
from neural_shrimp.tiny_lm_response import run_case, profile_forecast
BASE = Path('/data2/neural-shrimp-data/explore/tiny-lm-broader-data')
PARENTS = {'profile': (BASE / 'attempt-001-profile', 'ce56de837203182187912654b9ac0200096366a15d850ed7e917fb903837d2cd'), 'target': (BASE / 'attempt-002-collection', '2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d')}
SOURCES = ['src/neural_shrimp/tiny_lm.py', 'src/neural_shrimp/tiny_lm_response.py', 'tools/explore_tiny_lm_response.py', 'tools/launch_tiny_lm_response.py', 'tests/test_tiny_lm_response.py', 'docs/plans/explore-tiny-lm-frozen-curvature-candidate.md']
NATIVE_SHA = '27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8'
CAP = 256 * 1024 ** 2

def need(v, m):
    if not v:
        raise ValueError(m)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while (b := f.read(1024 ** 2)):
            h.update(b)
    return h.hexdigest()

def entry(p):
    return dict(bytes=p.stat().st_size, sha256=sha(p))

def load(p):
    return json.loads(p.read_bytes())

def inventory(root, pin):
    need(sha(root / 'manifest.json') == pin, 'pinned manifest')
    x = load(root / 'manifest.json')['files']
    if isinstance(x, list):
        x = {v['path']: {k: v[k] for k in ('bytes', 'sha256')} for v in x}
    need(set(x) == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}, 'complete inventory')
    for n, v in x.items():
        p = root / n
        need(not Path(n).is_absolute() and '..' not in Path(n).parts and (not p.is_symlink()) and (entry(p) == {k: v[k] for k in ('bytes', 'sha256')}), 'full input hashes')
    return x

class Store:

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.used = 0

    def write(self, name, data):
        need(self.used + len(data) <= CAP, '256MiB retained cap')
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.used += len(data)

    def json(self, name, value):
        self.write(name, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())

    def array(self, name, value):
        import io
        b = io.BytesIO()
        np.save(b, value, allow_pickle=False)
        self.write(name + '.npy', b.getvalue())

    def finish(self, result):
        self.json('result.json', result)
        self.json('manifest.json', dict(files={str(p.relative_to(self.root)): entry(p) for p in self.root.rglob('*') if p.is_file()}))

class Calls:

    def __init__(self, store, seconds, profile):
        self.store = store
        self.deadline = time.monotonic() + seconds
        self.cap = dict(gradient=4 if profile else 580, hvp=4 if profile else 256, ce=4 if profile else 96)
        self.count = {k: 0 for k in self.cap}
        self.times = {k: [] for k in self.cap}
        self.events = []

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('fixed local response deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024 ** 2:
            raise MemoryError('4GiB worker RSS')

    def call(self, name, fun, *args):
        self.check()
        need(self.count[name] < self.cap[name], 'fixed operation cap ' + name)
        self.count[name] += 1
        t = time.monotonic()
        value = fun(*args)
        elapsed = time.monotonic() - t
        self.times[name].append(elapsed)
        self.events.append(dict(operation=name, index=self.count[name], seconds=elapsed))
        return value

def select(parent, mode, seed, arm, step):
    folder = parent / f'seed-{seed}-arm-{arm}'
    r = load(folder / 'result.json')
    need(r['seed'] == seed and r['arm'] == arm and (r['outcome'] == 'completed'), 'fixed completed development state')
    chunk = folder / f'states-{step // 50 * 50:06d}-{step // 50 * 50 + 49:06d}.npy'
    a = np.load(chunk, allow_pickle=False)
    need(a.shape == (50, 15776) and a.dtype == np.float32 and np.isfinite(a).all(), 'retained native state chunk')
    theta = a[step % 50].astype(np.float64)
    data = load(parent / 'data.json')
    need(data['data_identity_sha256'] == 'a65b88c04f7de18031005672a345af8eaf929ff75cf67e1d0c401b02c7f6f6aa' and len(data['vocabulary']) == 66, 'same consumed data')
    key = 'control' if arm == 8 else 'expanded'
    with np.load(parent / 'data-tokens.npz', allow_pickle=False) as z:
        x = z[key + '_X'].copy()
        y = z[key + '_Y'].copy()
    need(x.shape == y.shape == (arm, 32) and x.dtype == y.dtype == np.int64, 'fixed training tokens only')
    expected = np.asarray(data['roles'][key]['window_token_ids'], dtype=np.int64)
    need(np.array_equal(x, expected[:, :-1]) and np.array_equal(y, expected[:, 1:]), 'retained exact token identities')
    eta = r['eta']
    need(np.isfinite(eta) and eta > 0, 'same finite physical rate')
    return (theta, x, y, eta, dict(seed=seed, arm=arm, step=step, eta=eta, state_chunk=str(chunk), state_chunk_identity=entry(chunk), state_float32_sha256=hashlib.sha256(a[step % 50].tobytes()).hexdigest(), data_identity=data['data_identity_sha256'], parent_mode=mode))

def main(args):
    need(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = {n: entry(ROOT / n) for n in SOURCES}
    for n, item in identity.items():
        need(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + n], cwd=ROOT)).hexdigest() == item['sha256'], 'committed source ' + n)
    need(identity[SOURCES[0]]['sha256'] == NATIVE_SHA, 'unchanged native transformer')
    store = Store(args.out)
    limit = 60 if args.mode == 'profile' else 300
    start = time.monotonic()
    calls = Calls(store, limit, args.mode == 'profile')

    def stop(*_):
        raise TimeoutError('owned response signal')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, limit)
    result = dict(status='running', passed=False, mode=args.mode, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_identity=identity, torch=torch.__version__, numpy=np.__version__, dtype='float64', device='cpu', threads=dict.fromkeys(THREADS, '1'), cases=[], model_parameters=15776, protected_outcomes_evaluated=False, profile_costs=[])
    try:
        store.json('runtime.json', result)
        for name in SOURCES:
            store.write('source/' + name, (ROOT / name).read_bytes())
        parent, pin = PARENTS[args.mode]
        inventory(parent, pin)
        store.json('input-binding.json', dict(parent_root=str(parent), parent_manifest=pin, scope='Only saved training-token arrays and fixed consumed states used; no probe/validation evaluation.'))
        if args.mode == 'target':
            need(args.profile is not None and args.profile_manifest is not None, 'completed measured profile')
            inventory(args.profile, args.profile_manifest)
            prior = load(args.profile / 'result.json')
            need(prior['passed'] and prior['mode'] == 'profile' and (prior['source_identity'] == identity) and (prior['torch'] == torch.__version__) and (prior['numpy'] == np.__version__), 'source/library-qualified profile')
            forecast = profile_forecast(prior['profile_costs'])
            need(forecast == prior['forecast_seconds'] and forecast < 300, 'recomputed measured300s cap')
            need(prior['operation_attempted'] == prior['operation_returned'] == dict(gradient=4, hvp=4, ce=4), 'complete measured operation inventory')
            store.json('profile-reference.json', dict(root=str(args.profile), manifest=args.profile_manifest, forecast_seconds=prior['forecast_seconds']))
        seeds = [28160] if args.mode == 'profile' else [28161, 28162]
        step = 50 if args.mode == 'profile' else 600
        for seed in seeds:
            for arm in [8, 256]:
                calls.check()
                theta, x, y, eta, binding = select(parent, args.mode, seed, arm, step)
                label = f'seed-{seed}-arm-{arm}'
                store.json(label + '/inputs.json', binding)
                store.array(label + '/initial', theta)
                problem = LMProblem(TinyTransformer(66), torch.tensor(x), torch.tensor(y), ridge=0.001)

                def tensor(v):
                    return torch.as_tensor(v, dtype=torch.float64)

                def gradient(v):
                    return calls.call('gradient', lambda: problem.grad(tensor(v)).detach().numpy().copy())

                def hvp(v, w):
                    return calls.call('hvp', lambda: problem.hvp(tensor(v), tensor(w)).detach().numpy().copy())

                def ce(v):
                    return calls.call('ce', lambda: float(problem.cross_entropy(tensor(v))))
                rng = np.random.default_rng(28166)
                random = rng.standard_normal(15776)
                if args.mode == 'profile':
                    costs = {k: [] for k in calls.cap}
                    for repeat in range(2):
                        for name, fun in [('gradient', lambda: gradient(theta)), ('hvp', lambda: hvp(theta, random)), ('ce', lambda: ce(theta))]:
                            v = fun()
                            need(np.isfinite(v).all(), 'finite profile result')
                            costs[name].append(calls.times[name][-1])
                            store.array(label + f'/profile-{name}-{repeat}', np.asarray(v))
                    result['profile_costs'].append(dict(arm=arm, times=costs))
                else:
                    answer = run_case(theta, eta, gradient, hvp, ce, 0.001, random, lambda n, v: store.array(label + '/' + n, v), calls.check)
                    answer.update(seed=seed, arm=arm, step=step)
                    store.json(label + '/outcome.json', answer)
                    result['cases'].append(answer)
        if args.mode == 'profile':
            need(calls.count == calls.cap, 'complete profile operation inventory')
            forecast = profile_forecast(result['profile_costs'])
            result.update(status='profile_complete', passed=bool(forecast < 300), forecast_seconds=forecast)
        else:
            need(calls.count == calls.cap, 'complete declared operation inventory')
            selected = [r for c in result['cases'] if c['arm'] == 256 for r in c['rows'] if r['direction'] == 'gradient' and r['horizon'] == 16]
            need(len(selected) == 2, 'two fixed expanded cases')
            result.update(status='response_complete', passed=True, narrow_frozen_curvature_inadequacy=bool(all((r['mechanism_qualified'] and r['frozen_vector_error'] > 0.2 for r in selected))), interpretation='Two consumed development seeds; transported match validates calculus, not utility. No controller, orbit, chaos, inference quality or novelty claim.')
    except BaseException as e:
        result.update(status='failed_or_interrupted', error=type(e).__name__ + ': ' + str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(elapsed_seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, operation_caps=calls.cap, operation_attempted=calls.count, operation_returned={k: len(v) for k, v in calls.times.items()})
        result['passed'] = bool(result['passed'] and result['elapsed_seconds'] < limit and (result['peak_rss_kib'] < 4 * 1024 ** 2))
        store.json('events.json', calls.events)
        store.finish(result)
    print(json.dumps({k: result.get(k) for k in ['status', 'passed', 'elapsed_seconds', 'forecast_seconds', 'narrow_frozen_curvature_inadequacy', 'error']}))
    return 0 if result['passed'] else 2
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['profile', 'target'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--profile-manifest')
    raise SystemExit(main(p.parse_args()))
