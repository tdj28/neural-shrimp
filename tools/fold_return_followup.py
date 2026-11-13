"""FGR3 local CPU execution with immutable, separately verified disk evidence."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, recurrence, sha, write_json
from fold_return_curve import reconstruct
from fold_return_line import reference_match
from fold_return_surrogate import DepartureDetector, make_model, refinement_points


def atomic_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix+'.partial')
    with open(partial, 'xb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(partial, path)


class Evidence:
    def __init__(self, out, backup):
        self.out, self.backup = Path(out), Path(backup)
        self.out.mkdir(parents=True, exist_ok=False)
        self.backup.mkdir(parents=True, exist_ok=False)
        if self.out.stat().st_dev == self.backup.stat().st_dev:
            raise ValueError('Independent filesystem required; physical disks checked separately')
        self.files = []

    def save(self, name, value):
        path = self.out/name; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        if name.endswith('.npz'):
            atomic_npz(path, **value)
        else:
            write_json(path, value)
        self.files.append(backup_file(path, self.out, self.backup))
        write_json(self.out/'retrieval.json', self.files)

    def finish(self):
        self.save('manifest.json', list(self.files))
        write_json(self.out/'complete.json', {'manifest_sha256': sha(self.out/'manifest.json'),
                                             'backup_manifest_sha256': sha(self.backup/'manifest.json')})
        backup_file(self.out/'complete.json', self.out, self.backup)


class Engine:
    def __init__(self, plan, evidence):
        from explore_llm_soft_loop import SoftLoop
        import explore_llm_phase_map as phase
        from torch.nn.attention import SDPBackend, sdpa_kernel
        self.plan, self.evidence = plan, evidence
        self.B = plan['batch_size']
        parent = Path(plan['parent_curve'])
        pp = json.loads((parent/'plan.json').read_text())
        if sha(parent/'inputs.npz') != pp['inputs_sha256']:
            raise ValueError('Parent input binding failed')
        with np.load(parent/'inputs.npz') as z:
            self.data = {k: z[k] for k in z.files}
        self.parameters = pp
        cycle = Path(plan['reference_cycle'])
        with np.load(cycle/'orbit-and-monodromy.npz') as z:
            self.M = z['monodromy']; self.orbit = z['orbit']
        self.evidence.save('parent-bindings.json', {
            str(p): sha(p) for p in (parent/'plan.json', parent/'inputs.npz',
                                     cycle/'verification.json', cycle/'orbit-and-monodromy.npz')})
        self.evidence.save('inputs.npz', self.data)
        so = pp['source']
        self.loop = SoftLoop(so['model'], so['layer'], plan['threads'], so['prompt'],
                             so['opener'], so['window'], 'float64', 'cpu')
        self.torch = self.loop.torch
        self.kernel = lambda: sdpa_kernel(SDPBackend.MATH)
        direction = phase.load_vectors(so['vectors'])[so['concept']]
        with self.kernel():
            self.loop.prepare(np.tile((so['alpha']+pp['epsilon'])*direction, (self.B, 1)))
        self.T = np.full(self.B, 10.**so['log10_temperature'])
        norms = [m for m in self.loop.model.modules() if type(m).__name__.endswith('RMSNorm')]
        rotary = [m for m in self.loop.model.modules() if type(m).__name__.endswith('RotaryEmbedding')]
        if not (norms and rotary and all(m.forward.__func__.__qualname__.startswith('keep_float64_norms.') for m in norms)
                and all(m._inv_freq64.dtype == self.torch.float64 for m in rotary)):
            raise ValueError('Effective float64 patch gate failed')
        runtime = ('fold_return_followup.py', 'fold_return_surrogate.py', 'fold_return_curve.py',
                   'fold_return_line.py', 'fold_global_return.py', 'fold_panel_data.py',
                   'explore_llm_soft_loop.py', 'explore_llm_attractors.py', 'explore_llm_phase_map.py')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        import hashlib
        for f in runtime:
            committed = subprocess.check_output(['git', 'show', f'{head}:tools/{f}'], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != sha(ROOT/'tools'/f):
                raise ValueError(f'Uncommitted runtime source: {f}')
        self.evidence.save('metadata.json', {
            'source_commit': head, 'runtime_sources': {f: sha(ROOT/'tools'/f) for f in runtime},
            'plan_sha256': sha(evidence.out/'plan.json'), 'model_revision': self.loop.sm.engine.revision,
            'prefix_text': self.loop.prefix_text, 'prefix_ids': self.loop.prefix_ids.tolist(),
            'direction_sha256': sha(phase.DATA/'explore'/so['vectors']/'vectors.npz'),
            'device': 'cpu', 'dtype': str(self.loop.dtype), 'batch': self.B,
            'attention': 'SDPBackend.MATH', 'threads': self.torch.get_num_threads(),
            'torch': self.torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
            'pid': os.getpid(), 'backup': str(evidence.backup),
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        self.gates()

    def tensor(self, x):
        return self.torch.tensor(np.asarray(x)[:, None], dtype=self.torch.float64)

    def step(self, x):
        with self.torch.no_grad(), self.kernel():
            y, p = self.loop.step(self.tensor(x), self.T)
            values, ids = self.torch.topk(p, 4, dim=1)
        return y.numpy()[:, 0].copy(), ids.numpy(), values.numpy()

    def jvp(self, x, v):
        y, _, jv = self.loop.step_jvp(self.tensor(x), self.T, self.tensor(v))
        return y.numpy()[:, 0].copy(), jv.numpy()[:, 0].copy()

    def gates(self):
        d, B = self.data, self.B
        x = np.tile(d['replay_states'], (B//8, 1))
        y, _, _ = self.step(x); z, _, _ = self.step(y)
        replay = [float(np.max(abs(a-np.tile(d[k], (B//8, 1)))))
                  for a, k in ((y, 'replay_F'), (z, 'replay_G'))]
        rng = np.random.default_rng(270113)
        v = rng.normal(size=896); v /= np.linalg.norm(v)
        base = np.tile(self.orbit[0], (B, 1)); tangent = np.tile(v, (B, 1)); x = base.copy()
        for _ in range(3):
            x, tangent = self.jvp(x, tangent)
        direct = tangent[0]; expected = self.M@v
        composed = float(np.linalg.norm(direct-expected)/np.linalg.norm(expected))
        h = np.linalg.norm(base[0])*np.array([1e-3, 3e-4, 1e-4])
        x = base.copy()
        for i in range(3):
            x[2*i] += h[i]*v; x[2*i+1] -= h[i]*v
        for _ in range(3): x, _, _ = self.step(x)
        error = [float(np.linalg.norm((x[2*i]-x[2*i+1])/(2*h[i])-direct)/np.linalg.norm(direct)) for i in range(3)]
        orders = [float(np.log(error[i]/error[i+1])/np.log(h[i]/h[i+1])) for i in range(2)]
        gate = {'saved_replay_max_abs': replay, 'composed_JVP_relative_error': composed,
                'finite_difference_h': h.tolist(), 'finite_difference_relative_errors': error,
                'orders': orders, 'passed': max(replay)<1e-11 and composed<1e-9 and min(error)<1e-5
                and (min(error)<1e-9 or all(1.5<q<2.5 for q in orders))}
        self.evidence.save('precision-gate.json', gate)
        if not gate['passed']: raise ValueError(f'Precision gate failed: {gate}')

    def run_batch(self, seeds, name, horizon, stop_on_departure):
        n = len(seeds); B = self.B
        if not 0<n<=B: raise ValueError('Invalid active batch size')
        x = np.tile(self.orbit[0], (B, 1)); x[:n] = seeds
        self.evidence.save(name+'/initial.npz', {'states': x, 'active_rows': np.arange(n)})
        d = self.data; U = self.parameters['width']
        detectors = [DepartureDetector(U, self.parameters['transverse_limit']) for _ in range(n)]
        events = [[] for _ in range(n)]; outcomes = [None]*n; valid = np.zeros(B, dtype=int)
        rolling = [x.copy()]; states = []; top_ids = []; top_probs = []
        started = time.monotonic()
        for step in range(1, horizon+1):
            active = np.zeros(B, dtype=bool); active[:n] = [o is None for o in outcomes]
            try:
                new, ids, probs = self.step(x)
            except BaseException:
                if states:
                    self.evidence.save(f'{name}/interrupted-prefix-{step-1:05d}.npz', {
                        'states': np.array(states), 'top_ids': np.array(top_ids),
                        'top_probabilities': np.array(top_probs), 'valid_through': valid.copy()})
                raise
            x[active] = new[active]; valid[active] = step
            if not np.isfinite(x).all():
                self.evidence.save(f'{name}/nonfinite-{step:05d}.npz', {'states': x, 'valid_through': valid})
                raise ValueError('Nonfinite network state')
            delta = x[:n]-d['centre']; v = delta@d['left']
            transverse = np.linalg.norm(delta-v[:, None]*d['right'], axis=1)
            for i in np.flatnonzero(active[:n]):
                event = detectors[i].update(step, v[i], transverse[i])
                if event:
                    events[i].append(event)
                    if stop_on_departure:
                        outcomes[i] = {'kind': 'return', 'updates': step, 'next_q': event['v_over_U']}
            rolling.append(x.copy()); rolling = rolling[-193:]
            states.append(x.copy()); top_ids.append(ids); top_probs.append(probs)
            if step >= 256 and step % 128 == 0:
                hist = np.array(rolling)
                for i in range(n):
                    if outcomes[i] is not None: continue
                    match = reference_match(hist[:, i], self.orbit)
                    if match['passed']:
                        outcomes[i] = {'kind': 'reference_p3', 'updates': step, **match}
                    else:
                        rec = recurrence(hist[:, i])
                        if rec: outcomes[i] = {'kind': 'other_recurrence_candidate', 'updates': step, **rec}
            finished = all(o is not None for o in outcomes)
            timeout = time.monotonic()-started > self.plan['local_wall_seconds_per_stage']
            if step % 128 == 0 or finished or step == horizon or timeout:
                lo = step-len(states)+1
                self.evidence.save(f'{name}/updates-{lo:05d}-{step:05d}.npz', {
                    'states': np.array(states), 'top_ids': np.array(top_ids),
                    'top_probabilities': np.array(top_probs), 'valid_through': valid.copy()})
                states.clear(); top_ids.clear(); top_probs.clear()
                status = {'batch': name, 'step': step, 'active': sum(o is None for o in outcomes),
                          'seconds': time.monotonic()-started, 'verified_files': len(self.evidence.files)}
                write_json(self.evidence.out/'status.json', status); print(json.dumps(status), flush=True)
            if finished or timeout: break
        outcomes = [o if o is not None else {'kind': 'unresolved_at_horizon' if step==horizon else 'wall_time_limit',
                                             'updates': int(valid[i])} for i, o in enumerate(outcomes)]
        result = {'outcomes': outcomes, 'events': events, 'valid_through': valid.tolist(),
                  'seconds': time.monotonic()-started}
        self.evidence.save(name+'/result.json', result)
        return result


def execute(args):
    plan = json.loads(Path(args.plan).read_text())
    evidence = Evidence(args.out, args.backup)
    evidence.save('plan.json', plan)
    started = time.monotonic()
    try:
        engine = Engine(plan, evidence)
        if args.mode == 'pilot':
            x = np.tile(engine.orbit[0], (engine.B, 1))
            t = time.monotonic(); states = []
            for _ in range(plan['pilot']['normal_updates']):
                x, _, _ = engine.step(x); states.append(x.copy())
            normal = (time.monotonic()-t)/len(states)
            v = np.random.default_rng(79).normal(size=x.shape); v /= np.linalg.norm(v, axis=1)[:, None]
            t = time.monotonic(); growth = []
            for _ in range(plan['pilot']['tangent_updates']):
                x, v = engine.jvp(x, v); norm = np.linalg.norm(v, axis=1)
                growth.append(np.log(norm)); v /= norm[:, None]
            tangent = (time.monotonic()-t)/len(growth)
            evidence.save('pilot.npz', {'states': np.array(states), 'growth': np.array(growth)})
            evidence.save('result.json', {'normal_seconds_per_update': normal,
                                          'jvp_seconds_per_update': tangent,
                                          'wall_seconds': time.monotonic()-started})
        elif args.mode == 'train':
            records = []; specification = plan['training']
            curve = {'mid': float(engine.data['curve_mid']), 'half': float(engine.data['curve_half']),
                     'coef': engine.data['curve_coef']}
            grid = np.linspace(*specification['domain'], specification['initial_points'])
            for round_id in range(specification['refinement_rounds']+1):
                if round_id:
                    grid = np.array(refinement_points(records, specification['new_points_per_round']))
                evidence.save(f'round-{round_id:02d}-selection.json', {'q': grid.tolist(),
                               'prior_rows': len(records), 'rule': specification['rule']})
                for batch, lo in enumerate(range(0, len(grid), engine.B)):
                    q = grid[lo:lo+engine.B]
                    seeds = reconstruct(q*engine.parameters['width'], curve, engine.data['centre'], engine.data['right'])
                    name = f'round-{round_id:02d}-batch-{batch:03d}'
                    result = engine.run_batch(seeds, name, specification['max_updates'], True)
                    for i, (value, outcome) in enumerate(zip(q, result['outcomes'])):
                        records.append({'q': float(value), 'batch': name, 'row': i, **outcome})
                    if time.monotonic()-started > plan['local_wall_seconds_per_stage']:
                        raise TimeoutError('Training stage wall bound reached; all completed batches preserved')
                evidence.save(f'round-{round_id:02d}-model.json', make_model(records))
            evidence.save('model.json', make_model(records))
            evidence.save('result.json', {'training_rows': len(records), 'wall_seconds': time.monotonic()-started})
        elif args.mode == 'validate':
            if not args.model: raise ValueError('A frozen autonomous model artifact is required')
            model = Path(args.model)
            evidence.save('frozen-model.json', json.loads(model.read_text()))
            line = json.loads((Path(plan['parent_line'])/'plan.json').read_text())
            prior = Path(line['prior_path'])
            with np.load(prior/'inputs.npz') as z:
                a, b = z['seeds'][[64, 77]]
            n = plan['validation']['rows']
            if n != engine.B: raise ValueError('This design requires one complete fixed validation batch')
            rng = np.random.default_rng(plan['validation']['seed'])
            s = (np.arange(n)+rng.random(n))/n
            seeds = a+s[:, None]*(b-a)
            evidence.save('selection.json', {'s': s.tolist(), 'parent_inputs': str(prior/'inputs.npz'),
                                               'parent_sha256': sha(prior/'inputs.npz'), 'model_sha256': sha(model)})
            result = engine.run_batch(seeds, 'trajectories', plan['validation']['max_updates'], False)
            evidence.save('result.json', result)
        else:
            raise ValueError(args.mode)
        evidence.finish()
        print(json.dumps({'completed': args.mode, 'wall_seconds': time.monotonic()-started,
                          'output': str(evidence.out)}), flush=True)
    except BaseException as exc:
        write_json(evidence.out/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                               'seconds': time.monotonic()-started})
        backup_file(evidence.out/'failure.json', evidence.out, evidence.backup)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['pilot', 'train', 'validate'])
    p.add_argument('--plan', default=str(ROOT/'docs/fold-return-autonomous-plan.json'))
    p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    p.add_argument('--model')
    execute(p.parse_args())
