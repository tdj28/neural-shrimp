#!/usr/bin/env python3
"""Fixed signed-half flip continuation, with native identity at every accepted step."""
import argparse
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('signed_critical', ROOT / 'tools/explore_clark_rnn_n12_signed_critical.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S, R, M = C.S, C.R, C.M
PIN = '56c5cf552ebeee4400962b7407662d0edc65835ec0e4ba8b3ed44c0588ad3a63'
TAGS = ('g16-dir-1-step80', 'g25-dir-1-step100')
STEPS = 32
EXTRA = C.EXTRA + ['tools/explore_clark_rnn_n12_flip_branch.py', 'tools/launch_clark_rnn_n12_flip_branch.py', 'tests/test_clark_rnn_n12_flip_branch.py', 'docs/plans/clark-rnn-n12-flip-branch.md']

def need(ok, message):
    if not ok:
        raise ValueError(message)

def gain_derivative(a, q):
    """Derivative of [G-x,(DG+I)q,(q.q-1)/2] at fixed state/leak/q."""
    return np.r_[a['first'][:, 12], a['second'][:, :12, 12] @ q, 0.]

def flip_coefficients(gm, second, param, q, w, third):
    B = second[:, :12, :12]
    bqq = np.einsum('ijk,j,k->i', B, q, q)
    rhs = np.column_stack([-param, bqq])
    sol = np.linalg.solve(gm - np.eye(12), rhs)
    alpha = float(w @ (np.einsum('ijk,j,k->i', B, q, sol[:, 0]) + second[:, :12, 13] @ q))
    cubic = float(w @ third / 6 - w @ np.einsum('ijk,j,k->i', B, q, sol[:, 1]) / 2)
    return alpha, cubic, rhs, sol, bqq

def native_join(a, b, da, db):
    scale = max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    error = float(np.max(np.abs(a-b)) / scale)
    return {'native_cycle_error': error, 'delta_error': abs(float(da-db)),
            'passed': bool(np.isfinite(error) and error <= 1e-7 and abs(da-db) <= 1e-9)}

def cubic_crossings(rows):
    # This labels numerical brackets only; zero is retained without a nondegeneracy claim.
    return [{'left': i-1, 'right': i, 'gains': [rows[i-1]['gain'], rows[i]['gain']],
             'cubics': [rows[i-1]['cubic'], rows[i]['cubic']]}
            for i in range(1, len(rows)) if rows[i-1]['cubic'] * rows[i]['cubic'] <= 0]

class Work(C.Work):
    def check(self):
        if self.interrupted:
            raise InterruptedError('SIGTERM')
        if time.monotonic() - self.start > 120:
            raise TimeoutError('120s ceiling')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 448*1024:
            raise MemoryError('448MiB RSS')
        if self.calls > 1323:
            raise RuntimeError('1323 combined calls')

    def it(self, x, j, g, d, tag):
        if self.native_calls >= 256:
            raise RuntimeError('256 native calls')
        return super().it(x, j, g, d, tag)

    def half(self, z, j, g, sign, tag):
        self.check()
        if self.augmented_calls >= 1024:
            raise RuntimeError('1024 half calls')
        self.calls += 1
        self.augmented_calls += 1
        t = time.monotonic()
        h, b, a = C.critical_fixed_gain(z, j, g, sign, 3)
        self.emit('signed_half', dict(a, h=h, dh=b, sign=sign, tag=tag))
        return h, b, a, time.monotonic()-t

    def third(self, x, q, j, g, d, tag):
        self.check()
        if self.third_calls >= 43:
            raise RuntimeError('43 third returns')
        self.calls += 1
        self.third_calls += 1
        t = time.monotonic()
        c, stages = C.third_directional(x, q, j, g, d, 3)
        self.emit('signed_third', {'tag': tag, 'theta': x, 'direction': q, 'gain': g,
                                 'delta': d, 'third': -c, 'stages': stages})
        return -c, time.monotonic()-t

    def predict(self, z, b, a, dg, tag):
        rhs = -gain_derivative(a, z[13:])
        tangent = np.linalg.solve(b, rhs)
        need(np.isfinite(tangent).all(), 'finite critical tangent')
        self.emit('critical_predictor', {'tag': tag, 'matrix': b, 'rhs': rhs,
                                       'tangent': tangent, 'gain_step': dg, 'physical': z})
        return z + dg*tangent

    def correct_critical(self, guess, anchor, j, g, tag):
        z = guess.copy()
        scale = max(1., float(np.max(np.abs(anchor[:12]))))
        for k in range(12):
            need(np.isfinite(z).all() and .6 < z[12] < 1 and abs(z[12]-anchor[12]) <= .02
                 and np.max(np.abs(z[:12]-anchor[:12])) <= .1*scale
                 and z[13:] @ anchor[13:] > 0 and np.linalg.norm(z[13:]-anchor[13:]) < .5,
                 'native state/leak/continued eigenvector guard')
            h, b, a, _ = self.half(z, j, g, -1, tag)
            need(np.isfinite(h).all() and np.isfinite(b).all(), 'finite critical equations')
            if np.max(np.abs(h)) < 1e-12:
                return z, h, b, a
            need(k < 11, 'critical iteration limit')
            direction = np.linalg.solve(b, -h)
            self.emit('critical_solve', {'tag': tag, 'iteration': k, 'matrix': b,
                                        'rhs': -h, 'direction': direction})
            z += direction
        raise AssertionError('unreachable')

    def qualify(self, z, h, b, a, j, g, tag):
        root = self.native(z[:12], j, g, float(z[12]), tag+'-native')
        adjacent = self.native(root['cycle'][1], j, g, float(z[12]), tag+'-adjacent')
        gm, q = a['monodromy'], z[13:]
        e, v = np.linalg.eig(gm)
        le, lv = np.linalg.eig(gm.T)
        idx = int(np.argmin(abs(e+1)))
        w = lv[:, int(np.argmin(abs(le+1)))]
        self.emit('critical_eigensystems', {'tag': tag, 'matrix': gm, 'eigenvalues': e,
                                          'eigenvectors': v, 'left_eigenvalues': le, 'left_eigenvectors': lv})
        need(max(abs(w.imag)) < 1e-9 and abs(w.real @ q) > 1e-8, 'real nonorthogonal left vector')
        w = w.real/(w.real @ q)
        other = np.delete(e, idx)
        third, _ = self.third(z[:12], q, j, g, float(z[12]), tag+'-cubic')
        alpha, cubic, rhs, sol, bqq = flip_coefficients(gm, a['second'], a['first'][:, 13], q, w, third)
        self.emit('criticality_solves', {'tag': tag, 'matrix': gm-np.eye(12), 'rhs': rhs,
                                        'solution': sol, 'left': w, 'bqq': bqq, 'third': third})
        half = float(np.max(abs(root['cycle']+np.roll(root['cycle'], 3, axis=0)))/max(1., np.max(abs(root['cycle']))))
        square = float(np.max(abs(root['monodromy']-gm@gm))/max(1., np.max(abs(root['monodromy']))))
        eigen_res = float(np.max(abs(gm@v-v*e[None, :]))/max(1., np.max(abs(gm))))
        left_res = float(max(abs(w@gm+w)))
        native_spectra = []
        for context in (root, adjacent):
            values = context['eigenvalues']
            k = int(np.argmin(abs(values-1)))
            rest = np.delete(values, k)
            native_spectra.append({'critical_error': float(abs(values[k]-1)),
                                   'other_plus_distance': float(min(abs(rest-1))),
                                   'complement_radius': float(max(abs(rest)))})
        gates = {'native6': root['passed'], 'native6_adjacent': adjacent['passed'],
                 'both_native_critical_spectra': all(p['critical_error'] < 1e-7 and p['other_plus_distance'] > 1e-4 and p['complement_radius'] < 1-1e-5 for p in native_spectra),
                 'G_fixed': bool(max(abs(a['endpoint']-z[:12])) < 1e-11),
                 'G_simple_minus': bool(abs(e[idx]+1) < 1e-7 and min(abs(other+1)) > 1e-4),
                 'G_complement': bool(max(abs(other)) < 1-1e-5),
                 'G_eigensystems': bool(eigen_res < 1e-10 and left_res < 1e-7),
                 'half_wave': half < 1e-8, 'native_square': square < 1e-7,
                 'transverse_delta': bool(np.isfinite(alpha) and abs(alpha) > 1e-5),
                 'finite_cubic': bool(np.isfinite(cubic))}
        row = {'tag': tag, 'gain': float(g), 'delta': float(z[12]), 'cubic': cubic, 'alpha_delta': alpha,
               'gates': gates, 'passed': all(gates.values()), 'residual': float(max(abs(h))),
               'half_wave_error': half, 'square_error': square, 'native_phase_spectra': native_spectra,
               'G_complement_radius': float(max(abs(other))), 'G_eigen_residual': eigen_res, 'G_left_residual': left_res}
        self.emit('branch_point', dict(row, physical=z, cycle=root['cycle'], native_monodromy=root['monodromy'],
                                      G_monodromy=gm, left_vector=w))
        need(row['passed'], 'native and signed-half qualification')
        return row, root

def verify_input(parent):
    need(S.sha(parent/'manifest.json') == PIN, 'signed critical outer pin')
    inv = json.loads((parent/'manifest.json').read_text())['files']
    need(set(inv) == {str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file() and p != parent/'manifest.json'}, 'complete inventory')
    for name, item in inv.items():
        p = parent/name
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink()
             and p.stat().st_size == item['bytes'] and S.sha(p) == item['sha256'] and not p.stat().st_mode & 0o222, 'immutable input hash')
    binding = json.loads((parent/'run/input-binding.json').read_text())
    inp = R.RAW/'n12-p6-refinement-001/run/inputs.npz'
    need(S.sha(inp) == binding['matrix_input_sha256'], 'same native matrix')
    with np.load(inp) as f:
        j = f['j'].copy()
    seeds = []
    for tag in TAGS:
        paths = [p for p in (parent/'run').glob('*signed_critical_result.json') if json.loads(p.read_text())['tag'] == tag]
        need(len(paths) == 1, 'unique fixed critical root')
        meta = json.loads(paths[0].read_text())
        need(meta['status'] == 'numerically_qualified_signed_half_critical' and meta['sign'] == -1 and all(meta['gates'].values()), 'qualified signed flip input')
        arr = paths[0].with_name(meta['array_file'])
        need(S.sha(arr) == meta['array_sha256'], 'critical array pin')
        with np.load(arr) as f:
            seeds.append((meta, f['physical'].copy(), f['cycle'].copy()))
    return j, seeds, binding

def main(out):
    identity = S.source()
    identity['sha256'].update({p: S.sha(ROOT/p) for p in EXTRA})
    need(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and all(os.environ.get(k) == '1' for k in R.M.THREADS), 'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS, (1024**3,)*2)
    out.mkdir(parents=True, exist_ok=False)
    S.ACTIVE_ROOT, S.RAW_CAP = out, 128*1024**2
    w, booked = Work(out), [0]
    signal.signal(signal.SIGTERM, lambda *_: setattr(w, 'interrupted', True))
    def reserve(path, size, final=False):
        old = Path(path).stat().st_size if Path(path).exists() else 0
        new = booked[0]-old+size
        if new > (S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):
            raise OSError('prewrite raw cap')
        booked[0] = new
    S.reserve = reserve
    S.save(out/'runtime-start.json', {'source': identity, 'pid': os.getpid(), 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'threads': {k: os.environ[k] for k in R.M.THREADS}, 'started_unix': time.time(), 'python': sys.version, 'numpy': np.__version__})
    result = {'status': 'incomplete', 'points': [], 'associations': []}
    try:
        need(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024 >= 1024**3, '1GiB headroom')
        j = np.eye(12)*.2+np.ones((12, 12))*.025
        x, q = np.arange(1, 13)/20, np.eye(12)[0]
        times = {k: [] for k in ('native', 'half', 'eigen', 'solve', 'third')}
        sizes = {k: [] for k in times}
        def measured(kind, f):
            before, t = booked[0], time.monotonic()
            value = f()
            times[kind].append(time.monotonic()-t)
            sizes[kind].append(booked[0]-before)
            return value
        for i in range(10):
            ret = measured('native', lambda: w.it(x, j, 2., .8, 'synthetic_profile'))
            half = measured('half', lambda: w.half(np.r_[x, .8, q], j, 2., -1, 'synthetic_profile'))
            def eig():
                e, v = np.linalg.eig(ret[1])
                w.emit('profile_eigen', {'matrix': ret[1], 'eigenvalues': e, 'eigenvectors': v})
            def solve():
                matrix, rhs = half[1], -half[0]
                sol = np.linalg.solve(matrix, rhs)
                w.emit('profile_solve', {'matrix': matrix, 'rhs': rhs, 'solution': sol})
            measured('eigen', eig)
            measured('solve', solve)
            measured('third', lambda: w.third(x, q, j, 2., .8, 'synthetic_profile'))
        t, b = ({k: max(v) for k, v in vals.items()} for vals in (times, sizes))
        seconds = 2*(256*(t['native']+4*t['eigen'])+1024*(t['half']+3*t['solve'])+43*t['third'])+15
        raw = 256*(2*b['native']+4*b['eigen']+4096)+1024*(b['half']+3*b['solve']+4096)+43*(b['third']+4096)+8*1024**2
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+128*1024**2
        gate = all(np.isfinite(v) and v > 0 for a in times.values() for v in a) and seconds <= 120 and raw <= 128*1024**2 and peak <= 448*1024**2
        S.save(out/'profile.json', {'times': times, 'bytes': sizes, 'projected_seconds': seconds, 'projected_raw_bytes': raw, 'projected_peak_bytes': peak, 'gate': bool(gate), 'target_unopened': True})
        need(gate, 'measured resource admission')
        parent = R.RAW/'n12-p6-signed-critical-001'
        j, seeds, binding = verify_input(parent)
        first, last = seeds
        gains = np.linspace(first[0]['gain'], last[0]['gain'], STEPS+1)
        need(gains[0] < gains[-1], 'ordered fixed endpoints')
        S.save(out/'input-binding.json', {'parent_manifest': PIN, 'matrix_input_sha256': binding['matrix_input_sha256'], 'tags': TAGS, 'gains': gains.tolist(), 'steps': STEPS, 'existing_both_start_context': 'fixed original panel and boundary003 pins inherited through parent; no new basin sampling'})
        z, h, jac, a = w.correct_critical(first[1], first[1], j, gains[0], 'point00')
        row, native = w.qualify(z, h, jac, a, j, gains[0], 'point00')
        start_join = native_join(native['cycle'], first[2], z[12], first[1][12])
        result['start_join'] = start_join
        need(start_join['passed'], 'unchanged native start root')
        result['points'].append(row)
        S.save(out/'progress.json', result)
        for i, g in enumerate(gains[1:], 1):
            guess = w.predict(z, jac, a, g-gains[i-1], f'step{i:02d}')
            nz, nh, njac, na = w.correct_critical(guess, z, j, g, f'point{i:02d}')
            row, newnative = w.qualify(nz, nh, njac, na, j, g, f'point{i:02d}')
            forward = M.phase_match(native['cycle'], newnative['cycle'])
            result['pending'] = {'point': row, 'forward': forward}
            S.save(out/'progress.json', result)
            need(forward['passed'], 'native forward phase association')
            reverse_guess = w.predict(nz, njac, na, gains[i-1]-g, f'reverse{i:02d}')
            rz, rh, rb, ra = w.correct_critical(reverse_guess, nz, j, gains[i-1], f'reverse{i:02d}')
            reverse_native = w.native(rz[:12], j, gains[i-1], float(rz[12]), f'reverse{i:02d}-native')
            reverse = native_join(native['cycle'], reverse_native['cycle'], z[12], rz[12])
            q_error = float(np.linalg.norm(rz[13:]-z[13:]))
            association = {'index': i, 'forward': forward, 'reverse': reverse, 'reverse_q_error': q_error, 'reverse_native_passed': reverse_native['passed']}
            w.emit('branch_association', dict(association, previous_physical=z, current_physical=nz, reverse_physical=rz, previous_cycle=native['cycle'], current_cycle=newnative['cycle'], reverse_cycle=reverse_native['cycle']))
            need(reverse['passed'] and reverse_native['passed'] and q_error <= 1e-7, 'full native/eigenvector reverse join')
            result['points'].append(row)
            result['associations'].append(association)
            result.pop('pending', None)
            z, jac, a, native = nz, njac, na, newnative
            S.save(out/'progress.json', result)
        endpoint = native_join(native['cycle'], last[2], z[12], last[1][12])
        result['endpoint_join'] = endpoint
        need(endpoint['passed'], 'independent saved native endpoint join')
        result['cubic_brackets'] = cubic_crossings(result['points'])
        need(len(result['points']) == 33 and len(result['associations']) == 32, 'complete fixed branch')
        w.check()
        result['status'] = 'completed_numerical_branch'
    except BaseException as e:
        result.update(status='failed_or_interrupted', error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      native_calls=w.native_calls, half_calls=w.augmented_calls, third_calls=w.third_calls, total_calls=w.calls,
                      scope='Numerical native-associated signed-half critical branch only; cubic brackets are not a generalized-flip proof. No interval proof, native12/24 daughter join, full shrimp or utility claim.')
        S.save(out/'result.json', result, final=True)
        S.manifest(out)
    return 0 if result['status'] == 'completed_numerical_branch' else 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    sys.exit(main(parser.parse_args().out))
