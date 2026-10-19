"""New stdlib mocked controls for the isolated seed-retention integration only.

These controls load function definitions from the new caller AST, not its NumPy
import or entry point. The unchanged capture helper is loaded with stdlib only.
All tensor/NumPy serialization/producer/family/trace objects below are synthetic;
no old fixture, NumPy, Torch, producer, model, original array, Git or host query
is imported or executed. Tiny fresh files test retention semantics, not the NPY
format, mathematical qualification, model precision or execution provenance.
"""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'seeded_branch_002.py'
OLD = HERE / 'seeded_branch.py'
HELPER = HERE / 'seed_phase_capture.py'
spec = importlib.util.spec_from_file_location('seed_phase_capture', HELPER)
capture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_module)


class Finite:
    def __init__(self, value): self.value = value
    def all(self): return self
    def item(self): return self.value
    def __bool__(self): return self.value


class Array:
    def __init__(self, values, shape=(1, 1, 896)):
        self.values = list(values)
        self.shape = shape
        self.dtype = 'float64'


class Tensor:
    shape = (1, 1, 896)
    dtype = 'torch.float64'
    device = SimpleNamespace(type='cpu')
    def __init__(self, values=None): self.values = list(values if values is not None else [1.] * 896)
    def isfinite(self): return Finite(all(math.isfinite(x) for x in self.values))
    def norm(self): return math.hypot(*self.values)
    def __sub__(self, other): return Tensor([a-b for a,b in zip(self.values, other.values)])
    def detach(self): return self
    def clone(self): return Tensor(self.values)
    def numpy(self): return Array(self.values)


class Row:
    def __init__(self, values): self.values = list(values)
    def reshape(self, *shape):
        if shape != (1, 1, 896): raise AssertionError('unexpected seed reshape')
        return self


class States:
    ndim, shape, dtype = 2, (1, 896), 'float64'
    values = [1.] * 896
    def __len__(self): return 1
    def __getitem__(self, index):
        if index != 0: raise AssertionError('unexpected seed row')
        return Row(self.values)


class FakeNumpy:
    float64 = 'float64'
    def __init__(self):
        self.save_calls = 0
        self.fail_save_at = None
        self.saved = []
    def load(self, path, allow_pickle=False):
        if allow_pickle is not False: raise AssertionError('pickle must be disabled')
        return States()
    def isfinite(self, value):
        return Finite(all(math.isfinite(x) for x in value.values))
    def array(self, value): return value
    def save(self, stream, value, allow_pickle=False):
        self.save_calls += 1
        if allow_pickle is not False: raise AssertionError('pickle must be disabled')
        self.saved.append(value)
        if self.save_calls == self.fail_save_at:
            stream.write(b'synthetic-partial-array')
            raise OSError('synthetic serializer failure')
        stream.write(b'FAKE-NPY\n' + json.dumps(value.values).encode())


class Cell:
    def __init__(self, kind='pass', events=None):
        self.calls, self.inputs = 0, []
        self.kind = kind
        self.events = events if events is not None else []
    def f(self, value):
        self.calls += 1
        self.events.append(('f', self.calls))
        self.inputs.append(value)
        if self.kind == 'primitive': return Tensor()
        if self.calls == 5: return Tensor([2. if self.kind == 'closure' else 1.] * 896)
        return Tensor([float(self.calls + 1)] * 896)
    def return_distances(self, *args): raise AssertionError('old second seed pass forbidden')
    def points(self, *args): raise AssertionError('points pass forbidden')
    def iterate(self, *args): raise AssertionError('iterate pass forbidden')


class TraceReached(RuntimeError): pass


def definitions(np):
    names = {'sha', 'validate', 'capture_seed_return', 'main'}
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names: raise AssertionError('new caller definitions missing')
    namespace = dict(np=np, argparse=__import__('argparse'), hashlib=hashlib, json=json,
                     sys=sys, time=__import__('time'), os=os, Path=Path, __file__=str(SOURCE))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class RetentionControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='task35-new-retention-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.out = self.base / 'attempt'
        self.out.mkdir()
        self.np = FakeNumpy()
        self.ns = definitions(self.np)
        self.card = dict(seed_path=str(self.base/'synthetic-seed.bin'), seed_sha256='a'*64,
                         seed_index=0, alpha=-.119153396, alpha2=.103056949, log_temperature=-.1)
        self.imports = mock.patch.dict(sys.modules, {'seed_phase_capture': capture_module})
        self.imports.start()
        self.addCleanup(self.imports.stop)

    def capture(self, cell):
        return self.ns['capture_seed_return'](cell, Tensor(), self.out, self.card)

    def rows(self):
        return [json.loads(line) for line in (self.out/'seed-phases.jsonl').read_text().splitlines()]

    def test_01_complete_order_hash_provenance_roles_and_syncs(self):
        events = []
        cell = Cell(events=events)
        real_fsync = os.fsync
        def fsync(fd):
            events.append(('fsync', 'dir' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file'))
            return real_fsync(fd)
        with mock.patch.object(os, 'fsync', side_effect=fsync):
            distances = self.capture(cell)
        self.assertEqual(cell.calls, 5)
        self.assertEqual(len(distances), 5)
        rows = self.rows()
        self.assertEqual([r['iterate'] for r in rows], list(range(6)))
        self.assertEqual([r['phase_index'] for r in rows], [0,1,2,3,4,None])
        self.assertEqual([r['role'] for r in rows], ['uncorrected_seed_phase']*5+['closing_image'])
        for index, row in enumerate(rows):
            data = (self.out/f'seed-phase-{index:03d}.npy').read_bytes()
            self.assertEqual(row['file'], dict(path=str(self.out/f'seed-phase-{index:03d}.npy'),
                                               bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            self.assertEqual(row['shape'], [1,1,896])
            self.assertEqual(row['dtype'], 'float64')
            self.assertEqual(row['seed_origin'], dict(path=self.card['seed_path'],
                                                      sha256=self.card['seed_sha256'], index=0))
            for name in ('alpha','alpha2','log_temperature'):
                self.assertEqual(row['parameters'][name], self.card[name])
                self.assertEqual(row['parameter_binary64_ratios'][name], list(self.card[name].as_integer_ratio()))
            self.assertFalse(row['corrected'])
            self.assertFalse(row['seed_qualified'])
            self.assertFalse(row['scientific_attempt_complete'])
        self.assertEqual(sum(event[0]=='fsync' for event in events), 27)
        for index, event in enumerate(events):
            if event[0] == 'f':
                self.assertEqual(events[index-4:index], [('fsync','file'),('fsync','dir'),
                                                         ('fsync','file'),('fsync','dir')])
        self.assertEqual(len(list(self.out.iterdir())), 7)

    def test_02_existing_journal_or_link_refuses_before_map(self):
        journal = self.out/'seed-phases.jsonl'
        target = self.base/'sentinel'
        target.write_bytes(b'unchanged')
        for kind in ('file','live-link','dangling-link'):
            with self.subTest(kind=kind):
                if kind == 'file': journal.write_bytes(b'journal-sentinel')
                else: journal.symlink_to(target if kind=='live-link' else self.base/'absent')
                cell = Cell()
                with self.assertRaises(FileExistsError): self.capture(cell)
                self.assertEqual(cell.calls, 0)
                self.assertEqual(target.read_bytes(), b'unchanged')
                if kind == 'file': self.assertEqual(journal.read_bytes(), b'journal-sentinel')
                journal.unlink()

    def test_03_existing_phase_or_link_is_not_overwritten(self):
        phase = self.out/'seed-phase-000.npy'
        target = self.base/'sentinel'
        target.write_bytes(b'unchanged')
        for kind in ('file','live-link','dangling-link'):
            with self.subTest(kind=kind):
                if kind == 'file': phase.write_bytes(b'phase-sentinel')
                else: phase.symlink_to(target if kind=='live-link' else self.base/'absent')
                cell = Cell()
                with self.assertRaises(FileExistsError): self.capture(cell)
                self.assertEqual(cell.calls, 0)
                self.assertEqual((self.out/'seed-phases.jsonl').read_bytes(), b'')
                self.assertEqual(target.read_bytes(), b'unchanged')
                if kind == 'file': self.assertEqual(phase.read_bytes(), b'phase-sentinel')
                phase.unlink(); (self.out/'seed-phases.jsonl').unlink()

    def test_04_late_occupied_phase_preserves_prior_rows_and_stops(self):
        (self.out/'seed-phase-003.npy').write_bytes(b'preexisting')
        cell = Cell()
        with self.assertRaises(FileExistsError): self.capture(cell)
        self.assertEqual(cell.calls, 3)
        self.assertEqual([r['iterate'] for r in self.rows()], [0,1,2])
        self.assertEqual((self.out/'seed-phase-003.npy').read_bytes(), b'preexisting')
        self.assertFalse((self.out/'seed-phase-004.npy').exists())

    def test_05_serializer_failure_keeps_partial_file_without_row(self):
        self.np.fail_save_at = 3
        cell = Cell()
        with self.assertRaisesRegex(OSError, 'serializer'): self.capture(cell)
        self.assertEqual(cell.calls, 2)
        self.assertEqual([r['iterate'] for r in self.rows()], [0,1])
        self.assertEqual((self.out/'seed-phase-002.npy').read_bytes(), b'synthetic-partial-array')
        self.assertFalse((self.out/'seed-phase-003.npy').exists())

    def test_06_parent_fsync_failure_precedes_journal_and_maps(self):
        cell = Cell()
        with mock.patch.object(os, 'fsync', side_effect=OSError('synthetic parent fsync failure')):
            with self.assertRaisesRegex(OSError, 'parent fsync'): self.capture(cell)
        self.assertEqual(cell.calls, 0)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_07_late_array_fsync_failure_preserves_prefix(self):
        real_fsync, counts = os.fsync, [0]
        def fsync(fd):
            counts[0] += 1
            if counts[0] == 12: raise OSError('synthetic X2 array fsync failure')
            return real_fsync(fd)
        cell = Cell()
        with mock.patch.object(os, 'fsync', side_effect=fsync):
            with self.assertRaisesRegex(OSError, 'X2 array'): self.capture(cell)
        self.assertEqual(cell.calls, 2)
        self.assertEqual([r['iterate'] for r in self.rows()], [0,1])
        self.assertTrue((self.out/'seed-phase-002.npy').exists())
        self.assertFalse((self.out/'seed-phase-003.npy').exists())

    def test_08_late_journal_fsync_failure_prevents_next_map(self):
        real_fsync, counts = os.fsync, [0]
        def fsync(fd):
            counts[0] += 1
            if counts[0] == 14: raise OSError('synthetic X2 journal fsync failure')
            return real_fsync(fd)
        cell = Cell()
        with mock.patch.object(os, 'fsync', side_effect=fsync):
            with self.assertRaisesRegex(OSError, 'X2 journal'): self.capture(cell)
        self.assertEqual(cell.calls, 2)
        self.assertEqual([r['iterate'] for r in self.rows()], [0,1,2])
        self.assertTrue((self.out/'seed-phase-002.npy').exists())
        self.assertFalse((self.out/'seed-phase-003.npy').exists())
        # Parseable retained bytes are not a successful durability/completion receipt.

    def test_09_partial_journal_write_preserves_fragment_and_stops(self):
        real_fdopen = os.fdopen
        class BrokenJournal:
            def __init__(self, stream): self.stream, self.writes = stream, 0
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def __getattr__(self, name): return getattr(self.stream, name)
            def write(self, data):
                self.writes += 1
                if self.writes == 3:
                    self.stream.write(data[:len(data)//2]); self.stream.flush()
                    raise OSError('synthetic partial journal write')
                return self.stream.write(data)
        def fdopen(fd, mode):
            stream = real_fdopen(fd, mode)
            return BrokenJournal(stream) if mode == 'wb' else stream
        cell = Cell()
        with mock.patch.object(os, 'fdopen', side_effect=fdopen):
            with self.assertRaisesRegex(OSError, 'partial journal'): self.capture(cell)
        self.assertEqual(cell.calls, 2)
        lines = (self.out/'seed-phases.jsonl').read_bytes().splitlines()
        self.assertEqual([json.loads(line)['iterate'] for line in lines[:2]], [0,1])
        self.assertEqual(len(lines), 3)
        with self.assertRaises(json.JSONDecodeError): json.loads(lines[2])
        self.assertTrue((self.out/'seed-phase-002.npy').exists())
        self.assertFalse((self.out/'seed-phase-003.npy').exists())

    def main_fixture(self, kind='pass'):
        self.out.rmdir()  # main owns creation of the fresh attempt directory.
        seed = self.base/'synthetic-seed.bin'
        seed.write_bytes(b'not-a-real-array')
        self.card.update(repository=str(self.base), data_root=str(self.base), period=5,
                         dtype='float64', free='alpha2', direction=-1, steps=1,
                         tol=1e-10, alpha_scale=10, chunk=8, ds_min=1e-7, ds=.001,
                         ds_max=.002, bounds=[.1025,.1032], threads=1, device='cpu',
                         vectors='synthetic-v4', vectors2='synthetic-v16', inputs=[],
                         seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest())
        card_path = self.base/'synthetic-card.json'
        card_path.write_text(json.dumps(self.card))
        cell = Cell(kind)
        X = []
        def as_tensor(row, dtype, device):
            if dtype != 'torch.float64' or device != 'cpu': raise AssertionError('wrong tensor request')
            value = Tensor(row.values); X.append(value); return value
        loop = SimpleNamespace(torch=SimpleNamespace(as_tensor=as_tensor), state_dtype='torch.float64',
                               device='cpu', sm=SimpleNamespace(add_steer_layer=lambda index: None))
        stats = dict(family=0, trace=0, trace_X=None)
        def soft_family(*args, **kwargs):
            stats['family'] += 1
            return SimpleNamespace(to_np=lambda value: value)
        def trace(family, value, *args, **kwargs):
            stats['trace'] += 1; stats['trace_X'] = value
            raise TraceReached('stop before unchanged continuation implementation')
        modules = {'explore_llm_soft_loop': SimpleNamespace(SoftLoop=lambda *args: loop,
                   SoftOrbit=lambda *args: cell, soft_family=soft_family),
                   'explore_llm_phase_map': SimpleNamespace(load_vectors=lambda path: dict(gratitude=1., hostility=1.)),
                   'bifurcation_curves': SimpleNamespace(), 'branch_trace': SimpleNamespace(trace_branch=trace)}
        argv = ['seeded_branch_002.py','--card',str(card_path),'--out',str(self.out)]
        return cell, X, stats, modules, argv

    def run_main(self, modules, argv):
        with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(sys, 'path', list(sys.path)), \
                mock.patch.dict(os.environ, {}, clear=False):
            self.ns['main']()

    def test_10_real_changed_main_callsite_retains_once_before_trace(self):
        cell, X, stats, modules, argv = self.main_fixture()
        capture = self.ns['capture_seed_return']
        with mock.patch.dict(self.ns, {'capture_seed_return': mock.Mock(wraps=capture)}) as _:
            with self.assertRaises(TraceReached): self.run_main(modules, argv)
            self.ns['capture_seed_return'].assert_called_once()
        self.assertEqual(cell.calls, 5)
        self.assertEqual((stats['family'], stats['trace']), (1,1))
        self.assertIs(stats['trace_X'], X[0])
        self.assertEqual(X[0].values, [1.] * 896)
        check = json.loads((self.out/'seed-check.json').read_text())
        self.assertEqual(check['seed_index'], 0)
        self.assertEqual(check['seed_sha256'], self.card['seed_sha256'])
        self.assertEqual(len(check['return_distances']), 5)
        self.assertEqual(len(self.rows()), 6)

    def test_11_original_fifth_return_gate_still_refuses_after_retention(self):
        cell, X, stats, modules, argv = self.main_fixture('closure')
        with self.assertRaisesRegex(ValueError, 'seed residual or primitive-period gate'):
            self.run_main(modules, argv)
        self.assertEqual((cell.calls, stats['family'], stats['trace']), (5,0,0))
        self.assertEqual(len(self.rows()), 6)
        self.assertGreater(json.loads((self.out/'seed-check.json').read_text())['return_distances'][4], 1e-10)

    def test_12_original_first_return_gate_still_refuses_after_retention(self):
        cell, X, stats, modules, argv = self.main_fixture('primitive')
        with self.assertRaisesRegex(ValueError, 'seed residual or primitive-period gate'):
            self.run_main(modules, argv)
        self.assertEqual((cell.calls, stats['family'], stats['trace']), (5,0,0))
        self.assertEqual(len(self.rows()), 6)
        self.assertEqual(json.loads((self.out/'seed-check.json').read_text())['return_distances'][0], 0.)

    def test_13_fifth_callback_failure_never_reaches_seed_check_or_branch(self):
        cell, X, stats, modules, argv = self.main_fixture()
        self.np.fail_save_at = 6
        with self.assertRaisesRegex(OSError, 'serializer'): self.run_main(modules, argv)
        self.assertEqual((cell.calls, stats['family'], stats['trace']), (5,0,0))
        self.assertEqual([r['iterate'] for r in self.rows()], [0,1,2,3,4])
        self.assertEqual((self.out/'seed-phase-005.npy').read_bytes(), b'synthetic-partial-array')
        self.assertFalse((self.out/'seed-check.json').exists())

    def test_14_static_old_caller_unchanged_and_only_main_call_replaced(self):
        old = OLD.read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),
                         'ec0aca18ddbef582fb810e89032079d979edd116467773ceca8cb8edb9018d37')
        original, changed = ast.parse(old), ast.parse(SOURCE.read_text())
        original_defs = {n.name:n for n in original.body if isinstance(n, ast.FunctionDef)}
        changed_defs = {n.name:n for n in changed.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(changed_defs)-set(original_defs), {'capture_seed_return'})
        for name in ('sha','validate'):
            self.assertEqual(ast.dump(original_defs[name]), ast.dump(changed_defs[name]))
        main_text = ast.get_source_segment(SOURCE.read_text(), changed_defs['main'])
        self.assertEqual(main_text.count('returns=capture_seed_return(cell,X,out,c)'), 1)
        self.assertNotIn('.return_distances(', main_text)
        restored = main_text.replace('returns=capture_seed_return(cell,X,out,c)',
                                     'returns=cell.return_distances(X,5)')
        self.assertEqual(ast.dump(ast.parse(restored).body[0]), ast.dump(original_defs['main']))
        self.assertEqual(hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                         'c3cbc0705e5614138691926955ba1178dded961f86bc4008068a4d0d59354276')


if __name__ == '__main__':
    unittest.main(verbosity=2)
