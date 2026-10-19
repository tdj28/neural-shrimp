"""New recovery-delta controls only; no original evidence or target execution.

Reads local frozen source, loads only the new wrapper and frozen stdlib helpers,
and uses mocked admission/resource records plus fresh tiny temporary directories.
No old test module/suite or supervisor module is imported. Only three hash-bound
supervisor environment statements execute in an isolated synthetic namespace.
These controls do not authenticate an actual audit environment or admit L.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).parent
FROZEN_001_SHA = 'a8fbfd0bd4da26b12ebf7aa3569a6fc4ae0991959c30b6ce74d253523043d127'
FROZEN_SUP_SHA = '306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44'
ALLOCATION_DELTA = (
    (b"OUT = B / 'tube-target-001-audit-001'", b"OUT = B / 'tube-target-001-audit-002'"),
    (b"CARD = D / 'tube-target-001-audit-001-card.json'", b"CARD = D / 'tube-target-001-audit-002-card.json'"),
    (b"PLAN = D / 'tube-target-001-audit-001-plan.json'", b"PLAN = D / 'tube-target-001-audit-002-plan.json'"),
    (b"SELF = D / 'tube_saved_authority.py'", b"SELF = D / 'tube_saved_authority_002.py'"),
)
SPEC = importlib.util.spec_from_file_location('tube_saved_authority_002', HERE/'tube_saved_authority_002.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def pin(path):
    return dict(path=str(path), bytes=2, sha256=hashlib.sha256(b'{}').hexdigest())


def admission_fixture(execution):
    """Local 002-bound construction; independently owned dicts at every boundary."""
    card_pin = dict(pin(M.CARD), sha256='a'*64)
    plan_pin = pin(M.PLAN)
    host_pin = pin(M.D/'tube-target-001-host-closure.json')
    card = dict(schema='codex-task33-tube-saved-audit-card-v1', status='reviewed_for_saved_audit',
                repository=str(M.ROOT), output=str(M.OUT), original_expected=copy.deepcopy(M.EXPECTED_PIN),
                original_closure=copy.deepcopy(M.CLOSURE_PIN), helper_sha256=copy.deepcopy(M.HELPERS),
                supervision_plan=str(M.PLAN), supervision_admission=str(M.SUP/'admission.json'),
                auditor_sha256='a'*64, limits=copy.deepcopy(M.LIMITS), cpu_ids=[0], **M.FLAGS)
    sources = [copy.deepcopy(card_pin), dict(pin(M.SELF), sha256='a'*64),
               dict(pin(M.ROOT/'codex/certify-chaos/supervise.py'), sha256=FROZEN_SUP_SHA),
               *[dict(pin(M.D/name), sha256=digest) for name, digest in M.HELPERS.items()],
               copy.deepcopy(M.EXPECTED_PIN), copy.deepcopy(M.CLOSURE_PIN), copy.deepcopy(host_pin)]
    argv = [M.PYTHON, '-I', '-B', str(M.SELF), '--card', str(M.CARD),
            '--card-sha256', card_pin['sha256'], '--out', str(M.OUT)]
    plan = dict(status='reviewed_for_execution', repository=str(M.ROOT), raw_root=str(M.B),
                python=M.PYTHON, cpu_threads=1, cpu_ids=[0], startup_ram_gib=24,
                source_files=copy.deepcopy(sources), lanes={'audit': dict(entry=str(M.SELF),
                arguments=argv[4:], output=str(M.OUT), supervision=str(M.SUP),
                limits=copy.deepcopy(M.LIMITS), preflight_pins=[copy.deepcopy(M.EXPECTED_PIN), copy.deepcopy(M.CLOSURE_PIN)])})
    admission = dict(source_commit='1'*40, plan=copy.deepcopy(plan_pin), lane='audit', argv=argv.copy(),
                     limits=copy.deepcopy(M.LIMITS), threads={key: '1' for key in execution.THREADS},
                     capacity=dict(available_ram_bytes=24*2**30, disk_free_bytes=16*2**30),
                     source_blobs=[dict(relative_path=str(Path(row['path']).relative_to(M.ROOT)),
                                        bytes=row['bytes'], sha256=row['sha256']) for row in sources])
    io = mock.Mock()
    io.e = execution
    io.current.return_value = (pin(M.SUP/'admission.json'), admission)
    def read(record, decode=False):
        if record['path'] == str(M.PLAN):
            return copy.deepcopy(plan)
        if record['path'] == M.EXPECTED_PIN['path']:
            return {'host_closure': copy.deepcopy(host_pin)}
        return b'synthetic pinned bytes'
    io.read.side_effect = read
    return io, card_pin, card, plan, admission


def supervisor_environment(parent):
    """Extract only the reviewed three statements; never import/run supervisor."""
    payload = (M.ROOT/'codex/certify-chaos/supervise.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != FROZEN_SUP_SHA:
        raise AssertionError('frozen supervisor source differs')
    tree = ast.parse(payload)
    expected = ast.parse("env=os.environ.copy();env.update({k:str(plan.get('cpu_threads',1)) for k in THREADS});env['PYTHONDONTWRITEBYTECODE']='1'").body
    signatures = [ast.dump(node, include_attributes=False) for node in expected]
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    if len(main) != 1:
        raise AssertionError('one supervisor main required')
    matches = [node for node in main[0].body if ast.dump(node, include_attributes=False) in signatures]
    if [ast.dump(node, include_attributes=False) for node in matches] != signatures:
        raise AssertionError('exact ordered environment construction required')
    thread_nodes = [node for node in tree.body if isinstance(node, ast.Assign) and
                    len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'THREADS']
    if len(thread_nodes) != 1:
        raise AssertionError('one literal supervisor thread tuple required')
    threads = ast.literal_eval(thread_nodes[0].value)
    calls = [node for node in ast.walk(main[0]) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
             node.func.value.id == 'subprocess' and node.func.attr == 'Popen']
    if len(calls) != 1 or len([kw for kw in calls[0].keywords if kw.arg == 'env' and
                               isinstance(kw.value, ast.Name) and kw.value.id == 'env']) != 1:
        raise AssertionError('child must receive exactly constructed env')
    namespace = {'os': types.SimpleNamespace(environ=parent), 'plan': {'cpu_threads': 1},
                 'THREADS': threads, '__builtins__': {'str': str}}
    selected = ast.Module(body=matches, type_ignores=[])
    exec(compile(selected, '<synthetic-supervisor-environment-only>', 'exec'), namespace)
    return namespace['env'], threads


class RecoveryDeltaControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution = M.load_helpers()[1]

    def admitted(self, fixture, environment):
        io, card_pin, card, _, _ = fixture
        values = {M.resource.RLIMIT_AS: M.LIMITS['as_bytes'],
                  M.resource.RLIMIT_CPU: M.LIMITS['cpu_seconds'],
                  M.resource.RLIMIT_FSIZE: M.LIMITS['output_bytes']}
        with mock.patch.dict(M.os.environ, copy.deepcopy(environment), clear=True), \
             mock.patch.object(M.os, 'getpriority', return_value=10), \
             mock.patch.object(M.os, 'sched_getaffinity', return_value={0}), \
             mock.patch.object(M.os, 'getresuid', return_value=(1000, 1000, 1000)), \
             mock.patch.object(M.resource, 'getrlimit', side_effect=lambda kind: (values[kind],)*2):
            return M.audit_admission(io, card_pin, card)

    def environment(self):
        return dict({key: '1' for key in self.execution.THREADS}, HF_HUB_OFFLINE='1')

    def test_01_exact_four_assignment_delta_and_inverse_hash(self):
        original = (HERE/'tube_saved_authority.py').read_bytes()
        current = (HERE/'tube_saved_authority_002.py').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), FROZEN_001_SHA)
        forward, inverse = original, current
        for old, new in ALLOCATION_DELTA:
            self.assertEqual(forward.count(old), 1)
            self.assertEqual(inverse.count(new), 1)
            forward = forward.replace(old, new)
            inverse = inverse.replace(new, old)
        self.assertEqual(current, forward)
        self.assertEqual(inverse, original)
        self.assertEqual(hashlib.sha256(inverse).hexdigest(), FROZEN_001_SHA)
        self.assertEqual(M.EXPECTED_PIN['path'], str(M.D/'tube-target-001-audit-001-expected.json'))

    def test_02_hf_one_and_four_threads_accept(self):
        fixture = admission_fixture(self.execution)
        result = self.admitted(fixture, self.environment())
        self.assertEqual(result['card'], fixture[1])
        self.assertEqual(result['plan']['path'], str(M.PLAN))
        fixture[0].verify_pins.assert_called_once_with(fixture[3]['lanes']['audit']['preflight_pins'])

    def test_03_hf_absent_and_non_one_refuse(self):
        for value in (None, '', '0', 'true', '2'):
            fixture = admission_fixture(self.execution)
            environment = self.environment()
            if value is None:
                environment.pop('HF_HUB_OFFLINE')
            else:
                environment['HF_HUB_OFFLINE'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'offline single-thread runtime required'):
                self.admitted(fixture, environment)
            fixture[0].sources.assert_not_called()

    def test_04_each_worker_thread_absent_or_non_one_refuses(self):
        for key in self.execution.THREADS:
            for value in (None, '0', '2'):
                fixture = admission_fixture(self.execution)
                environment = self.environment()
                if value is None:
                    environment.pop(key)
                else:
                    environment[key] = value
                self.assertEqual(fixture[4]['threads'][key], '1')
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'offline single-thread runtime required'):
                    self.admitted(fixture, environment)

    def test_05_each_recorded_thread_absent_or_non_one_refuses(self):
        for key in self.execution.THREADS:
            for value in (None, '0', '2'):
                fixture = admission_fixture(self.execution)
                environment = self.environment()
                if value is None:
                    fixture[4]['threads'].pop(key)
                else:
                    fixture[4]['threads'][key] = value
                self.assertEqual(environment[key], '1')
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'offline single-thread runtime required'):
                    self.admitted(fixture, environment)

    def test_06_held_card_still_refuses(self):
        fixture = admission_fixture(self.execution)
        fixture[2]['status'] = 'held'
        with self.assertRaisesRegex(ValueError, 'reviewed exact saved-audit card required'):
            self.admitted(fixture, self.environment())

    def test_07_literal_audit001_and_mixed_cli_refuse_before_allocation(self):
        old_card = str(M.D/'tube-target-001-audit-001-card.json')
        old_out = str(M.B/'tube-target-001-audit-001')
        for card, out in ((old_card, old_out), (old_card, str(M.OUT)), (str(M.CARD), old_out)):
            args = argparse.Namespace(card=card, card_sha256='a'*64, out=out)
            with self.subTest(card=card, out=out), mock.patch.object(M, '_allocate') as allocate, \
                 self.assertRaises(ValueError):
                M.execute(args)
            allocate.assert_not_called()

    def test_08_literal_audit002_cli_accepts_without_io(self):
        args = argparse.Namespace(card=str(M.CARD), card_sha256='a'*64, out=str(M.OUT))
        self.assertEqual(M.OUT, M.B/'tube-target-001-audit-002')
        self.assertEqual(M.SELF, M.D/'tube_saved_authority_002.py')
        self.assertEqual(M.PLAN, M.D/'tube-target-001-audit-002-plan.json')
        self.assertEqual(M.SUP, Path(str(M.OUT)+'-supervision'))
        with mock.patch.object(M, '_allocate') as allocate:
            self.assertIsNone(M.pure_arguments(args))
        allocate.assert_not_called()

    def test_09_fresh_synthetic002_allocation_is_exclusive(self):
        with tempfile.TemporaryDirectory(prefix='tube-audit002-control-') as directory:
            out = Path(directory)/'tube-target-001-audit-002'
            with mock.patch.object(M, 'OUT', out):
                fd = M._allocate()
                try:
                    M._check_output_identity(fd)
                    self.assertEqual(list(out.iterdir()), [])
                    with self.assertRaises(FileExistsError):
                        M._allocate()
                finally:
                    os.close(fd)

    def test_10_supervisor_inherits_hf_and_overrides_only_thread_values(self):
        for value in (None, '0', '1'):
            parent = {key: '8' for key in self.execution.THREADS}
            parent['SYNTHETIC_UNRELATED'] = 'retained'
            if value is not None:
                parent['HF_HUB_OFFLINE'] = value
            before = copy.deepcopy(parent)
            child, threads = supervisor_environment(parent)
            with self.subTest(value=value):
                self.assertEqual(parent, before)
                self.assertIsNot(child, parent)
                self.assertEqual(set(threads), set(self.execution.THREADS))
                self.assertEqual(len(threads), 4)
                self.assertEqual(child.get('HF_HUB_OFFLINE'), value)
                if value is None:
                    self.assertNotIn('HF_HUB_OFFLINE', child)
                self.assertTrue(all(child[key] == '1' for key in threads))
                self.assertEqual(child['PYTHONDONTWRITEBYTECODE'], '1')
                self.assertEqual(child['SYNTHETIC_UNRELATED'], 'retained')
                fixture = admission_fixture(self.execution)
                if value == '1':
                    self.assertEqual(self.admitted(fixture, child)['card'], fixture[1])
                else:
                    with self.assertRaisesRegex(ValueError, 'offline single-thread runtime required'):
                        self.admitted(fixture, child)


if __name__ == '__main__':
    unittest.main()
