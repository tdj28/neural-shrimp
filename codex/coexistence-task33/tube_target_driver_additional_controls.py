"""Additional source-pinned driver interfaces and durable failure controls.

Only repository source and /tmp synthetic evidence are read. All Git, resource,
upstream-file and model interfaces are mocked. No production numerical module
or target data is imported. Prior 66 controls are inspected, never rerun here.
"""
import ast
import contextlib
import copy
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

R = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
SOURCE = R / 'codex/coexistence-task33/tube_target_driver.py'
SOURCE_SHA = 'ba32975a90ddc65c32cd3153ece35d0e3d938ab6d920a758d42dda75d3f18a88'
PRIOR = Path('/tmp/tube_target_driver_controls.py')
PRIOR_SHA = '245cb11e8d1b680efe89e36834f77ea5e2c66b18f1db093887d5b3bf114b8fa3'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
assert hashlib.sha256(PRIOR.read_bytes()).hexdigest() == PRIOR_SHA
spec = importlib.util.spec_from_file_location('tube_driver_interfaces_tested', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)  # Standard-library-only definitions; main is not called yet.

# Reuse only manufactured fixture definitions, not any previous test invocation.
fixture_names = {'native', 'Fixture', 'base_card', 'current_fixture'}
fixture_nodes = [node for node in ast.parse(PRIOR.read_bytes()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in fixture_names]
assert {node.name for node in fixture_nodes} == fixture_names
exec(compile(ast.fix_missing_locations(ast.Module(body=fixture_nodes, type_ignores=[])), str(PRIOR), 'exec'))
rows = []
WORK = Path(tempfile.mkdtemp(prefix='tube-driver-interface-controls-', dir='/tmp'))


def checked(name, function):
    detail = function()
    rows.append(dict(name=name, passed=True, detail=detail))


def refusal(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), (message, str(error))
        return str(error)
    raise AssertionError('expected driver refusal absent: ' + message)


def admission_fault(kind):
    f, card, authority = current_fixture()
    limits = f.plan(authority)['lanes']['tube']['limits']
    values = {m.resource.RLIMIT_AS: limits['as_bytes'], m.resource.RLIMIT_CPU: limits['cpu_seconds'],
              m.resource.RLIMIT_FSIZE: limits['output_bytes']}
    nice, affinity, source = 10, {0}, 'c' * 40
    initial = kind != 'changed source after call'
    if kind == 'address space': values[m.resource.RLIMIT_AS] -= 1
    if kind == 'CPU limit': values[m.resource.RLIMIT_CPU] -= 1
    if kind == 'file limit': values[m.resource.RLIMIT_FSIZE] -= 1
    if kind == 'nice': nice = 9
    if kind == 'affinity': affinity = {0, 1}
    if kind == 'initial source': source = 'd' * 40
    if kind == 'regularity source':
        f.sources[authority['source_commit'], card['regularity_review']['path']] = {}
    if kind == 'changed source after call':
        relative = 'codex/certify-llm/ball_qwen.py'
        f.pins[str(m.R / relative)] = dict(f.pins[str(m.R / relative)], sha256='0' * 64)
    expected = ('actual worker resource bound differs' if kind in ('address space', 'CPU limit', 'file limit')
                else 'bounded nice10 single-CPU worker required' if kind in ('nice', 'affinity')
                else 'initial private checkpoint differs' if kind == 'initial source'
                else 'synthetic original source differs' if kind == 'regularity source'
                else 'synthetic pinned bytes differ')
    with f.mocked(), patch.object(m.subprocess, 'check_output', return_value=(source + '\n').encode()) as git, \
         patch.object(m.resource, 'getrlimit', side_effect=lambda key: (values[key],) * 2), \
         patch.object(m.os, 'getpriority', return_value=nice), \
         patch.object(m.os, 'sched_getaffinity', return_value=affinity), \
         patch.dict(m.os.environ, {key: '1' for key in m.THREADS}):
        answer = refusal(lambda: m.current_admission(card, authority['card'], Path(card['output']),
                                                     initial_checkpoint=initial), expected)
        if not initial: assert git.call_count == 0
    return answer


for kind in ('address space', 'CPU limit', 'file limit', 'nice', 'affinity',
             'initial source', 'regularity source', 'changed source after call'):
    checked('actual current admission refuses ' + kind, lambda kind=kind: admission_fault(kind))


def prefix_fixture():
    f = Fixture()
    root = Path(m.PREFIX['prefix_output'])
    files = [f.add(root / name, {}, expected) for name, expected in m.PREFIX_FILES.items()]
    report = f.add(root / 'report.json', dict(completed=True, precision=2048, layers=list(range(24))))
    files.append(report)
    card = dict(alpha=m.ALPHA, log10T=m.LOG10_TEMPERATURE, precision=2048)
    argv = [m.PYTHON, '-I', '-B', str(m.D / 'prefix_setup.py'), '--card', str(m.D / 'prefix-card.json'),
            '--out', str(root)]
    authority = f.execution('prefix', m.PREFIX['prefix_source_commit'], m.D / 'prefix-card.json',
        m.D / 'prefix-plan.json', root, card,
        '6c2ad90ef0332733f8f249833ec9d2adcd0799b1ec6c4dc415202df0b78e851c', argv, files, True,
        {'codex/coexistence-task33/prefix_setup.py': '72a95c042b4d5e229264b07886733b23c385f37b0dd95009dfa067935877fbc3',
         'codex/certify-llm/ball_qwen.py': m.PREFIX['prefix_map_source_sha256']})
    for field, key in (('receipt', 'prefix_receipt_sha256'), ('admission', 'prefix_admission_sha256')):
        authority[field]['sha256'] = m.PREFIX[key]
    return f, dict(prefix_execution=authority), authority, report


def prefix_check(kind):
    f, card, authority, report = prefix_fixture()
    if kind == 'missing original layer': f.docs[report['path']]['layers'].pop()
    if kind == 'wrong original precision': f.docs[report['path']]['precision'] = 1024
    if kind == 'wrong original cache bytes':
        f.rc(authority)['output_files'][1]['sha256'] = '0' * 64
    if kind == 'wrong prefix source': f.rc(authority)['source_commit'] = 'f' * 40
    if kind == 'unclosed prefix': f.rc(authority)['worker_identity_absent'] = False
    with f.mocked():
        if kind == 'complete':
            assert m.prefix_authority(card) == authority
            return 'all original prefix joins accepted; no cache decoded'
        expected = ('original prefix completion differs' if kind in ('missing original layer', 'wrong original precision')
                    else 'original prefix produced bytes differ' if kind == 'wrong original cache bytes'
                    else 'original exact command/source/lane differs' if kind == 'wrong prefix source'
                    else 'original worker closure absent')
        return refusal(lambda: m.prefix_authority(card), expected)


for kind in ('complete', 'missing original layer', 'wrong original precision',
             'wrong original cache bytes', 'wrong prefix source', 'unclosed prefix'):
    checked('actual prefix authority ' + kind, lambda kind=kind: prefix_check(kind))


def backend_check():
    modules = {name: value for name, value in sys.modules.items() if name.startswith('flint.')}
    assert not modules, 'test must never import production flint'
    fake = types.SimpleNamespace(__version__='manufactured', __FLINT_VERSION__='manufactured',
                                 ctx=types.SimpleNamespace(prec=2048, threads=1))
    ext = types.SimpleNamespace(__file__='/tmp/synthetic-flint-extension.so')
    def fake_pin(path):
        return dict(path=str(path), bytes=123, sha256=hashlib.sha256(str(path).encode()).hexdigest())
    with patch.object(m, 'pin', side_effect=fake_pin):
        refusal(lambda: m.backend_definition(fake, fake, fake, fake, fake), 'actual FLINT extension binary inventory required')
        with patch.dict(sys.modules, {'flint.manufactured_extension': ext}):
            value = m.backend_definition(fake, fake, fake, fake, fake)
    assert value['precision'] == 2048 and value['threads'] == 1
    assert value['flint_extensions']['flint.manufactured_extension'] == fake_pin(ext.__file__)
    return 'empty extension inventory refused; original complete fake binary pin retained'


checked('actual backend interface requires and retains loaded extension pins', backend_check)


class Scalar:
    def __init__(self, pair): self.pair = tuple(pair)
    def is_exact(self): return True
    def is_finite(self): return True
    def mid(self): return self
    def man_exp(self): return self.pair
    def __gt__(self, value): return F(self.pair[0]) * F(2) ** self.pair[1] > value


class Array:
    def __init__(self): self.flags = types.SimpleNamespace(writeable=True)
    def numpy(self): return self
    def __getitem__(self, row):
        assert row == 198
        return [0.0] * 896


def integration(kind):
    """Run unaltered main with model/Arb/receiver interfaces wholly synthetic."""
    root = WORK / kind
    repo, raw = root / 'repo', root / 'raw'
    directory = repo / 'codex/coexistence-task33'
    directory.mkdir(parents=True); raw.mkdir()
    script, card_path, out = directory / 'tube_target_driver.py', directory / 'tube-card.json', raw / 'tube-case'
    counter = dict(admission=0, loader=0, receiver=0, backend=0)
    modules, docs, pins = {}, {}, {}
    original_pin = m.pin
    original_read = m.read
    def fake_pin(path, sha=None):
        path = str(path)
        record = dict(path=path, bytes=123, sha256=sha or hashlib.sha256(path.encode()).hexdigest())
        pins[path] = record
        return record
    def guarded_pin(path):
        path = Path(path)
        if path.is_relative_to(out): return original_pin(path)
        assert str(path) in pins, 'unmocked pin access: ' + str(path)
        return copy.deepcopy(pins[str(path)])
    def guarded_read(path):
        path = Path(path)
        if path.is_relative_to(out): return original_read(path)
        assert str(path) in docs, 'unmocked read access: ' + str(path)
        return copy.deepcopy(docs[str(path)])
    def guarded_verify(record):
        assert guarded_pin(record['path']) == record
        return record
    cp = fake_pin(card_path)
    initial = fake_pin(raw / 'original-centre-000.json')
    step = fake_pin(raw / 'original-step-001.json')
    snapshot, vectorfile = root / 'assets', root / 'vectors.npz'
    asset_pins = [fake_pin(vectorfile if name == 'vectors.npz' else snapshot / name, sha)
                  for name, sha in m.ASSETS.items()]
    prefix_pins = [fake_pin(Path(m.PREFIX['prefix_receipt']), m.PREFIX['prefix_receipt_sha256']),
                   fake_pin(Path(m.PREFIX['prefix_admission_path']), m.PREFIX['prefix_admission_sha256'])]
    prefix_pins += [fake_pin(Path(m.PREFIX['prefix_output']) / name, sha) for name, sha in m.PREFIX_FILES.items()]
    for relative, sha in m.CORE_SHA.items():
        fake_pin(repo / relative, sha)
        name = Path(relative).stem
        module = types.ModuleType(name); module.__file__ = str(repo / relative)
        modules[name] = module
    ctx = types.SimpleNamespace(prec=0, threads=0)
    flint = types.ModuleType('flint'); flint.arb = Scalar; flint.ctx = ctx
    numpy = types.ModuleType('numpy')
    torch = types.ModuleType('torch'); torch.set_num_threads = lambda n: None; torch.set_num_interop_threads = lambda n: None
    safetensors = types.ModuleType('safetensors')
    threadpoolctl = types.ModuleType('threadpoolctl'); threadpoolctl.threadpool_limits = lambda **kw: contextlib.nullcontext()
    modules.update(flint=flint, numpy=numpy, torch=torch, safetensors=safetensors, threadpoolctl=threadpoolctl)
    class FakeLoader:
        @classmethod
        def from_pinned_files(cls):
            counter['loader'] += 1
            if kind == 'loader-refusal': raise ValueError('manufactured loader refusal')
            return types.SimpleNamespace(d=896, layers=24, vocab=151936, steering_layer=12, config={},
                weights={'model.embed_tokens.weight': Array()}, vector=Array())
    class FakeBall:
        def __init__(self, *args, **kwargs):
            assert kwargs['direction_count'] == 1 and kwargs['token_chunk'] == 1024
            assert kwargs['alpha'] == m.ALPHA and kwargs['log10_temperature'] == m.LOG10_TEMPERATURE
            self.cache = None; self.precision = 2048
            self.epsilon = Scalar((1, 0)); self.temperature = Scalar((1, 0))
            self.scale = Scalar((1, 0)); self.frequencies = [Scalar((1, 0))]
        def forward(self, *args):
            raise AssertionError('no model forward, even the fake method, is allowed')
    def install_prefix(*args): args[1].cache = [None] * 24
    modules['qwen_map'].ManualQwenSoftMap = FakeLoader
    modules['qwen_map'].PREFIX_IDS = (198,)
    modules['qwen_map'].PINNED_SHA256 = copy.deepcopy(m.ASSETS)
    modules['qwen_map'].SNAPSHOT = snapshot; modules['qwen_map'].VECTOR_FILE = vectorfile
    modules['ball_linear'].DyadicWeightCache = lambda **kw: ('manufactured cache', kw)
    modules['cache_io'].unpack = lambda record: record
    modules['prefix_import'].install_prefix = install_prefix
    modules['streamed_head'].StreamedHeadQwenJet = FakeBall
    result = dict(L=m.qrecord(F(2)), full_output=[native() for _ in range(896)],
                  signed_derivative=[[native(r=2)] for _ in range(896)],
                  row_abs_endpoint_upper=[m.qrecord(F(2)) for _ in range(896)])
    def receiver(ball, centres, radius, save):
        counter['receiver'] += 1
        assert len(centres) == 896 and radius == F(1, 2)
        save('tube-inputs.json', {'manufactured': True})
        if kind == 'domain-refusal': raise ValueError('manufactured denominator refusal')
        save('actual-receiver.json', {'manufactured': True})
        save('tube-operator.json', result)
        if kind == 'duplicate-save': save('tube-operator.json', {'replacement': 'must not overwrite'})
        if kind == 'operator-then-refusal': raise ValueError('manufactured bookkeeping refusal')
        if kind == 'missing-coverage': return result
        cover = dict(started_layers=list(range(24)), ended_layers=list(range(24)),
                     head_passes=[[tag, first, min(first + 1024, 151936)] for tag in
                       ('head_logits', 'head_derivatives') for first in range(0, 151936, 1024)],
                     actual_receiver_equals_original=True, all_direction_cube_retained=True)
        if kind == 'bad-coverage': cover['head_passes'].pop()
        save('coverage.json', cover)
        return result
    modules['tube_receiver'].evaluate_receiver = receiver
    backend = {'manufactured_backend': True}
    card = dict(input_files=[initial, step, *asset_pins, *prefix_pins], backend=backend,
                backend_sha256=m.canonical(backend), map=m.map_definition((198,)),
                map_sha256=m.canonical(m.map_definition((198,))), regularity_review={'manufactured': True})
    docs[str(card_path)] = card
    selection = dict(initial_file=initial, step1_file=step, embedding={'asset': asset_pins[0]})
    def candidate(_):
        if kind == 'candidate-refusal': raise ValueError('manufactured saved-evidence refusal')
        return [native((i % 5) - 2, -2) for i in range(896)], [F(0)] * 896, F(1, 2), selection
    binding = dict(manufactured_authority=True, **m.FLAGS)
    def admission(*args, initial_checkpoint=True):
        counter['admission'] += 1
        assert initial_checkpoint == (counter['admission'] == 1)
        if kind == 'pre-admission-refusal' or kind == 'post-admission-refusal' and not initial_checkpoint:
            raise ValueError('manufactured authority refusal')
        return copy.deepcopy(binding)
    def backend_read(*args):
        counter['backend'] += 1
        return {'changed': True} if kind == 'backend-drift' and counter['backend'] > 1 else copy.deepcopy(backend)
    real_open = os.open
    fsyncs = []
    def local_open(path, *args, **kwargs):
        assert Path(path).is_relative_to(out), 'non-synthetic write forbidden'
        return real_open(path, *args, **kwargs)
    real_fsync = os.fsync
    def sync(fd): fsyncs.append(fd); return real_fsync(fd)
    error = None
    with contextlib.ExitStack() as stack:
        for key, value in dict(R=repo, D=directory, B=raw, SELF=script, __file__=str(script)).items():
            stack.enter_context(patch.object(m, key, value))
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(m.sys, 'argv', [str(script), '--card', str(card_path),
                                                       '--card-sha256', cp['sha256'], '--out', str(out)]))
        stack.enter_context(patch.object(m.sys, 'path', list(sys.path)))
        stack.enter_context(patch.dict(m.os.environ, {}, clear=False))
        for key, value in dict(pin=guarded_pin, read=guarded_read, verify=guarded_verify,
                               current_admission=admission, candidate=candidate,
                               prefix_authority=lambda _: {'manufactured': True}, backend_definition=backend_read).items():
            stack.enter_context(patch.object(m, key, value))
        stack.enter_context(patch.object(m.importlib, 'import_module', side_effect=lambda name: modules[name]))
        stack.enter_context(patch.object(m.subprocess, 'check_output', side_effect=AssertionError('real subprocess forbidden')))
        stack.enter_context(patch.object(m.resource, 'getrusage', return_value=types.SimpleNamespace(ru_maxrss=123)))
        stack.enter_context(patch.object(m.os, 'open', side_effect=local_open))
        stack.enter_context(patch.object(m.os, 'fsync', side_effect=sync))
        try: m.main()
        except (ValueError, FileExistsError) as caught: error = caught
    if kind == 'pre-admission-refusal':
        assert error and not out.exists() and counter['loader'] == counter['receiver'] == 0
        return dict(counter=counter, expected='external supervisor owns pre-allocation failure')
    saved = {path.name: json.loads(path.read_text()) for path in out.iterdir()}
    assert len(fsyncs) == 2 * len(saved)
    assert all(all(record[key] is value for key, value in m.FLAGS.items()) for record in saved.values())
    if kind == 'complete':
        assert error is None and 'failure.json' not in saved and saved['status.json']['target_calls'] == 1
        assert counter['receiver'] == 1 and counter['admission'] == 2
        assert saved['status.json']['artifacts'] == [original_pin(out / name) for name in
            ('admission.json', 'selection.json', 'prefix-binding.json', 'tube-inputs.json',
             'actual-receiver.json', 'tube-operator.json', 'coverage.json')]
    else:
        assert error is not None and 'status.json' not in saved and saved['failure.json']['status'] == 'failed_or_incomplete'
        assert saved['failure.json']['target_calls_started'] == counter['receiver'] <= 1
        failed_inventory = saved['failure.json']['artifacts']
        expected_inventory = [original_pin(out / name) for name in saved if name != 'failure.json']
        assert len(failed_inventory) == len(expected_inventory)
        assert {row['path']: row for row in failed_inventory} == {row['path']: row for row in expected_inventory}
    if 'tube-operator.json' in saved:
        assert all(saved['tube-operator.json'][key] == value for key, value in result.items())
        assert len(saved['tube-operator.json']['signed_derivative']) == 896
    return dict(counter=counter, expected_failure=None if error is None else str(error),
                synthetic_output=str(out), durable_files=[original_pin(out / name) for name in sorted(saved)])


for kind in ('complete', 'pre-admission-refusal', 'candidate-refusal', 'loader-refusal', 'domain-refusal',
             'operator-then-refusal', 'missing-coverage', 'bad-coverage', 'duplicate-save',
             'post-admission-refusal', 'backend-drift'):
    checked('unaltered main with entirely fake dependencies: ' + kind, lambda kind=kind: integration(kind))

result = dict(schema='codex-task33-tube-driver-additional-interface-controls-v1',
    source_sha256=SOURCE_SHA, controls_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reused_fixture_source_sha256=PRIOR_SHA, count=len(rows), passed=True, controls=rows,
    actual_model_imports=0, actual_model_calls=0, actual_target_reads=0, actual_target_evaluations=0,
    Git_or_process_operations=0, prior_controls_repeated=False, synthetic_evidence_root=str(WORK),
    scope='Actual current_admission/prefix_authority/backend_definition/main source; external authority and model '
          'interfaces mocked. Durable exclusive writing/fsync is real only under the named /tmp synthetic root. '
          'No target numerical accuracy, asset authenticity or actual resource feasibility is established.')
print(json.dumps(result, indent=2))
