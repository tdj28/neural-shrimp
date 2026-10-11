"""Synthetic-only controls; never evaluate the S1R target or start its launcher."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import s1r_lower_association as a
from neural_shrimp.period2_branch import Options, Budget, PilotStop
from neural_shrimp.period2_branch_io import Store, file_entry, identity, load_record
from neural_shrimp.s1r_bootstrap import HiddenBridge, InterruptibleBudget


@pytest.fixture(autouse=True)
def simulated_permitted_priority(monkeypatch):
    # Synthetic drivers do not require the parent pytest process to be niced.
    # Production keeps its real guard; the explicit rejection test overrides this.
    import os
    monkeypatch.setattr(os, 'getpriority', lambda *_: 10)


def module(name):
    path = Path(__file__).parents[1] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def family(params, unstable=False):
    center = float(params[0] + 2 * params[1])
    coefficient = 1.2 if unstable else .2
    def f(x):
        z = np.asarray(x) - center
        return center-z+coefficient*z*(z*z-1)
    def jac(x):
        z = np.asarray(x) - center
        return np.diag(-1+coefficient*(3*z*z-1))
    return f, jac


def endpoint(hidden):
    params = np.array([hidden, a.OUTPUT_LOG])
    center = float(params[0]+2*params[1])
    return dict(id='saved-synthetic', params=params, rates=np.array([10.**float(v) for v in params]),
                pair=np.array([[center-1], [center+1]]))


class MemoryStore:
    def __init__(self): self.events = []
    def emit(self, kind, payload, final=False): self.events.append((kind, deepcopy(payload), final))


def setup(family_fn=family, budget=None):
    store = MemoryStore()
    progress = a.Progress(store)
    engine = HiddenBridge(family_fn, Options(), budget or Budget(20, a.MAX_PHASES), progress.emit, a.MAX_LINKS)
    progress.engine = engine
    result = dict(status='running', bridge=[], association=None)
    return engine, progress, result, store


def test_fixed_75_macros_are_exact_python_arithmetic_and_final_copy():
    upper, lower = endpoint(1.1), endpoint(.95)
    targets = a.macro_targets(upper['params'], lower['params'])
    assert len(targets) == 75
    for i, target in enumerate(targets[:-1], 1):
        assert target[0] == float(1.1+(.95-1.1)*i/75)
        assert target[1:].tobytes() == upper['params'][1:].tobytes()
    assert targets[-1].tobytes() == lower['params'].tobytes()
    targets[-1][0] = 9
    assert lower['params'][0] == .95
    changed = lower['params'].copy(); changed[1] = np.nextafter(changed[1], np.inf)
    with pytest.raises(ValueError, match='fixed output'): a.macro_targets(upper['params'], changed)


@pytest.mark.parametrize('unstable', [False, True])
def test_full_75_link_bridge_has_604_calls_and_preserves_unstable_pairs(unstable):
    engine, progress, result, store = setup(lambda p: family(p, unstable))
    a.compare_along_bridge(engine, endpoint(1.1), endpoint(.95), result, progress)
    assert result['status'] == 'completed_common_control_comparison'
    assert result['association']['status'] == 'same_pair_within_tolerance'
    assert result['association']['both_numerically_attracting'] is not unstable
    assert engine.budget.calls == 604 and len(engine.edges) == 75
    assert len(progress.accepted_edges) == 75
    assert progress.last_accepted['params'].tobytes() == endpoint(.95)['params'].tobytes()
    assert len(result['bridge']) == 75
    for edge in engine.edges:
        assert edge['status'] == 'accepted'
        assert all(edge[k]['passed'] for k in ('parent_match', 'half_parent_match', 'half_child_match', 'direct_half_match', 'reverse_match'))
        assert len(edge['nodes']) == 4


def test_secant_predecessor_and_carried_phase_survive_macro_boundary():
    engine, progress, result, store = setup()
    seen = []
    original = engine.hidden_segment
    def inspect(parent, target, previous=None, **kwargs):
        seen.append((a.snapshot(parent), a.snapshot(previous)))
        return original(parent, target, previous, **kwargs)
    engine.hidden_segment = inspect
    upper = endpoint(1.1); upper['pair'] = upper['pair'][::-1].copy()
    a.compare_along_bridge(engine, upper, endpoint(.95), result, progress)
    assert len(seen) == 75 and seen[0][1] is None
    for parent, previous in seen[1:]:
        assert previous is not None and previous['params'][0] > parent['params'][0]
        assert parent['pair'][0, 0] > parent['pair'][1, 0]
        assert previous['pair'][0, 0] > previous['pair'][1, 0]
    # The native endpoint records were not reversed in place by phase carrying.
    for edge in engine.edges:
        native = engine.nodes[edge['direct']]
        assert native['pair'][0, 0] > native['pair'][1, 0]
    assert result['association']['match']['phase'] == 1


def test_caps_stop_before_attempt81_and_phase645():
    engine, progress, result, store = setup()
    node = engine.node(endpoint(1.1)['params'], endpoint(1.1)['pair'], 'synthetic')
    engine.edges = [dict(id=str(i)) for i in range(80)]
    with pytest.raises(PilotStop, match='global_link_attempt_limit'):
        engine.link(node, endpoint(1.098)['params'])
    assert len(engine.edges) == 80
    budget = Budget(10, 644)
    assert [budget.corrector() for _ in range(644)][-1] == 644
    with pytest.raises(PilotStop, match='phase_corrector_limit'): budget.corrector()
    assert budget.calls == 644


def test_gap_is_retained_without_association_or_jump():
    def gap(p):
        if .99 < p[0] < 1.099: raise ValueError('synthetic gap')
        return family(p)
    engine, progress, result, store = setup(gap)
    a.compare_along_bridge(engine, endpoint(1.1), endpoint(.95), result, progress)
    assert result['status'] == 'bridge_gap' and result['association'] is None
    assert progress.last_accepted['params'][0] >= 1.099
    assert any(edge['status'] == 'rejected' for edge in engine.edges)
    assert not any(k == 'association' for k, _, _ in store.events)
    assert len(result['bridge']) == 1


def test_minimum_width_can_consume_initial_plus_four_failed_attempts():
    engine, progress, result, store = setup()
    root = engine.node(endpoint(1.1)['params'], endpoint(1.1)['pair'], 'synthetic')
    engine.family = lambda p: (_ for _ in ()).throw(ValueError('all targets fail'))
    segment, last, _ = engine.hidden_segment(root, endpoint(1.098)['params'])
    assert segment['status'] == 'unresolved_gap' and len(engine.edges) == 5
    assert last['id'] == root['id']


@pytest.mark.parametrize('event, reason', [('corrector_start', 'interrupted'), ('corrector_iteration', 'converged')])
def test_deadline_during_next_link_keeps_last_complete_witness_and_latest_proposal(event, reason):
    clock = [0.]
    budget = InterruptibleBudget(1, 644, clock=lambda: clock[0])
    engine, progress, result, store = setup(budget=budget)
    original = progress.emit
    def interrupt(kind, payload):
        original(kind, payload)
        if kind == event and len(progress.accepted_edges) == 1:
            clock[0] = 2.
    engine.emit = interrupt
    with pytest.raises(PilotStop, match='deadline'):
        a.compare_along_bridge(engine, endpoint(1.1), endpoint(.95), result, progress)
    terminal = progress.terminal()
    assert len(terminal['accepted_edges']) == 1 and result['association'] is None
    assert terminal['last_witnessed']['params'][0] == a.macro_targets(endpoint(1.1)['params'], endpoint(.95)['params'])[0][0]
    assert terminal['pending_edges'] == ['edge-0001']
    assert terminal['latest_corrector']['reason'] == reason
    assert any(k == 'corrector_end' and final for k, _, final in store.events) == (reason == 'interrupted')


def test_same_distinct_intermediate_and_exact_rate_controls():
    from neural_shrimp.s1r_bootstrap import compare_common_controls
    engine, _, _, _ = setup()
    p = endpoint(.95); n = engine.node(p['params'], p['pair'], 'synthetic')
    same = deepcopy(n)
    assert compare_common_controls(n, same)['status'] == 'same_pair_within_tolerance'
    same['pair'] += 1e-3
    assert compare_common_controls(n, same)['numerically_coexisting_attractors']
    same['spectra']['phases'][0]['stability'] = 'unstable'
    assert not compare_common_controls(n, same)['numerically_coexisting_attractors']
    same['pair'] = n['pair'] + 1e-6
    assert compare_common_controls(n, same)['status'] == 'unresolved_match'
    same['rates'][0] = np.nextafter(same['rates'][0], np.inf)
    with pytest.raises(ValueError, match='controls'): compare_common_controls(n, same)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 0., True, 4.])
def test_cost_gate_rejects_nonfinite_wrong_or_excessive_timings(bad):
    timing = dict(status='completed_local_segment', phase_corrector_calls=42, elapsed_seconds=bad)
    collection = dict(status='completed_common_control_comparison', link_attempts=12)
    with pytest.raises(ValueError): a.measured_cost_gate(timing, collection, 27218975)


def test_cost_gate_fixed_counts_and_both_caps():
    timing = dict(status='completed_local_segment', phase_corrector_calls=42, elapsed_seconds=2.5109625079203397)
    collection = dict(status='completed_common_control_comparison', link_attempts=12)
    gate = a.measured_cost_gate(timing, collection, 27218975)
    assert 87 < gate['projected_seconds'] < 88
    assert gate['ordinary_byte_allowance'] == 200540160
    assert gate['projected_raw_bytes'] < gate['ordinary_byte_allowance']
    for field, value in [('status', 'resource_stop'), ('phase_corrector_calls', 41), ('phase_corrector_calls', 42.)]:
        with pytest.raises(ValueError): a.measured_cost_gate(dict(timing, **{field: value}), collection, 27218975)
    with pytest.raises(ValueError): a.measured_cost_gate(timing, dict(collection, link_attempts=13), 27218975)
    with pytest.raises(ValueError): a.measured_cost_gate(timing, collection, 27218976)


def test_manifest_rejects_corruption_extra_file_or_symlink(tmp_path):
    runner = module('explore_s1r_lower_association')
    store = Store(tmp_path/'inputs'); store.json('fixed.json', {'value': 1}); store.finish({'status': 'done'})
    expected = file_entry(store.root/'manifest.json')['sha256']
    assert runner.check_manifest(store.root, expected)['status'] == 'done'
    original = (store.root/'fixed.json').read_bytes()
    (store.root/'fixed.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='binding'): runner.check_manifest(store.root, expected)
    (store.root/'fixed.json').write_bytes(original)
    (store.root/'extra').write_text('new')
    with pytest.raises(ValueError, match='inventory'): runner.check_manifest(store.root, expected)
    (store.root/'extra').unlink(); (store.root/'fixed.json').unlink()
    target = tmp_path/'outside'; target.write_bytes(original); (store.root/'fixed.json').symlink_to(target)
    with pytest.raises(ValueError, match='binding'): runner.check_manifest(store.root, expected)


def test_fixed_endpoint_validation_rejects_phase_swap_and_control_change(monkeypatch):
    node = dict(id='synthetic', accepted=True, pair=np.arange(144, dtype=np.float64).reshape(2,72),
                params=np.array([1.1,a.OUTPUT_LOG]), rates=np.array([10.**1.1,10.**a.OUTPUT_LOG]))
    spec = dict(node='synthetic', hidden=1.1, **{k: identity(node[k])['sha256'] for k in ('pair','params','rates')})
    monkeypatch.setitem(a.ENDPOINTS, 'synthetic', spec)
    a.validate_endpoint(node, 'synthetic')
    for key in ('pair','params','rates'):
        changed = deepcopy(node); changed[key].flat[0] = np.nextafter(changed[key].flat[0], np.inf)
        with pytest.raises(ValueError, match='identity'): a.validate_endpoint(changed, 'synthetic')
    changed = dict(node, pair=node['pair'][::-1].copy())
    with pytest.raises(ValueError, match='identity'): a.validate_endpoint(changed, 'synthetic')
    with pytest.raises(ValueError, match='qualified'): a.validate_endpoint(dict(node, accepted=False), 'synthetic')


def test_terminal_checkpoint_uses_reserve_and_keeps_raw_failed_attempt(tmp_path):
    store = Store(tmp_path/'reserved', 65536)
    progress = a.Progress(store)
    progress.emit('fill', {'array': np.zeros(5300)})
    with pytest.raises(PilotStop, match='raw_byte_limit'):
        progress.emit('corrector_iteration', {'state': np.zeros(1200)})
    progress.emit('corrector_end', {'reason':'interrupted','state':np.zeros(100),'call':1})
    store.emit('terminal', progress.terminal(), final=True)
    store.finish(dict(status='resource_stop'))
    assert sum(p.stat().st_size for p in store.root.iterdir()) <= 65536
    kind, terminal = load_record(store.root, '00003-terminal.json')
    assert terminal['last_witnessed'] is None and terminal['latest_corrector']['reason'] == 'interrupted'


def test_driver_full_synthetic_and_failure_finalization(tmp_path, monkeypatch):
    runner = module('explore_s1r_lower_association')
    @dataclass
    class Model: n_parameters:int = 1
    class Grid:
        def __init__(self, net, x, y, hidden, output): self.params=np.log10([hidden[0],output[0]])
        def cell_map(self, index): return family(self.params)
    monkeypatch.setattr(runner, 'sources', lambda: {'synthetic': {}})
    monkeypatch.setattr(runner, 'bound_inputs', lambda: (Model(),None,None,dict(upper=endpoint(1.1),lower=endpoint(.95)),{},dict(passes=True)))
    monkeypatch.setattr(runner, 'GridMap', Grid)
    review='8c26537e71ff12f8e6a4b7916775cc8160e3b8a1'
    result = runner.run(tmp_path/'full', review)
    assert result['status']=='completed_common_control_comparison' and result['phase_corrector_calls']==604
    assert result['link_attempts']==75 and len(result['witnessed_edge_ids'])==75
    assert (tmp_path/'full'/'manifest.json').exists()
    def fail(): raise ValueError('synthetic input failure')
    monkeypatch.setattr(runner,'bound_inputs',fail)
    with pytest.raises(ValueError,match='synthetic input'): runner.run(tmp_path/'failure',review)
    result=json.loads((tmp_path/'failure'/'result.json').read_text())
    assert result['status']=='failed_or_interrupted' and result['phase_corrector_calls']==0 and result['association'] is None
    assert (tmp_path/'failure'/'manifest.json').exists()


def test_launcher_fixed_command_contains_timeout_cpu_nice_and_no_profile():
    launcher=module('launch_s1r_lower_association')
    command=launcher.command(Path('/synthetic/out'),'a'*40)
    assert command[:7]==['timeout','--signal=TERM','--kill-after=5s','115s','nice','-n','10']
    assert 'profile' not in command and command.count('--review-commit')==1
    assert a.STORE_BYTES+a.LAUNCHER_BYTES==a.TOTAL_BYTES


@pytest.mark.parametrize('exit_code', [0, 2])
def test_launcher_retains_synthetic_child_exit_log_and_raw_manifest(tmp_path, monkeypatch, exit_code):
    import subprocess
    import sys
    launcher = module('launch_s1r_lower_association')
    monkeypatch.setattr(launcher, 'committed_sources', lambda: {'synthetic': {}})
    def fake(out, review):
        code = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); "
                "(p/'result.json').write_text('{}'); (p/'manifest.json').write_text('{}'); "
                "print('synthetic child only'); sys.exit(int(sys.argv[2]))")
        return [sys.executable, '-c', code, str(out), str(exit_code)]
    monkeypatch.setattr(launcher, 'command', fake)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=launcher.ROOT,text=True).strip()
    receipt = launcher.launch(tmp_path/'raw', tmp_path/'launcher', head, head)
    assert receipt['process_exit'] == exit_code and receipt['child_pid'] > 0
    assert receipt['raw_file_count'] == 2 and receipt['raw_manifest'] is not None
    assert receipt['stdout'] == file_entry(tmp_path/'launcher'/'stdout.log')
    assert receipt['within_store_cap'] and not receipt['log_truncated']
    assert (tmp_path/'launcher'/'manifest.json').exists()
    assert sum(p.stat().st_size for p in (tmp_path/'launcher').iterdir()) < a.LAUNCHER_BYTES


def test_launcher_log_overflow_stops_and_records_synthetic_child(tmp_path, monkeypatch):
    import subprocess
    import sys
    launcher = module('launch_s1r_lower_association')
    monkeypatch.setattr(launcher, 'committed_sources', lambda: {'synthetic': {}})
    monkeypatch.setattr(launcher, 'LOG_LIMIT', 1024)
    monkeypatch.setattr(launcher, 'command', lambda out, review: [sys.executable, '-c',
        "import os,time; os.write(1,b'x'*16384); time.sleep(20)"])
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=launcher.ROOT,text=True).strip()
    receipt = launcher.launch(tmp_path/'raw', tmp_path/'launcher', head, head)
    assert receipt['log_truncated'] and receipt['process_exit'] != 0
    assert receipt['stdout']['bytes'] == 1024 and receipt['raw_file_count'] == 0
    assert receipt['elapsed_seconds'] < 5


def test_launcher_rejects_wrong_execution_source_before_output(tmp_path):
    launcher = module('launch_s1r_lower_association')
    with pytest.raises(ValueError, match='HEAD mismatch'):
        launcher.launch(tmp_path/'raw',tmp_path/'launcher','0'*40,'0'*40)
    assert not (tmp_path/'raw').exists() and not (tmp_path/'launcher').exists()


def test_failed_accepted_edge_emit_keeps_pending_journal_edge():
    engine, progress, result, store = setup()
    original=store.emit
    def fail_edge(kind,payload,final=False):
        if kind=='edge' and payload['status']=='accepted': raise PilotStop('raw_byte_limit')
        original(kind,payload,final)
    store.emit=fail_edge
    with pytest.raises(PilotStop,match='raw_byte_limit'):
        a.compare_along_bridge(engine,endpoint(1.1),endpoint(.95),result,progress)
    terminal=progress.terminal()
    assert engine.edges[0]['status']=='accepted'  # mutated before failed emit
    assert terminal['pending_edges']==['edge-0000'] and terminal['durably_finished_edges']==[]
    assert terminal['accepted_edges']==[] and terminal['last_witnessed']['params'][0]==1.1
    assert result['association'] is None


def test_failed_edge_start_is_an_unpersisted_attempt():
    engine, progress, result, store = setup()
    original=store.emit
    def fail_start(kind,payload,final=False):
        if kind=='edge_pending': raise PilotStop('raw_byte_limit')
        original(kind,payload,final)
    store.emit=fail_start
    with pytest.raises(PilotStop,match='raw_byte_limit'):
        a.compare_along_bridge(engine,endpoint(1.1),endpoint(.95),result,progress)
    terminal=progress.terminal()
    assert terminal['pending_edges']==[] and terminal['unpersisted_edge_attempts']==['edge-0000']
    assert len(engine.edges)==1 and result['association'] is None


def test_launcher_exited_leader_with_inherited_pipe_is_bounded(tmp_path,monkeypatch):
    import subprocess
    import sys
    launcher=module('launch_s1r_lower_association')
    monkeypatch.setattr(launcher,'committed_sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(launcher,'SUPERVISOR_SECONDS',.3)
    code="import os,time; pid=os.fork(); time.sleep(20) if pid==0 else os._exit(0)"
    monkeypatch.setattr(launcher,'command',lambda out,review:[sys.executable,'-c',code])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=launcher.ROOT,text=True).strip()
    receipt=launcher.launch(tmp_path/'raw',tmp_path/'launcher',head,head)
    assert receipt['failure']=='supervisor_timeout_or_inherited_pipe'
    assert receipt['elapsed_seconds']<3 and receipt['process_exit']==0
    assert receipt['raw_file_count']==0 and (tmp_path/'launcher'/'manifest.json').exists()


@pytest.mark.parametrize('failed_event', ['node_pending', 'node'])
def test_node_pending_metadata_follows_durable_events(failed_event):
    engine, progress, result, store = setup()
    original=store.emit
    def fail(kind,payload,final=False):
        if kind==failed_event: raise PilotStop('raw_byte_limit')
        original(kind,payload,final)
    store.emit=fail
    with pytest.raises(PilotStop,match='raw_byte_limit'):
        a.compare_along_bridge(engine,endpoint(1.1),endpoint(.95),result,progress)
    terminal=progress.terminal()
    assert terminal['durably_finished_nodes']==[]
    assert terminal['pending_nodes']==(['node-0000'] if failed_event=='node' else [])
    assert terminal['unpersisted_node_attempts']==(['node-0000'] if failed_event=='node_pending' else [])
    assert terminal['last_witnessed'] is None and result['association'] is None


def test_driver_and_launcher_reject_ordinary_priority_before_output(tmp_path,monkeypatch):
    import os
    import subprocess
    monkeypatch.setattr(os,'getpriority',lambda *_:0)
    runner=module('explore_s1r_lower_association')
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    with pytest.raises(ValueError,match='nice10'):
        runner.run(tmp_path/'raw',head)
    launcher=module('launch_s1r_lower_association')
    monkeypatch.setattr(launcher,'committed_sources',lambda:{'synthetic':{}})
    with pytest.raises(ValueError,match='nice10'):
        launcher.launch(tmp_path/'raw',tmp_path/'launcher',head,head)
    assert not (tmp_path/'raw').exists() and not (tmp_path/'launcher').exists()
