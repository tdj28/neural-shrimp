"""EXP-013 recovery regressions using fabricated packets; no fits or dynamics."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_fresh_preview as original
import run_fresh_preview_recovery as recovery


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def forbidden(*args, **kwargs):
    raise AssertionError('Recovery fixture attempted a fit, trajectory, process or network call')


def fabricated_plan():
    anchors = []
    for index in range(24):
        alpha, gamma = 1.06 + index*.0001, 10.2 + index*.001
        anchors.append(dict(id=f'fresh-{index:02d}', alpha=alpha, gamma=gamma,
            group=recovery.old_features.physical_group(alpha, gamma), row=25+index, column=30+index))
    return dict(**copy.deepcopy(recovery.CONTRACT), selection=dict(anchors=anchors, synthetic=True),
        census=recovery.fresh.build_census(anchors), fixed_policies=recovery.evaluation.policy_catalog(),
        source_files=[dict(path=p, sha256='0'*64, bytes=0) for p in recovery.SOURCE_PATHS],
        old_plan=dict(path='synthetic-old-plan.json', sha256='0'*64, bytes=0),
        origin_source_bindings=[], claim_boundary='Synthetic fixture only.')


def test_exact_models_sealed_event_preserves_payload_path(tmp_path):
    log = tmp_path/'events.jsonl'
    fields = dict(path='models/payloads/final-old-models/models.json',
                  sha256='a'*64, packet_sha256='b'*64)
    # This exact call shape caused attempt-001's argument collision.
    recovery.append_event(log, 'models_sealed', **fields)
    event, = [json.loads(line) for line in log.read_text().splitlines()]
    assert event['stage'] == 'models_sealed'
    assert all(event[key] == value for key, value in fields.items())
    assert type(event['monotonic_ns']) is int and event['monotonic_ns'] > 0
    assert isinstance(event['at'], str)
    with pytest.raises(TypeError, match='multiple values'):
        original.append_event(log, 'models_sealed', **fields)
    assert len(log.read_text().splitlines()) == 1


def test_recovery_only_changes_declared_administration_and_writer_normalization():
    """Compare every function/class AST, permitting only the two exact repairs."""
    before = ast.parse(Path(original.__file__).read_text())
    after = ast.parse(Path(recovery.__file__).read_text())
    normalizations = []

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace('EXP-013-recovery-review.json', 'EXP-013-local-review.json')
                node.value = node.value.replace('EXP-013-recovery-plan.json', 'EXP-013-plan.json')
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == 'work.baseline_engine.safe' and len(node.args) == 1 and ast.unparse(node.args[0]) == 'replay':
                assert not node.keywords
                normalizations.append(ast.unparse(node))
                return node.args[0]
            return node

        def visit_FunctionDef(self, node):
            if node.name == 'append_event':
                assert node.args.args[0].arg == 'log_path'
                node.args.args[0].arg = 'path'
                for part in ast.walk(node):
                    if isinstance(part, ast.Name) and part.id == 'log_path':
                        part.id = 'path'
            return self.generic_visit(node)

    revised = Normalize().visit(after)
    definitions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert definitions(before) == definitions(revised)
    assert normalizations == ['work.baseline_engine.safe(replay)']
    assert original.CONTRACT == recovery.CONTRACT
    assert original.HISTORICAL_HASHES == recovery.HISTORICAL_HASHES
    assert recovery.SOURCE_PATHS[:3] == ['tools/run_fresh_preview_recovery.py',
        'tests/test_fresh_preview_recovery.py', 'docs/EXP-013-operational-amendment.md']
    assert recovery.SOURCE_PATHS[3:] == original.SOURCE_PATHS
    plan = fabricated_plan()
    assert original.stage_specs(plan) == recovery.stage_specs(plan)
    assert recovery.validate(plan) == dict(origins=24, candidates=3456, shams=48, short_traces=10608, readouts=6144)
    assert {k: len(v) for k, v in recovery.stage_specs(plan).items()} == dict(
        models=1, baselines=24, sensitivities=48, previews=10608)


def test_saved_model_replay_uses_existing_writer_scalar_normalization(tmp_path):
    replay = dict(training_groups=[np.str_('physical-a'), np.str_('physical-b')],
        selected_penalty=np.float64(.01), coefficients=np.asarray([1., 2., 3.], dtype=np.float64),
        diagnostics=dict(passed=np.bool_(True), folds=np.int64(24)))
    path = tmp_path/'models.json'
    recovery.work.write(path, replay)
    saved = recovery.read(path)
    with pytest.raises(ValueError, match='Scoring category differs'):
        recovery.legacy.compare_scoring(saved, replay)
    normalized = recovery.work.baseline_engine.safe(replay)
    assert type(normalized['training_groups'][0]) is str
    assert type(normalized['selected_penalty']) is float
    assert normalized == saved
    recovery.legacy.compare_scoring(saved, normalized)
    changed = copy.deepcopy(normalized)
    changed['coefficients'][0] += 1e-5
    with pytest.raises(ValueError, match='Scoring number differs'):
        recovery.legacy.compare_scoring(saved, changed)
    changed = copy.deepcopy(normalized)
    changed['training_groups'][0] = 'other-origin'
    with pytest.raises(ValueError, match='Scoring category differs'):
        recovery.legacy.compare_scoring(saved, changed)


def test_mock_run_seals_models_then_preserves_reference_failure(tmp_path, monkeypatch):
    """Run real producer event/finalization logic; all research work is mocked."""
    plan = fabricated_plan()
    repo = tmp_path/'repo'; repo.mkdir()
    monkeypatch.setattr(recovery, 'ROOT', repo)
    plan_path = repo/'experiments/EXP-013-recovery-plan.json'; write(plan_path, plan)
    old_plan_path = repo/'experiments/synthetic-old-plan.json'
    write(old_plan_path, dict(rows=[dict(group=f'old-{i:02d}') for i in range(24)]))
    monkeypatch.setattr(recovery, 'OLD_PLAN', old_plan_path)
    review_path = repo/'docs/EXP-013-recovery-review.json'
    write(review_path, dict(status='ready_for_exploratory_execution', plan_sha256=recovery.legacy.sha(plan_path)))
    validations = []
    monkeypatch.setattr(recovery, 'validate', lambda value, **kw: validations.append(kw))
    monkeypatch.setattr(recovery, 'source_guard', lambda *a: dict(source_commit='1'*40,
        pushed_branch='codex/synthetic', remote_head_verified_at='synthetic'))
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda command: review_path.read_bytes()
        if command[-1] == '1'*40+':docs/EXP-013-recovery-review.json' else forbidden())
    monkeypatch.setattr(recovery.legacy, 'binding', lambda p: dict(path=str(Path(p).relative_to(repo)),
        sha256=recovery.legacy.sha(p), bytes=Path(p).stat().st_size))
    monkeypatch.setattr(recovery.importlib.metadata, 'version', lambda name: recovery.CONTRACT['required_packages'][name])
    resource = dict(available_ram_bytes=32*1024**3, disk_free_bytes=32*1024**3, synthetic=True)
    monkeypatch.setattr(recovery, 'resources', lambda *a, **kw: resource)
    monkeypatch.setattr(recovery, 'require_headroom', lambda *a, **kw: resource)
    monkeypatch.setattr(recovery.work, 'build_libraries', lambda *a: {'synthetic': 'no-binary'})
    monkeypatch.setattr(recovery.final, 'fit_four_recipes', forbidden)
    monkeypatch.setattr(recovery.work, 'identity', lambda pid: dict(pid=pid, start_ticks='synthetic', boot_id='synthetic'))
    monkeypatch.setattr(recovery.platform, 'platform', lambda: 'synthetic-platform')
    timers = []
    monkeypatch.setattr(recovery.signal, 'setitimer', lambda *args: timers.append(args))
    executed = []
    model_payload = recovery.seal(dict(models={f'recipe-{i}': dict(synthetic=True) for i in range(4)}))

    def execute(specs, contexts, directory, libraries, **kwargs):
        executed.append(directory.name)
        records = recovery.fill_terminal(specs, [], 'synthetic_reference_failure')
        if directory.name == 'models':
            assert len(specs) == 1 and len(contexts['old24']['groups']) == 24
            model_rel = 'payloads/final-old-models/models.json'
            write(directory/model_rel, model_payload)
            records[0].update(status='complete', qualified=True, models_path=model_rel, wall_seconds=.01)
        elif directory.name == 'baselines':
            assert len(specs) == len(contexts) == 24
        else:
            forbidden()
        return records

    monkeypatch.setattr(recovery.work, 'execute', execute)
    summary = recovery.run(plan_path, tmp_path/'data', 'attempt-002')
    directory = tmp_path/'data/EXP-013/attempt-002'
    assert executed == ['models', 'baselines']
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    assert [e['stage'] for e in events] == ['stage_started', 'stage_finished', 'models_sealed',
        'fresh_dynamics_started', 'stage_started', 'stage_finished']
    model_event = events[2]
    assert model_event['path'] == 'models/payloads/final-old-models/models.json'
    assert model_event['sha256'] == recovery.legacy.sha(directory/model_event['path'])
    assert model_event['packet_sha256'] == model_payload['packet_sha256']
    assert model_event['monotonic_ns'] < events[3]['monotonic_ns'] < events[4]['monotonic_ns']
    assert not summary['complete'] and not summary['qualified']
    assert 'Full cohort reference qualification failed' in summary['error']
    assert summary['qualification']['models'] is True
    assert not any(v for k, v in summary['qualification'].items() if k != 'models')
    assert summary['cohort_statistics'] is None
    terminal = recovery.read(directory/'all-terminal-records.json')
    assert {k: len(v) for k, v in terminal.items()} == dict(models=1, baselines=24, sensitivities=48, previews=10608)
    assert terminal['models'][0]['status'] == 'complete'
    assert all(r['status'] == 'synthetic_reference_failure' for r in terminal['baselines'])
    assert all(r['status'] == 'not_started_prerequisite_or_producer_failure'
        for stage in ('sensitivities', 'previews') for r in terminal[stage])
    evaluation = recovery.read(directory/'evaluation-unavailable.json')
    assert evaluation['retained_readouts'] == 6144 and evaluation['observed_readouts'] == 0
    assert evaluation['cohort_statistics'] is None
    receipt = recovery.read(directory/'receipt.json')
    assert receipt['status'] == 'incomplete'
    assert receipt['artifacts'] == recovery.work.inventory(directory)
    assert validations == [dict(bindings=True, reconstruct=True), dict(bindings=True, reconstruct=False)]
    assert timers[-1] == (recovery.signal.ITIMER_REAL, 0)
