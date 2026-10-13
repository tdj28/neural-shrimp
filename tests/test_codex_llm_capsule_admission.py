"""Whole completed-capsule admission using tiny synthetic saved evidence.

No loader or authentication function is mocked. Only model dimensions, owned
paths, and environment metadata are small fixtures; no checkpoint/model is read.
The fixture runs the real proposal writer and capsule writer, then admits their
retained output through the complete receipt/source/worker/capsule reader.
"""
import copy
import json
from pathlib import Path
import sys

import pytest
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codex/certify-llm'))
import center_continue as driver

parent, saved, base = driver.parent, driver.saved, driver.base
worker, scheduling = driver.worker, driver.scheduling
PERIOD = 8
COMMIT = 'a' * 40
ASSETS = {'synthetic_weights': 'b' * 64}
PREFIX_IDS = [1, 2]
TORCH_VERSION = 'synthetic-no-torch-import'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(path, value)


def mutate(path, update):
    value = saved.read_json(path)
    update(value)
    write(path, value)


def synthetic_state(index, centers):
    # Deliberately retain nonzero, tiny image radii through all three layers of
    # saved image / saved residual / reconstructed residual admission.
    images = [[arb((c[0] / 2 + arb(1) / 8).mid(), (1, -1800))] for c in centers]
    records = saved.center_records(images)
    provenance = [dict(phase_index=i, input_sha256='c' * 64, point_sha256='d' * 64)
                  for i in range(PERIOD)]
    return parent.tuple_from_records(saved.center_records(centers), records, records,
                                     index=index, provenance=provenance)


def save_synthetic_worker_pass(candidate, request):
    """Write finite synthetic images with the *actual* worker artifact schema."""
    request_sha = base.digest(candidate / 'request.json')
    report = dict(status='completed', request_sha256=request_sha,
                  source_commit=COMMIT, worker_sha256=request['source_sha256'][scheduling.WORKER_KEY],
                  scheduler_sha256=base.digest(Path(scheduling.__file__)),
                  max_workers=4, worker_timeout_seconds=600, reserve_seconds=600,
                  aggregate_rss_bytes=32 * 1024**3, minimum_available_bytes=12 * 1024**3,
                  sampled_rss_is_hard_cap=False, scientific_endpoint_audit_performed=False,
                  certificate_passed=False, unstarted_phases=[], phase_receipts=[],
                  termination_verification=[])
    centers = saved.decoded_centers(request['centers'], PERIOD, 1)
    for i in range(PERIOD):
        out = candidate / f'phase-{i:02d}'
        out.mkdir()
        (out / 'request.json').write_bytes((candidate / 'request.json').read_bytes())
        write(out / 'inputs.json', dict(request_sha256=request_sha, source_commit=COMMIT,
            phase_index=i, map_sha256=request['map_sha256'], backend_sha256=request['backend_sha256'],
            source_sha256=request['source_sha256'], certificate_passed=False))
        write(out / 'input.json', dict(phase_index=i, center=request['centers'][i],
                                      next_center=request['centers'][(i + 1) % PERIOD]))
        zero = [base.packed(arb(0))]
        write(out / 'live-input.json', dict(phase_index=i, center=request['centers'][i], tangent=zero))
        value = arb((centers[i][0] / 2 + arb(1) / 8).mid(), (1, -1800))
        write(out / 'point.json', dict(value=[base.packed(value)], tangent=zero))
        write(out / 'prefix-binding.json', dict(receipt_sha256=request['prefix_receipt_sha256'],
            graph_sha256=request['prefix_map_source_sha256'], prefix_sha256=request['prefix_cache_sha256'],
            precision=2048, receiver_construction_checked=True))
        events = [dict(event=event, layer=layer) for layer in range(2)
                  for event in ('step_start', 'step_end')]
        (out / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
        write(out / 'status.json', dict(status='phase_point_completed', phase_index=i,
            request_sha256=request_sha, certificate_passed=False, input_sha256=base.digest(out / 'input.json'),
            point_sha256=base.digest(out / 'point.json'), live_input_sha256=base.digest(out / 'live-input.json')))
        receipt = dict(status='completed', phase_index=i, exit_code=0, verified_exit=True,
            request_sha256=request_sha, worker_sha256=request['source_sha256'][scheduling.WORKER_KEY],
            certificate_passed=False, artifacts=scheduling.inventory(out))
        receipt_path = candidate / 'parallel-workers' / f'phase-{i:02d}.receipt.json'
        write(receipt_path, receipt)
        report['phase_receipts'].append(dict(phase_index=i, status='completed', receipt_sha256=base.digest(receipt_path)))
        report['termination_verification'].append(dict(phase_index=i, leader_exit_code=0,
            verified_termination=True, remaining_observed_owned_processes=[]))
    write(candidate / 'parallel-pass.json', report)


class CompleteFixture:
    def __init__(self, attempt, current_plan, binding):
        self.attempt, self.plan, self.binding = attempt, current_plan, binding
        self.out = attempt / 'run'
        self.candidate = self.out / 'candidate-01'

    def reseal(self):
        """Rebind modified fixture bytes, so refusals test semantics, not stale SHA."""
        capsule_sha = base.digest(self.out / 'continuation-capsule.json')
        mutate(self.out / 'status.json', lambda status: status.update(capsule_sha256=capsule_sha))
        self.plan['parent_capsule_sha256'] = capsule_sha
        source = self.attempt / 'source'
        rows = [dict(path=str(path.relative_to(source)), bytes=path.stat().st_size, sha256=base.digest(path))
                for path in sorted(source.rglob('*')) if path.is_file()]
        write(self.attempt / 'manifest.json', dict(source_commit=COMMIT, source=rows))
        artifacts = [dict(path=str(path.relative_to(self.attempt)), bytes=path.stat().st_size,
                          sha256=base.digest(path)) for path in sorted(self.attempt.rglob('*'))
                     if path.is_file() and path != self.attempt / 'receipt.json']
        write(self.attempt / 'receipt.json', dict(source_commit=COMMIT, exit_code=0,
                                                 timed_out=False, artifacts=artifacts))
        self.plan['parent_receipt_sha256'] = base.digest(self.attempt / 'receipt.json')

    def load(self):
        return parent.load_capsule(self.plan, ASSETS, PREFIX_IDS, TORCH_VERSION, self.binding)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    old = ctx.prec, ctx.threads
    ctx.prec, ctx.threads = 2048, 1
    monkeypatch.setattr(base, 'PERIOD', PERIOD)
    monkeypatch.setattr(base, 'DIMENSION', 1)
    monkeypatch.setattr(base, 'OWNED', tmp_path)
    monkeypatch.setattr(driver, 'OWNED', tmp_path)
    monkeypatch.setattr(worker, 'PERIOD', PERIOD)
    monkeypatch.setattr(worker, 'DIMENSION', 1)
    monkeypatch.setattr(worker, 'LAYERS', 2)
    monkeypatch.setattr(worker, 'OWNED', tmp_path)
    monkeypatch.setenv('CERT_SOURCE_COMMIT', COMMIT)
    prefix = tmp_path / 'prefix'
    (prefix / 'run').mkdir(parents=True)
    (prefix / 'run/prefix-cache.json.gz').write_bytes(b'Synthetic prefix bytes, never decoded.\n')
    seed = tmp_path / 'seed'
    seed.mkdir()
    ancestor = tmp_path / 'ancestor'
    ancestor.mkdir()
    attempt = tmp_path / 'continued'
    out = attempt / 'run'
    out.mkdir(parents=True)
    base_plan = dict(schema='synthetic-upstream-base', prefix_attempt=str(prefix),
        center_seed_attempt=str(seed), prefix_receipt_sha256='e' * 64,
        prefix_map_source_sha256='f' * 64)
    source_pins = {path: base.digest(ROOT / path) for path in driver.SOURCES}
    old_plan = dict(schema=driver.SCHEMA, parent_kind='center_secant', parent_attempt=str(ancestor),
        parent_receipt_sha256='c' * 64, parent_capsule_sha256=None,
        parent_audit={'path': 'codex/certify-llm/synthetic-audit.md', 'sha256': 'd' * 64},
        base_plan=base_plan, source_sha256=source_pins)
    before = synthetic_state(3, [[arb(2 * i + 1)] for i in range(PERIOD)])
    after = synthetic_state(4, [[arb(3 * i + 2)] for i in range(PERIOD)])
    admitted = dict(before=before, after=after, history=[], factor_references=[],
        pending_pair_sha256=parent.pending_identity(before, after), archive_links=[],
        prior_phase_evaluations=40, admission={'kind': 'synthetic-complete-older-history'})
    derivatives = [arb_mat([[arb(1) / 2]]) for _ in range(PERIOD)]
    inverse = arb_mat([[arb(5) / 4]])
    proposal, history, refs = driver.prepare_candidate(admitted, derivatives, inverse, out,
        lambda *args, **kwargs: None, source_commit=COMMIT, attempt=attempt)
    request, request_sha = driver.freeze_request(old_plan, proposal['centers'], out / 'candidate-01',
                                                 ASSETS, PREFIX_IDS, TORCH_VERSION, COMMIT)
    save_synthetic_worker_pass(out / 'candidate-01', request)
    write(out / 'request-binding.json', dict(request_sha256=request_sha, centers_sha256=request['centers_sha256']))
    comparison, capsule = driver.finish_candidate(admitted, proposal, history, refs, out,
        old_plan, ASSETS, PREFIX_IDS, TORCH_VERSION, lambda *args, **kwargs: None)
    assert history, 'fixture must exercise a nonempty post-proposal factor history'
    for path in driver.SOURCES:
        dest = attempt / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / path).read_bytes())
    plan_path = attempt / 'source/codex/certify-llm/synthetic-parent-plan.json'
    write(plan_path, old_plan)
    binding = dict(prefix_source='1' * 40, seed_source='2' * 40)
    write(out / 'inputs.json', dict(source_commit=COMMIT, precision=2048, threads=1,
        certificate_passed=False, pinned_assets=ASSETS, prefix_ids=PREFIX_IDS,
        versions=parent.backend_versions(TORCH_VERSION), plan=old_plan, plan_sha256=base.digest(plan_path),
        driver_sha256=source_pins['codex/certify-llm/center_continue.py'], source_binding=binding))
    write(out / 'status.json', dict(status='proposal_continuation_completed', certificate_passed=False,
        candidates=1, new_phase_evaluations=PERIOD, cumulative_phase_evaluations=48,
        comparison=comparison, capsule_sha256=base.digest(out / 'continuation-capsule.json')))
    current_plan = dict(parent_kind='center_continue', parent_attempt=str(attempt),
        parent_source_sha256=source_pins, base_plan=base_plan,
        parent_audit={'path': 'codex/certify-llm/independent-capsule-audit.md', 'sha256': '3' * 64})
    fixture = CompleteFixture(attempt, current_plan, binding)
    fixture.reseal()
    yield fixture
    ctx.prec, ctx.threads = old


def test_complete_writer_to_capsule_admission_preserves_pending_pair_and_radii(complete):
    admitted = complete.load()
    cap = saved.read_json(complete.out / 'continuation-capsule.json')
    assert admitted['admission']['kind'] == 'audited_capsule'
    assert admitted['admission']['older_ancestry_rehashed'] is False
    assert admitted['before']['record']['sequence_index'] == 4
    assert admitted['after']['record']['sequence_index'] == 5
    assert admitted['prior_phase_evaluations'] == 48
    assert admitted['pending_pair_sha256'] == parent.pending_identity(admitted['before'], admitted['after'])
    assert parent.factor_records(admitted['history'], admitted['factor_references']) == cap['post_proposal_factors']
    assert all(value.rad() > 0 for state in ('before', 'after')
               for row in admitted[state]['images'] for value in row)
    assert all(value.rad() > 0 for row in admitted['after']['residuals'] for value in row)
    # The new pending pair is previous AFTER -> fresh candidate, not the pair
    # that the completed parent consumed before its proposal.
    consumed = saved.read_json(complete.out / 'factor-state.json')['pending_pair_consumed_once']
    assert consumed != admitted['pending_pair_sha256']
    assert not (Path(complete.plan['base_plan']['center_seed_attempt']) / 'receipt.json').exists()


@pytest.mark.parametrize('fault', [
    'pending_pair', 'map', 'backend', 'factor_radius', 'history_count', 'archive_parent',
    'proposal_residual', 'proposal_correction', 'proposal_clear', 'request_center_binding',
    'before_image_radius', 'cumulative_count', 'captured_source', 'incomplete_status',
])
def test_full_semantic_capsule_corruptions_refused_after_resealing(complete, fault):
    cap_path = complete.out / 'continuation-capsule.json'
    if fault == 'pending_pair':
        mutate(cap_path, lambda cap: cap.update(pending_pair_sha256='0' * 64))
    elif fault == 'map':
        mutate(cap_path, lambda cap: cap['map'].update(steering_layer=13))
    elif fault == 'backend':
        mutate(cap_path, lambda cap: cap['backend'].update(precision=1024))
    elif fault == 'factor_radius':
        mutate(cap_path, lambda cap: cap['post_proposal_factors'][0]['u'][0].update(rad=[1, -1800]))
    elif fault == 'history_count':
        mutate(cap_path, lambda cap: cap.update(active_history_after_proposal=0))
    elif fault == 'archive_parent':
        mutate(cap_path, lambda cap: cap['archive_links'][-1].update(receipt_sha256='0' * 64))
    elif fault == 'proposal_residual':
        mutate(complete.out / 'proposal.json', lambda p: p['residuals'][0].update(mid=[0, 0], rad=[0, 0]))
    elif fault == 'proposal_correction':
        def change(proposal):
            proposal['chosen_correction'][0] = base.packed(arb(123))
            proposal['accelerated_correction'][0] = base.packed(arb(123))
        mutate(complete.out / 'proposal.json', change)
    elif fault == 'proposal_clear':
        mutate(complete.out / 'proposal.json', lambda p: p.update(reason='baseline_no_history'))
    elif fault == 'request_center_binding':
        mutate(complete.out / 'request-binding.json', lambda p: p.update(centers_sha256='0' * 64))
    elif fault == 'before_image_radius':
        mutate(cap_path, lambda cap: cap['before']['images'][0][0].update(rad=[0, 0]))
    elif fault == 'cumulative_count':
        mutate(cap_path, lambda cap: cap.update(cumulative_phase_evaluations=14))
    elif fault == 'captured_source':
        path = complete.attempt / 'source/codex/certify-llm/center_continue.py'
        path.write_bytes(path.read_bytes() + b'\n# Changed captured source\n')
    elif fault == 'incomplete_status':
        mutate(complete.out / 'status.json', lambda status: status.update(status='proposal_continuation_failed'))
    complete.reseal()
    with pytest.raises(ValueError):
        complete.load()


def test_outer_capsule_pin_is_required_even_with_valid_receipt(complete):
    complete.plan['parent_capsule_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='external admitted capsule pin'):
        complete.load()


def test_unlisted_extra_evidence_cannot_hide_outside_complete_receipt(complete):
    (complete.attempt / 'omitted-from-receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory'):
        complete.load()
