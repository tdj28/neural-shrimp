#!/usr/bin/env python3
"""Compile EXP008 exclusively from old atlas/manifests and frozen source files."""
import json
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'src'))
import run_hopfield as base
from hopfield_information_design import load_information_packet, GEOMETRY_FEATURE_NAMES
from hopfield_information_features import BASE_FEATURE_NAMES
from run_hopfield_information import PLAN, PHASES, RADII, validate


def compile_plan():
    source = ROOT/'evidence/EXP-004/attempt-001'
    old_manifests = [ROOT/f'evidence/{eid}/attempt-001/manifest.json' for eid in ['EXP-006', 'EXP-007']]
    packet = load_information_packet(source, *old_manifests)
    development = packet.pop('development_anchors')
    test = packet.pop('anchors')
    anchors = []
    for role, group in [('development', development), ('test', test)]:
        for index, anchor in enumerate(group):
            a = dict(anchor, role=role, original_origin_id=anchor['id'])
            if role == 'development':
                a.update(id=f'development-{index:02d}', anchor=f'development-{index:02d}')
            anchors.append(a)
    # Keep local paths portable without removing the identities/hashes of old inputs.
    for item in packet.get('prior_manifests', []):
        item['path'] = str(Path(item['path']).relative_to(ROOT))
    for item in packet.get('source_files', []):
        if isinstance(item, dict) and 'path' in item and Path(item['path']).is_absolute():
            item['path'] = str(Path(item['path']).relative_to(ROOT))
    old_paths = [source/'receipt.json', source/'manifest.json', source/'selected.json', *old_manifests,
                 *sorted((source/'focus').glob('row-*.npz'))]
    execution = [*sorted((ROOT/'tools').glob('*.py')), *sorted((ROOT/'src').rglob('*.py')),
                 *sorted((ROOT/'native').glob('*.cpp')), ROOT/'requirements.lock', ROOT/'pyproject.toml',
                 ROOT/'docs/EXP-008-protocol.md',
                 *sorted((ROOT/'tests').glob('test_hopfield_information*.py'))]
    entry = lambda path: dict(path=str(path.relative_to(ROOT)), sha256=base.sha(path), bytes=path.stat().st_size)
    p = dict(experiment_id='EXP-008', date='2026-09-13',
        status='Prospectively frozen private exploratory representation ablation; no registry registration',
        source_artifacts=[entry(path) for path in old_paths], source_files=[entry(path) for path in execution],
        selection=packet, anchors=anchors, phases=PHASES, radii=RADII, dt=[.01,.005], primary_dt=.005,
        decision_dt=.005, initial_state=[.1,.1,.1], burn_time=10000, sample_time=2000,
        event_capacity=128, max_peak_period=32, minimum_repeats=4, recurrence_atol=2e-6, recurrence_rtol=2e-5,
        preview_duration=75, probe_duration=300, sample_dt=.1, phase_grid=128, workers=4,
        wall_clock_ceiling_seconds=3600, candidate_count=18, shuffle_shift=5,
        base_feature_names=list(BASE_FEATURE_NAMES), geometry_feature_names=list(GEOMETRY_FEATURE_NAMES),
        planned_targets=7008, planned_previews=3456, bootstrap_samples=10000, bootstrap_seed=20260913,
        lambda_grid=[1e-6,1e-4,1e-2,1,100], lambda_objective='equal-origin leave-one-origin-out prediction MSE; smallest lambda on exact tie',
        required_runtime_packages={'numpy':'2.5.3', 'scipy':'1.18.1'},
        primary='Fine-step pooled baseline minus geometry selected tail RMSE; equals paired reduction in finite-18-action regret',
        chronology=['freeze and push all code/plan', 'all development previews and targets', 'fit and seal development model',
                    'all test previews', 'seal all test predictions/actions', 'all test target outcomes', 'frozen summary and verification'],
        candidate_rule='Ordered uniform16 exact axes, then local+/-; duplicates retained and charged; identical access for all policies',
        claim_boundary='Incremental usable representation for fixed ridge learner in seen three-neuron atlas. Not new information conditional on coordinates, topology-specific mechanism, independent population, unseen network, training benefit, memory capacity, net speedup or sequential control.',
        primary_failure_rule='All7008 targets at both steps including shams required; any required incomplete outcome makes primary undefined. No complete-case fitting/oracle, replacements or repeated attempts.',
        display_contract={'map':'Every attempted development/test origin and test pixel group; old-map geometry coverage',
                          'utility':'All policy means by dose at both steps; pooled and all-block contrasts; finite-action oracle headroom; all development CV curves',
                          'example':dict(anchor=test[0]['id'],phase=675,radius=.016,dt=.005,
                                         panels='Three activation waveforms and all18 preview/target errors; no outcome-dependent selection')},
        paid_compute=False, paid_review=False)
    validate(p, check_sources=True)
    return p

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    p = compile_plan(); base.write(PLAN, p)
    print(json.dumps(dict(plan=str(PLAN), targets=validate(p), previews=p['planned_previews'],
                         test_anchors=[a['id'] for a in p['anchors'] if a['role']=='test'])))
