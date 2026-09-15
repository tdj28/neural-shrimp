#!/usr/bin/env python3
"""EXP-011 retrospective four-preview feasibility; never a fresh confirmation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_hopfield_refinement as runtime  # numerical thread caps before NumPy
import numpy as np
import preview_shortlist_features as features
import preview_shortlist_model as model

PLAN = ROOT / 'experiments/hopfield/EXP-011-plan.json'
LABEL_ROOT = Path('/data2/neural-shrimp-data/raw/EXP-010/attempt-001')
RESOLUTIONS = ('rk4-0.00125', 'rk4-0.0025')
INDEPENDENT = ('dop853-1e-11', 'dop853-1e-12')
SCHEMAS = ('shared', 'geometry')
CONTROLS = dict(axes=[0, 4, 8, 12], diagonal=[2, 6, 10, 14], local_coverage=[0, 8, 16, 17])
SOURCE_PATHS = list(dict.fromkeys(['tools/run_preview_shortlist.py', 'tools/preview_shortlist_model.py',
    'tests/test_preview_shortlist.py', 'tests/test_preview_shortlist_features.py',
    'docs/EXP-011-protocol.md', 'docs/EXP-010-qualification.json', *features.SOURCE_PATHS, *runtime.SOURCE_PATHS]))
CONTRACT = dict(schema_version='exp011-shortlist-v1', candidate_count=18, shortlist_size=4,
    expected_origins=24, expected_contexts=192, expected_rows=3456, expected_outer_folds=24,
    expected_inner_folds=23, resolutions=list(RESOLUTIONS), canonical_resolution=RESOLUTIONS[0],
    model_schemas=list(SCHEMAS), lambda_grid=list(model.LAMBDA_GRID), controls=CONTROLS,
    numerical_score_atol=model.CHECK_ATOL, numerical_score_rtol=model.CHECK_RTOL,
    bootstrap_resamples=20000, bootstrap_seed=20260914, bootstrap_percentiles=[2.5, 97.5],
    workers=4, thread_limit=1, per_task_ceiling_seconds=600, wall_clock_ceiling_seconds=3600,
    output_reserve_bytes=1024**3, required_runtime_packages={'numpy': '2.5.3', 'scipy': '1.18.1'},
    paid_compute=False, paid_review=False)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime.write(path, value)


def external_binding(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('Qualified labels must remain external')
    return dict(path=str(path), sha256=runtime.sha(path), bytes=path.stat().st_size)


def feature_identity(bundle):
    return dict(schema_version=bundle['schema_version'], shared_names=bundle['shared_names'],
        geometry_names=bundle['geometry_names'], rows=bundle['rows'],
        shared_sha256=hashlib.sha256(np.asarray(bundle['shared'], dtype='<f8').tobytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(np.asarray(bundle['geometry'], dtype='<f8').tobytes()).hexdigest())


def validate_feature_census(bundle):
    rows = bundle['rows']
    if (len(rows) != 3456 or len({r['id'] for r in rows}) != 3456
            or len({r['group'] for r in rows}) != 24
            or np.shape(bundle['shared']) != (3456, 37) or np.shape(bundle['geometry']) != (3456, 6)
            or bundle['shared_names'] != list(features.SHARED_FEATURE_NAMES)
            or bundle['geometry_names'] != list(features.GEOMETRY_FEATURE_NAMES)
            or bundle['schema_version'] != features.SCHEMA_VERSION
            or not np.isfinite(bundle['shared']).all() or not np.isfinite(bundle['geometry']).all()):
        raise ValueError('Complete original feature census/schema required')
    contexts = defaultdict(list)
    for row in rows:
        contexts[model.context_key(row)].append(row['candidate'])
    if len(contexts) != 192 or any(len(v) != 18 or set(v) != set(range(18)) for v in contexts.values()):
        raise ValueError('All 192 contexts require exactly 18 candidate entries')
    if any(n != 144 for n in Counter(r['group'] for r in rows).values()):
        raise ValueError('Every physical origin requires its complete phase/radius product')


def qualification_certificate(certificate):
    expected = dict(schema_version='EXP-010-qualification-certificate-v1', experiment_id='EXP-010',
        status='qualified_at_tested_resolutions', planned_traces=6944, complete_traces=6944,
        preview_cases=3456, native_scalar_qualified=3456, native_waveform_qualified=3456,
        independent_cases=16, independent_scalar_qualified=16, independent_waveform_qualified=16,
        contexts=192, all_native_rankings_unchanged=True, all_required_methods_complete=True,
        all_scalar_and_waveform_checks_pass=True)
    if any(certificate.get(k) != v for k, v in expected.items()):
        raise ValueError('EXP-010 complete scalar/waveform qualification certificate required')
    return certificate


def build_plan():
    bundle = features.load_features(root=ROOT)
    validate_feature_census(bundle)
    receipt_path = LABEL_ROOT / 'receipt.json'
    receipt = read(receipt_path)  # inventory/status only: no preview scores
    if receipt['status'] != 'complete' or receipt['expected_records'] != 6944 or receipt['observed_records'] != 6944:
        raise ValueError('Complete EXP-010 receipt is required')
    inventory = {r['path']: r for r in receipt['artifacts']}
    qualification = qualification_certificate(read(ROOT / 'docs/EXP-010-qualification.json'))
    if qualification['source_commit'] != receipt['source_commit']:
        raise ValueError('Qualification source commit differs')
    external_artifacts = qualification['artifacts'] + qualification['independent_audits']
    for item in external_artifacts:
        if external_binding(item['path']) != item:
            raise ValueError('Qualification certificate artifact binding differs')
    for name in ['manifest.json', 'summary.json', 'terminal-records.json']:
        item = external_binding(LABEL_ROOT / name)
        if (item['sha256'], item['bytes']) != (inventory[name]['sha256'], inventory[name]['bytes']):
            raise ValueError('EXP-010 receipt membership differs')
    labels, independent = {}, []
    for row in bundle['rows']:
        labels[row['id']] = {}
        for resolution in (*RESOLUTIONS, *INDEPENDENT):
            name = f'records/{row["id"]}--{resolution}.json'
            if name not in inventory:
                if resolution in RESOLUTIONS:
                    raise ValueError('Missing mandatory qualified label record')
                continue
            item = external_binding(LABEL_ROOT / name)
            if (item['sha256'], item['bytes']) != (inventory[name]['sha256'], inventory[name]['bytes']):
                raise ValueError('EXP-010 label record binding differs')
            labels[row['id']][resolution] = item
        if any(r in labels[row['id']] for r in INDEPENDENT):
            if not all(r in labels[row['id']] for r in INDEPENDENT):
                raise ValueError('Incomplete independent label panel')
            independent.append(row['id'])
    if len(independent) != 16:
        raise ValueError('All 16 predeclared independent sentinels required')
    return dict(experiment_id='EXP-011', date='2026-09-14', **CONTRACT,
        status='Retrospective development-only shortlist feasibility; no untouched test or new utility confirmation',
        rows=bundle['rows'], feature_identity=feature_identity(bundle),
        feature_source_bindings=bundle['source_bindings'],
        source_files=[runtime.binding(ROOT / name) for name in SOURCE_PATHS],
        label_records=labels, independent_case_ids=independent,
        qualification=qualification,
        qualification_certificate_binding=runtime.binding(ROOT / 'docs/EXP-010-qualification.json'),
        label_source_commit=receipt['source_commit'],
        external_artifacts=external_artifacts,
        default_data_root=str(runtime.DEFAULT_DATA_ROOT),
        scope='All 24 old physical origins are development. The endpoint is 75-unit error; '
              '300-unit labels, geometry confirmation, and net speedup remain untested.')


def validate(plan, *, check_bindings=False, reconstruct=False):
    if plan['experiment_id'] != 'EXP-011':
        raise ValueError('Wrong experiment')
    for name, value in CONTRACT.items():
        if plan[name] != value:
            raise ValueError(f'Frozen contract differs: {name}')
    rows = plan['rows']
    if (len(rows) != 3456 or len({r['id'] for r in rows}) != 3456
            or len({r['group'] for r in rows}) != 24
            or set(plan['label_records']) != {r['id'] for r in rows}
            or len(plan['independent_case_ids']) != 16
            or plan['feature_identity']['rows'] != rows
            or [r['path'] for r in plan['source_files']] != SOURCE_PATHS):
        raise ValueError('Frozen identity/source census differs')
    independent = []
    for row in rows:
        mapping = plan['label_records'][row['id']]
        if not set(RESOLUTIONS) <= set(mapping) or set(mapping) - set((*RESOLUTIONS, *INDEPENDENT)):
            raise ValueError('Label-resolution census differs')
        if set(INDEPENDENT) & set(mapping):
            if not set(INDEPENDENT) <= set(mapping):
                raise ValueError('Independent pair incomplete')
            independent.append(row['id'])
        for resolution, item in mapping.items():
            if Path(item['path']).name != f'{row["id"]}--{resolution}.json':
                raise ValueError('Label file identity differs')
    qualification_certificate(plan['qualification'])
    if plan['external_artifacts'] != plan['qualification']['artifacts'] + plan['qualification']['independent_audits']:
        raise ValueError('Qualification external source binding differs')
    if independent != plan['independent_case_ids']:
        raise ValueError('Independent identities differ')
    if check_bindings:
        for item in plan['source_files'] + plan['feature_source_bindings'] + [plan['qualification_certificate_binding']]:
            if runtime.binding(ROOT / item['path']) != item:
                raise ValueError(f'Source/feature input binding differs: {item["path"]}')
        for item in [*plan['external_artifacts'], *[b for v in plan['label_records'].values() for b in v.values()]]:
            if external_binding(item['path']) != item:
                raise ValueError('External qualified-label binding differs')
    if reconstruct and plan != build_plan():
        raise ValueError('Plan reconstruction differs')
    return dict(origins=24, contexts=192, cases=3456, outer_folds=24, inner_folds=23,
                model_recipes=96, required_label_records=6944)


class LabelStore:
    """Training and reveal are separate capabilities; no wildcard/full-label API."""
    def __init__(self, rows, bindings, *, allowed_training_groups=(), held_group=None, log_path=None):
        self.rows = {r['id']: r for r in rows}
        self.bindings = bindings
        self.allowed = frozenset(allowed_training_groups)
        self.held = held_group
        self.log_path = Path(log_path) if log_path else None
        self.events = []
        if held_group in self.allowed:
            raise ValueError('Held physical origin appears in allowed training origins')

    def _event(self, event):
        event = dict(event, event_index=len(self.events), monotonic_ns=time.monotonic_ns())
        self.events.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def _values(self, ids, resolution):
        values = []
        for ident in ids:
            item = self.bindings[ident][resolution]
            if external_binding(item['path']) != item:
                raise ValueError('Changed qualified-label source')
            row = read(item['path'])
            if (row['id'] != f'{ident}--{resolution}' or row['case_id'] != ident
                    or row['method'] != resolution or row['status'] != 'complete'
                    or isinstance(row['rmse'], bool) or not isinstance(row['rmse'], (int, float))
                    or not math.isfinite(row['rmse']) or not 0 <= row['rmse'] <= 2
                    or row['metrics']['aligned_rms']['aggregate'] != row['rmse']):
                raise ValueError('Qualified-label identity, status or score invalid')
            values.append(row['rmse'])
        return np.asarray(values, dtype=float)

    def training(self, ids, resolution):
        ids = list(ids)
        if (resolution not in RESOLUTIONS or len(set(ids)) != len(ids)
                or any(self.rows[i]['group'] not in self.allowed for i in ids)):
            raise ValueError('Training label access outside allowed physical origins')
        self._event(dict(stage='training', resolution=resolution, ids=ids,
                         groups=sorted({self.rows[i]['group'] for i in ids}), held_group=self.held))
        return self._values(ids, resolution)

    def reveal(self, ids, resolution, seal_path, *, purpose):
        ids = list(ids)
        seal = validate_seal(seal_path, self.rows, expected_group=self.held)
        if (purpose not in ('selected_four', 'full18_evaluation', 'independent_diagnostic')
                or len(set(ids)) != len(ids) or any(self.rows[i]['group'] != self.held for i in ids)):
            raise ValueError('Reveal access outside sealed held physical origin')
        if purpose == 'selected_four':
            allowed = [[i for context in packet for i in context] for name, packet in seal['shortlist_ids'].items()
                       if name in CONTROLS or name.startswith(resolution + '--')]
            if ids not in allowed:
                raise ValueError('Unselected label requested in selected-only reveal stage')
        else:
            validate_winner_seal(Path(seal_path).parent, seal_path)
            if purpose == 'full18_evaluation' and (resolution not in RESOLUTIONS or ids != seal['held_ids']):
                raise ValueError('Full reference requires the exact complete held census')
            if purpose == 'independent_diagnostic' and resolution not in INDEPENDENT:
                raise ValueError('Independent diagnostic resolution differs')
        self._event(dict(stage='reveal', purpose=purpose, resolution=resolution, ids=ids,
                         held_group=self.held, seal_sha256=runtime.sha(seal_path)))
        return self._values(ids, resolution)


def matrix(bundle, schema):
    if schema == 'shared':
        return np.asarray(bundle['shared']), list(bundle['shared_names'])
    if schema == 'geometry':
        return np.column_stack((bundle['shared'], bundle['geometry'])), [*bundle['shared_names'], *bundle['geometry_names']]
    raise ValueError('Unknown feature schema')


def tasks(plan):
    return [dict(id=f'outer-{i:02d}', case=dict(id=group), method=dict(id='nested-ridge-both-resolutions'))
            for i, group in enumerate(sorted({r['group'] for r in plan['rows']}))]


def recipe_name(resolution, schema):
    return f'{resolution}--{schema}'


def seal_fold(folder, task, plan, bundle):
    held_rows = [r for r in plan['rows'] if r['group'] == task['case']['id']]
    artifacts, shortlist_ids = [], {}
    for resolution in RESOLUTIONS:
        for schema in SCHEMAS:
            name = recipe_name(resolution, schema)
            path = folder / f'{name}.json'
            packet = read(path)
            artifacts.append(runtime.binding(path, folder))
            shortlist_ids[name] = [c['top4_ids'] for c in packet['shortlists_svd']]
    by_context = defaultdict(dict)
    for row in held_rows:
        by_context[model.context_key(row)][row['candidate']] = row['id']
    for policy, candidates in CONTROLS.items():
        shortlist_ids[policy] = [[entries[c] for c in candidates] for _, entries in sorted(by_context.items())]
    seal = dict(schema_version='exp011-before-held-reveal-v1', fold_id=task['id'],
        held_group=task['case']['id'], held_ids=[r['id'] for r in held_rows],
        feature_identity_sha256=model.digest(feature_identity(bundle)),
        artifacts=artifacts, shortlist_ids=shortlist_ids, controls=CONTROLS,
        created_at=runtime.now(), monotonic_ns=time.monotonic_ns(), pid=os.getpid())
    write(folder / 'pre-reveal-seal.json', seal)
    return seal


def validate_seal(path, rows, *, expected_group=None):
    path = Path(path)
    seal = read(path)
    indexed = rows if isinstance(rows, dict) else {r['id']: r for r in rows}
    expected_ids = [r['id'] for r in indexed.values() if r['group'] == seal['held_group']]
    expected_recipes = {recipe_name(r, s) for r in RESOLUTIONS for s in SCHEMAS}
    if (seal['held_group'] != expected_group or seal['controls'] != CONTROLS
            or seal['held_ids'] != expected_ids
            or set(seal['shortlist_ids']) != expected_recipes | set(CONTROLS)
            or len(seal['artifacts']) != 4
            or {r['path'] for r in seal['artifacts']} != {n + '.json' for n in expected_recipes}):
        raise ValueError('Incomplete or changed pre-reveal seal')
    for item in seal['artifacts']:
        if runtime.binding(path.parent / item['path'], path.parent) != item:
            raise ValueError('Sealed model/prediction bytes changed')
        packet = read(path.parent / item['path'])
        digest = packet.pop('packet_sha256')
        if (model.digest(packet) != digest or packet['held_group'] != expected_group
                or packet['held_row_ids'] != expected_ids
                or expected_group in packet['training_groups']):
            raise ValueError('Invalid model packet in pre-reveal seal')
        name = item['path'][:-5]
        expected_lists = model.shortlist_rows([indexed[i] for i in expected_ids], packet['held_predictions_svd'])
        if packet['shortlists_svd'] != expected_lists or seal['shortlist_ids'][name] != [c['top4_ids'] for c in expected_lists]:
            raise ValueError('Sealed shortlist is not the exact four predicted entries')
    expected_contexts = sorted({model.context_key(indexed[i]) for i in expected_ids})
    for name, contexts in seal['shortlist_ids'].items():
        actual_contexts = [model.context_key(indexed[ids[0]]) for ids in contexts if ids]
        if actual_contexts != expected_contexts:
            raise ValueError('Sealed context bijection or deterministic context order differs')
        if len(contexts) != len(expected_ids) // 18:
            raise ValueError('Sealed decision context count differs')
        for selected in contexts:
            if (len(selected) != 4 or len(set(selected)) != 4 or not set(selected) <= set(expected_ids)
                    or len({model.context_key(indexed[i]) for i in selected}) != 1):
                raise ValueError('Every sealed shortlist requires four distinct entries in one context')
            if name in CONTROLS and [indexed[i]['candidate'] for i in selected] != CONTROLS[name]:
                raise ValueError('Fixed control shortlist changed')
    return seal


def fold_worker(task, context, directory, library, parent_pid):
    begin = time.monotonic()
    folder = directory / 'folds' / task['id']
    folder.mkdir(parents=True, exist_ok=False)
    plan, bundle = context['plan'], context['bundle']
    group = task['case']['id']
    row = dict(id=task['id'], case_id=group, method=task['method']['id'], status='failed',
               pid=os.getpid(), started_at=runtime.now(), completed_recipes=[])
    try:
        training = [i for i, r in enumerate(bundle['rows']) if r['group'] != group]
        held = [i for i, r in enumerate(bundle['rows']) if r['group'] == group]
        train_rows, held_rows = [bundle['rows'][i] for i in training], [bundle['rows'][i] for i in held]
        groups = [r['group'] for r in train_rows]
        store = LabelStore(bundle['rows'], plan['label_records'], allowed_training_groups=set(groups),
                           held_group=group, log_path=folder / 'label-access.jsonl')
        all_numerical = True
        for resolution in RESOLUTIONS:
            labels = store.training([r['id'] for r in train_rows], resolution)
            write(folder / f'training-{resolution}.json', dict(ids=[r['id'] for r in train_rows],
                groups=groups, values=labels.tolist(), source='permitted EXP-010 per-case score records'))
            for schema in SCHEMAS:
                start = time.monotonic()
                x, names = matrix(bundle, schema)
                stage_timings = {}
                packet = model.fit_nested(x[training], labels, groups, names, x[held], held_rows, stage_timings=stage_timings)
                name = recipe_name(resolution, schema)
                write(folder / f'{name}.json', packet)
                row['completed_recipes'].append(dict(name=name, seconds=time.monotonic() - start, stage_timings=stage_timings,
                    numerical_agreement=packet['numerical_audit']['passed'], packet_sha256=packet['packet_sha256']))
                all_numerical &= packet['numerical_audit']['passed']
                write(folder / 'progress.json', row)
        seal_fold(folder, task, plan, bundle)
        validate_seal(folder / 'pre-reveal-seal.json', bundle['rows'], expected_group=group)
        row.update(status='complete' if all_numerical else 'numerical_disagreement',
                   seal=f'folds/{task["id"]}/pre-reveal-seal.json',
                   seal_sha256=runtime.sha(folder / 'pre-reveal-seal.json'), numerical_agreement=all_numerical)
    except BaseException as exc:
        row['error'] = repr(exc)
        write(folder / 'failure.json', dict(error=repr(exc), completed_recipes=row['completed_recipes']))
    row.update(ended_at=runtime.now(), wall_seconds=time.monotonic() - begin)
    write(directory / 'records' / f'{task["id"]}.json', row)


def make_winner_seal(folder, seal_path, by_id):
    selected = read(folder / 'selected-label-reveals.json')
    winners = {}
    for name, packet in selected['data'].items():
        rows = []
        for start in range(0, len(packet['ids']), 4):
            pairs = list(zip(packet['ids'][start:start + 4], packet['values'][start:start + 4]))
            if len(pairs) != 4 or len({i for i, _ in pairs}) != 4:
                raise ValueError('Winner requires exactly four observed entries')
            ident, score = min(pairs, key=lambda pair: (pair[1], by_id[pair[0]]['candidate']))
            rows.append(dict(context=list(model.context_key(by_id[ident])), chosen_id=ident,
                             chosen_candidate=by_id[ident]['candidate'], rmse=score))
        winners[name] = rows
    return dict(schema_version='exp011-four-observed-winner-v1', pre_reveal_seal_sha256=runtime.sha(seal_path),
                selected_reveals_sha256=runtime.sha(folder / 'selected-label-reveals.json'),
                winners=winners, monotonic_ns=time.monotonic_ns())


def validate_winner_seal(folder, seal_path, *, by_id=None):
    winner = read(folder / 'winner-seal.json')
    if (winner['pre_reveal_seal_sha256'] != runtime.sha(seal_path)
            or winner['selected_reveals_sha256'] != runtime.sha(folder / 'selected-label-reveals.json')):
        raise ValueError('Missing or changed winner seal before oracle reveal')
    if by_id is not None:
        expected = make_winner_seal(folder, seal_path, by_id)
        if {k: v for k, v in winner.items() if k != 'monotonic_ns'} != {k: v for k, v in expected.items() if k != 'monotonic_ns'}:
            raise ValueError('Winner seal is not minimum of the four observed labels')
    return winner


def observe_shortlist(rows, labels, selected):
    indexed = {r['candidate']: (r['id'], float(value)) for r, value in zip(rows, labels)}
    if len(indexed) != 18 or set(indexed) != set(range(18)) or len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError('Four-of-18 evaluation identity differs')
    ranked = sorted(indexed, key=lambda c: (indexed[c][1], c))
    chosen = min(selected, key=lambda c: (indexed[c][1], c))
    full = ranked[0]
    return dict(selected=list(selected), chosen_candidate=chosen, chosen_id=indexed[chosen][0],
        rmse=indexed[chosen][1], full18_min_rmse=indexed[full][1], full18_winner=full,
        excess_rmse=indexed[chosen][1] - indexed[full][1], winner_inclusion=full in selected,
        any_minimizer_inclusion=any(indexed[c][1] == indexed[full][1] for c in selected),
        best_selected_rank=min(ranked.index(c) + 1 for c in selected),
        mean_selected_rank=float(np.mean([ranked.index(c) + 1 for c in selected])),
        required_preview_calls=4, full18_preview_calls=18)


def numerical_context(rows, canonical, comparison):
    ca = np.asarray(canonical, float)
    co = np.asarray(comparison, float)
    order_a = sorted(range(18), key=lambda i: (ca[i], rows[i]['candidate']))
    order_b = sorted(range(18), key=lambda i: (co[i], rows[i]['candidate']))
    return dict(signed_label_changes=(ca - co).tolist(),
        empirical_max_absolute_score_change=float(np.max(np.abs(ca - co))),
        envelope_is_certified_error_bound=False,
        canonical_best_second_gap=float(ca[order_a[1]] - ca[order_a[0]]),
        canonical_fourth_fifth_gap=float(ca[order_a[4]] - ca[order_a[3]]),
        comparison_best_second_gap=float(co[order_b[1]] - co[order_b[0]]),
        comparison_fourth_fifth_gap=float(co[order_b[4]] - co[order_b[3]]),
        canonical_order=[rows[i]['candidate'] for i in order_a],
        comparison_order=[rows[i]['candidate'] for i in order_b],
        full_ranking_changed=order_a != order_b,
        oracle_argmin_changed=order_a[0] != order_b[0],
        oracle_minimum_signed_change=float(ca[order_a[0]] - co[order_b[0]]))


def evaluate(plan, bundle, records, directory, *, persist):
    """Runs only after the fitting phase. Every holdout access requires four sealed recipes."""
    result, independent, failures = [], [], []
    all_rows = bundle['rows']
    by_id = {r['id']: r for r in all_rows}
    for row in records:
        folder = directory / 'folds' / row['id']
        if 'seal' not in row:
            failures.append(dict(fold_id=row['id'], group=row['case_id'], status=row['status']))
            continue
        path = directory / row['seal']
        if runtime.sha(path) != row['seal_sha256']:
            raise ValueError('Terminal row seal binding differs')
        seal = validate_seal(path, all_rows, expected_group=row['case_id'])
        store = LabelStore(all_rows, plan['label_records'], held_group=row['case_id'],
                           log_path=folder / 'reveal-access.jsonl' if persist else None)
        selected_reveals, values = {}, {}
        # Reveal only the four sealed entries per policy/context before the
        # separately charged all-eighteen retrospective oracle evaluation.
        for resolution in RESOLUTIONS:
            for schema in SCHEMAS:
                name = recipe_name(resolution, schema)
                ids = [i for context in seal['shortlist_ids'][name] for i in context]
                selected_reveals[name] = dict(ids=ids, values=store.reveal(ids, resolution, path, purpose='selected_four').tolist())
            for policy in CONTROLS:
                ids = [i for context in seal['shortlist_ids'][policy] for i in context]
                selected_reveals[f'{resolution}--{policy}'] = dict(ids=ids,
                    values=store.reveal(ids, resolution, path, purpose='selected_four').tolist())
        selected_packet = dict(seal_sha256=runtime.sha(path), data=selected_reveals)
        if persist:
            write(folder / 'selected-label-reveals.json', selected_packet)
            write(folder / 'winner-seal.json', make_winner_seal(folder, path, by_id))
        else:
            if read(folder / 'selected-label-reveals.json') != selected_packet:
                raise ValueError('Selected-only source labels changed')
            validate_winner_seal(folder, path, by_id=by_id)
        for resolution in RESOLUTIONS:
            vals = store.reveal(seal['held_ids'], resolution, path, purpose='full18_evaluation')
            values[resolution] = dict(zip(seal['held_ids'], vals.tolist()))
        independent_values = {}
        for ident in seal['held_ids']:
            if ident in plan['independent_case_ids']:
                independent_values[ident] = {resolution: float(store.reveal([ident], resolution, path,
                    purpose='independent_diagnostic')[0]) for resolution in INDEPENDENT}
        if persist:
            write(folder / 'full18-label-reveals.json', dict(seal_sha256=runtime.sha(path), data=values,
                                                            independent=independent_values))
        else:
            if (read(folder / 'selected-label-reveals.json') != dict(seal_sha256=runtime.sha(path), data=selected_reveals)
                or read(folder / 'full18-label-reveals.json') != dict(seal_sha256=runtime.sha(path), data=values, independent=independent_values)):
                raise ValueError('Saved label reveal does not reproduce qualified source values')
            expected_events = store.events
            actual_events = [json.loads(line) for line in (folder / 'reveal-access.jsonl').read_text().splitlines()]
            if [{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in actual_events] != [
                    {k: v for k, v in e.items() if k != 'monotonic_ns'} for e in expected_events]:
                raise ValueError('Reveal access sequence differs from sealed four-then-oracle recipe')
            winner_time = read(folder / 'winner-seal.json')['monotonic_ns']
            if (any(e['monotonic_ns'] < seal['monotonic_ns']
                    or (e['purpose'] != 'selected_four' and e['monotonic_ns'] < winner_time)
                    or (e['purpose'] == 'selected_four' and e['monotonic_ns'] > winner_time) for e in actual_events)
                    or [e['monotonic_ns'] for e in actual_events] != sorted(e['monotonic_ns'] for e in actual_events)):
                raise ValueError('Held labels were accessed before the required seal')
        packets = {recipe_name(r, s): read(folder / f'{recipe_name(r, s)}.json') for r in RESOLUTIONS for s in SCHEMAS}
        ctx_rows = defaultdict(list)
        for ident in seal['held_ids']:
            ctx_rows[model.context_key(by_id[ident])].append(by_id[ident])
        for key, rows in sorted(ctx_rows.items()):
            rows.sort(key=lambda r: r['candidate'])
            labels = {r: [values[r][a['id']] for a in rows] for r in RESOLUTIONS}
            entry = dict(context=list(key), role=rows[0]['role'], anchor=rows[0]['anchor'],
                qualified=row['status'] == 'complete', policies={}, numerical_resolution=numerical_context(rows, labels[RESOLUTIONS[0]], labels[RESOLUTIONS[1]]),
                model_resolution_sensitivity={}, solver_sensitivity={})
            for resolution in RESOLUTIONS:
                policies = {}
                for schema in SCHEMAS:
                    packet = packets[recipe_name(resolution, schema)]
                    shortlist = next(c for c in packet['shortlists_svd'] if tuple(c['context']) == key)
                    alternative = next(c for c in packet['shortlists_qr'] if tuple(c['context']) == key)
                    policies[schema] = observe_shortlist(rows, labels[resolution], shortlist['top4'])
                    alternate_result = observe_shortlist(rows, labels[resolution], alternative['top4'])
                    prediction_map = dict(zip(packet['held_row_ids'], packet['held_predictions_svd']))
                    qr_map = dict(zip(packet['held_row_ids'], packet['held_predictions_qr']))
                    scores, qr_scores = [prediction_map[r['id']] for r in rows], [qr_map[r['id']] for r in rows]
                    gap = numerical_context(rows, scores, qr_scores)
                    entry['solver_sensitivity'][recipe_name(resolution, schema)] = dict(
                        predicted_score_diagnostic=gap, numerical_gate=packet['numerical_audit']['passed'],
                        selected_penalty_svd=packet['selected_penalty_svd'], selected_penalty_qr=packet['selected_penalty_qr'],
                        ordered_shortlist_changed=shortlist['top4'] != alternative['top4'],
                        shortlist_set_changed=set(shortlist['top4']) != set(alternative['top4']),
                        observed_selected_error_signed_change=policies[schema]['rmse'] - alternate_result['rmse'],
                        observed_winner_changed=policies[schema]['chosen_candidate'] != alternate_result['chosen_candidate'])
                for policy, shortlist in CONTROLS.items():
                    policies[policy] = observe_shortlist(rows, labels[resolution], shortlist)
                entry['policies'][resolution] = policies
            for schema in SCHEMAS:
                a = entry['policies'][RESOLUTIONS[0]][schema]
                b = entry['policies'][RESOLUTIONS[1]][schema]
                fixed_comparison = observe_shortlist(rows, labels[RESOLUTIONS[1]], a['selected'])
                pa, pb = packets[recipe_name(RESOLUTIONS[0], schema)], packets[recipe_name(RESOLUTIONS[1], schema)]
                map_a, map_b = dict(zip(pa['held_row_ids'], pa['held_predictions_svd'])), dict(zip(pb['held_row_ids'], pb['held_predictions_svd']))
                entry['model_resolution_sensitivity'][schema] = dict(
                    ordered_shortlist_changed=a['selected'] != b['selected'],
                    shortlist_set_changed=set(a['selected']) != set(b['selected']),
                    selected_action_changed=a['chosen_candidate'] != b['chosen_candidate'],
                    selected_error_signed_change=a['rmse'] - b['rmse'],
                    fixed_canonical_shortlist_error_signed_change=a['rmse'] - fixed_comparison['rmse'],
                    fixed_canonical_shortlist_action_changed=a['chosen_candidate'] != fixed_comparison['chosen_candidate'],
                    prediction_diagnostic=numerical_context(rows, [map_a[r['id']] for r in rows], [map_b[r['id']] for r in rows]))
            result.append(entry)
            for candidate in rows:
                ident = candidate['id']
                if ident in independent_values:
                    independent.append(dict(id=ident, group=key[0], context=list(key), candidate=candidate['candidate'],
                        native={r: values[r][ident] for r in RESOLUTIONS}, independent=independent_values[ident],
                        canonical_minus_independent={r: values[RESOLUTIONS[0]][ident] - v for r, v in independent_values[ident].items()},
                        selected_membership={r: {p: candidate['candidate'] in out['selected']
                            for p, out in entry['policies'][r].items()} for r in RESOLUTIONS},
                        scope='Partial scalar sentinel only; no independent 18-entry oracle or independent retraining'))
    return dict(contexts=result, independent=independent, failures=failures)


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ValueError('Descriptive interval requires all 24 equal physical origins')
    rng = np.random.default_rng(CONTRACT['bootstrap_seed'])
    indices = rng.integers(0, 24, size=(CONTRACT['bootstrap_resamples'], 24))
    resampled = np.mean(values[indices], axis=1)
    return dict(mean=float(values.mean()), descriptive_percentile_interval=np.percentile(resampled, [2.5, 97.5]).tolist(),
                origins=24, seed=CONTRACT['bootstrap_seed'], resamples=CONTRACT['bootstrap_resamples'],
                interpretation='Descriptive resampling of these development origins, not confirmatory uncertainty')


def aggregate_contexts(contexts, resolution):
    groups = sorted({c['context'][0] for c in contexts})
    if len(groups) != 24 or len(contexts) not in (48, 192):
        raise ValueError('All-origin equal-weight denominator incomplete')
    expected_per_group = len(contexts) // 24
    per_origin = []
    metrics = ['rmse', 'excess_rmse', 'winner_inclusion', 'any_minimizer_inclusion', 'best_selected_rank', 'mean_selected_rank']
    for group in groups:
        subset = [r for r in contexts if r['context'][0] == group]
        if len(subset) != expected_per_group:
            raise ValueError('Unequal or incomplete origin context census')
        policies = {policy: {metric: float(np.mean([r['policies'][resolution][policy][metric] for r in subset]))
                             for metric in metrics} for policy in (*SCHEMAS, *CONTROLS)}
        per_origin.append(dict(group=group, contexts=len(subset), policies=policies,
                              full18_min_rmse=float(np.mean([r['policies'][resolution]['shared']['full18_min_rmse'] for r in subset]))))
    answer = dict(origins=24, contexts=len(contexts), per_origin=per_origin, policies={}, paired_contrasts={})
    for policy in (*SCHEMAS, *CONTROLS):
        answer['policies'][policy] = {metric: interval([o['policies'][policy][metric] for o in per_origin]) for metric in metrics}
    for a, b in [('shared', 'geometry'), *[(c, s) for c in CONTROLS for s in SCHEMAS]]:
        answer['paired_contrasts'][f'{a}_minus_{b}'] = interval([o['policies'][a]['rmse'] - o['policies'][b]['rmse'] for o in per_origin])
    answer['full18_min_rmse'] = interval([o['full18_min_rmse'] for o in per_origin])
    return answer


def summarize(plan, records, evaluation):
    contexts = evaluation['contexts']
    all_fits = len(records) == 24 and all(r['status'] in ('complete', 'numerical_disagreement') for r in records)
    qualified = all_fits and all(r['status'] == 'complete' for r in records) and len(contexts) == 192 and len(evaluation['independent']) == 16
    observed = None
    if len(contexts) == 192:
        observed = {r: dict(all_origins=aggregate_contexts(contexts, r), by_radius={str(radius): aggregate_contexts(
            [c for c in contexts if c['context'][2] == radius], r) for radius in sorted({c['context'][2] for c in contexts})}) for r in RESOLUTIONS}
    return dict(experiment_id='EXP-011', status='qualified_retrospective_feasibility' if qualified else 'incomplete_or_numerically_unqualified',
        all_required_fits_complete=all_fits, all_required_numerical_agreement=qualified,
        fixed_denominators=dict(origins=24, contexts=192, rows=3456, recipes=96, independent_cases=16),
        observed_contexts=len(contexts), status_counts=dict(Counter(r['status'] for r in records)),
        primary_equal_origin_readout=observed[RESOLUTIONS[0]] if qualified else None,
        comparison_resolution_readout=observed[RESOLUTIONS[1]] if qualified else None,
        unqualified_observed_readouts=observed if not qualified else None,
        numerical_resolution_summary=dict(
            contexts=len(contexts), all_label_rankings_unchanged=all(not c['numerical_resolution']['full_ranking_changed'] for c in contexts) if len(contexts) == 192 else None,
            maximum_label_change=max((c['numerical_resolution']['empirical_max_absolute_score_change'] for c in contexts), default=None),
            models={schema: dict(ordered_shortlist_changes=sum(c['model_resolution_sensitivity'][schema]['ordered_shortlist_changed'] for c in contexts),
                shortlist_set_changes=sum(c['model_resolution_sensitivity'][schema]['shortlist_set_changed'] for c in contexts),
                action_changes=sum(c['model_resolution_sensitivity'][schema]['selected_action_changed'] for c in contexts),
                maximum_selected_error_change=max((abs(c['model_resolution_sensitivity'][schema]['selected_error_signed_change']) for c in contexts), default=None)) for schema in SCHEMAS}),
        evaluations=evaluation,
        cost_contract=dict(preview_calls_per_policy_context=4, full_reference_calls_per_context=18,
            reduction_calls=14, reduction_fraction=14 / 18, new_preview_integrations=0,
            measured_stage_timings='timings.json and per-fold recipe seconds; replay timing is separate',
            prerequisite_dependencies=dict(shared=['reference orbit', 'saved 300-unit unperturbed sensitivity'],
                geometry=['shared prerequisites', 'historical EXP-004 atlas'],
                full18_reference=['reference orbit', 'saved 300-unit unperturbed sensitivity'], axes=['reference orbit'],
                diagonal=['reference orbit'], local_coverage=['reference orbit', 'saved 300-unit unperturbed sensitivity']),
            reused_prerequisites_are_free=False, acquisition_cost_newly_measured=False, net_speedup_claim=False),
        scope='All origins are retrospective development. This is 75-unit shortlist feasibility; no held-out confirmation, '
              '300-unit utility, verified geometry advantage or net speedup is established by this study.')


def external_attempt(data_root, attempt):
    root = Path(data_root).expanduser()
    if not root.is_absolute() or root.resolve().is_relative_to(ROOT):
        raise ValueError('Raw output root must be absolute and external')
    if not attempt.startswith('attempt-') or not attempt[8:].isdigit():
        raise ValueError('Attempt must be attempt-NNN')
    path = (root / 'EXP-011' / attempt).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('Resolved EXP-011 path enters checkout')
    return path


def source_guard(plan_path):
    if runtime.git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Tracked and untracked source must be clean before fitting')
    source = runtime.git('rev-parse', 'HEAD')
    for path in [*SOURCE_PATHS, str(Path(plan_path).resolve().relative_to(ROOT))]:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{source}:{path}'])
        if hashlib.sha256(content).hexdigest() != runtime.sha(ROOT / path):
            raise ValueError(f'Source or plan not committed: {path}')
    branch = runtime.git('symbolic-ref', '--short', 'HEAD')
    if runtime.git('ls-remote', '--exit-code', 'origin', f'refs/heads/{branch}').split()[0] != source:
        raise ValueError('Exact plan and executable source must be pushed before fitting')
    return dict(source_commit=source, pushed_branch=branch, remote_head_verified_at=runtime.now())


def check_records(plan, records, directory):
    expected = {task['id']: task for task in tasks(plan)}
    if len(records) != 24 or {r['id'] for r in records} != set(expected):
        raise ValueError('Terminal outer-fold census differs')
    for row in records:
        task = expected[row['id']]
        if row['case_id'] != task['case']['id'] or row['method'] != task['method']['id']:
            raise ValueError('Terminal fold identity differs')
        if row['status'] in ('complete', 'numerical_disagreement'):
            if (row['seal'] != f'folds/{row["id"]}/pre-reveal-seal.json'
                    or row['seal_sha256'] != runtime.sha(directory / row['seal'])
                    or row != read(directory / 'records' / f'{row["id"]}.json')):
                raise ValueError('Fold record, planned seal path or bytes differ')
            seal = validate_seal(directory / row['seal'], plan['rows'], expected_group=row['case_id'])
            if seal['fold_id'] != row['id'] or seal['feature_identity_sha256'] != model.digest(plan['feature_identity']):
                raise ValueError('Seal fold or feature-input identity differs')
            if len(row['completed_recipes']) != 4 or {r['name'] for r in row['completed_recipes']} != {
                    recipe_name(r, s) for r in RESOLUTIONS for s in SCHEMAS}:
                raise ValueError('Required recipe census differs')
            numeric = []
            for item in row['completed_recipes']:
                packet = read(directory / 'folds' / row['id'] / (item['name'] + '.json'))
                if item['packet_sha256'] != packet['packet_sha256'] or item['numerical_agreement'] != packet['numerical_audit']['passed']:
                    raise ValueError('Recipe packet/numerical status binding differs')
                numeric.append(packet['numerical_audit']['passed'])
            if row['numerical_agreement'] != all(numeric) or (row['status'] == 'complete') != all(numeric):
                raise ValueError('Numerical failures were not retained in fold status')


def replay_models(plan, bundle, records, directory):
    count = 0
    for record in records:
        if record['status'] not in ('complete', 'numerical_disagreement'):
            continue
        group = record['case_id']
        folder = directory / 'folds' / record['id']
        training = [i for i, r in enumerate(bundle['rows']) if r['group'] != group]
        held = [i for i, r in enumerate(bundle['rows']) if r['group'] == group]
        train_rows, held_rows = [bundle['rows'][i] for i in training], [bundle['rows'][i] for i in held]
        groups = [r['group'] for r in train_rows]
        store = LabelStore(bundle['rows'], plan['label_records'], allowed_training_groups=set(groups), held_group=group)
        for resolution in RESOLUTIONS:
            labels = store.training([r['id'] for r in train_rows], resolution)
            if read(folder / f'training-{resolution}.json') != dict(ids=[r['id'] for r in train_rows],
                    groups=groups, values=labels.tolist(), source='permitted EXP-010 per-case score records'):
                raise ValueError('Saved training labels differ from allowed source records')
            for schema in SCHEMAS:
                x, names = matrix(bundle, schema)
                packet = model.fit_nested(x[training], labels, groups, names, x[held], held_rows)
                saved = read(folder / f'{recipe_name(resolution, schema)}.json')
                # Exact packet digest plus full recursive replay is deliberately
                # a same-platform check, not a cross-platform tolerance claim.
                runtime.compare_scoring(saved, packet, f'{record["id"]}/{resolution}/{schema}')
                count += 1
        actual_events = [json.loads(line) for line in (folder / 'label-access.jsonl').read_text().splitlines()]
        if ([{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in actual_events] !=
                [{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in store.events]):
            raise ValueError('Training access census included forbidden or missing origins')
        seal = read(folder / 'pre-reveal-seal.json')
        if any(e['monotonic_ns'] > seal['monotonic_ns'] for e in actual_events):
            raise ValueError('Training access continued after the pre-reveal seal')
    return dict(replayed_recipes=count, expected_complete_recipes=4 * sum(r['status'] in ('complete', 'numerical_disagreement') for r in records))


def require_artifacts(records, directory, paths):
    required = {'manifest.json', 'started.json', 'terminal-records.json', 'timings.json'}
    if any(r['status'] in ('complete', 'numerical_disagreement') for r in records):
        required |= {'feature-identity.json', 'feature-arrays.npz'}
    if all(r['status'] in ('complete', 'numerical_disagreement') for r in records):
        required |= {'summary.json', 'model-replay.json'}
    for row in records:
        if row['status'] not in ('complete', 'numerical_disagreement'):
            continue
        prefix = f'folds/{row["id"]}/'
        required |= {prefix + name for name in ['pre-reveal-seal.json', 'label-access.jsonl', 'reveal-access.jsonl',
             'winner-seal.json', 'selected-label-reveals.json', 'full18-label-reveals.json']}
        required.add(f'records/{row["id"]}.json')
        for resolution in RESOLUTIONS:
            required.add(prefix + f'training-{resolution}.json')
            for schema in SCHEMAS:
                required.add(prefix + recipe_name(resolution, schema) + '.json')
    if not required <= paths:
        raise ValueError(f'Required completed artifacts absent: {sorted(required - paths)}')


def verify(directory, *, replay=True):
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('Raw attempt must remain external')
    receipt = read(directory / 'receipt.json')
    seen = set()
    for item in receipt['artifacts']:
        path = directory / item['path']
        if (not path.resolve().is_relative_to(directory) or item['path'] in seen
                or runtime.binding(path, directory) != item):
            raise ValueError('Raw receipt binding differs')
        seen.add(item['path'])
    if seen != {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'receipt.json'}:
        raise ValueError('Raw inventory census differs')
    plan = read(directory / 'manifest.json')
    validate(plan, check_bindings=True)
    if runtime.sha(directory / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Manifest bytes differ')
    records = read(directory / 'terminal-records.json')
    check_records(plan, records, directory)
    require_artifacts(records, directory, seen)
    if (receipt['expected_records'] != 24 or receipt['observed_records'] != len(records)
            or receipt['status_counts'] != dict(Counter(r['status'] for r in records))
            or (receipt['status'] == 'complete') != (receipt['error'] is None and all(r['status'] == 'complete' for r in records))):
        raise ValueError('Receipt completion/failure census differs')
    checked = 0
    if 'feature-identity.json' in seen:
        bundle = features.load_features(root=ROOT)
        validate_feature_census(bundle)
        if feature_identity(bundle) != plan['feature_identity'] or read(directory / 'feature-identity.json') != plan['feature_identity']:
            raise ValueError('Label-free feature reconstruction differs')
        with np.load(directory / 'feature-arrays.npz', allow_pickle=False) as raw:
            if not np.array_equal(raw['shared'], bundle['shared']) or not np.array_equal(raw['geometry'], bundle['geometry']):
                raise ValueError('Saved feature arrays differ')
        if replay:
            checked = replay_models(plan, bundle, records, directory)['replayed_recipes']
        if 'summary.json' in seen:
            evaluation = evaluate(plan, bundle, records, directory, persist=False)
            if read(directory / 'summary.json') != summarize(plan, records, evaluation):
                raise ValueError('Complete decision/score/summary replay differs')
    return dict(status='verified_saved_shortlist_evidence', outer_folds=24, replayed_recipes=checked,
                artifacts=len(seen), all_complete=receipt['status'] == 'complete')


def run(plan_path, data_root, attempt):
    plan = read(plan_path)
    # Preparation and production decode only the metadata qualification
    # certificate; all label-bearing external artifacts are hash-checked opaque.
    validate(plan, check_bindings=True)
    if sys.platform != 'linux':
        raise ValueError('Frozen owned worker lifecycle requires Linux')
    for name, version in plan['required_runtime_packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Runtime package differs: {name}')
    freeze = source_guard(plan_path)
    directory = external_attempt(data_root, attempt)
    directory.parent.mkdir(parents=True, exist_ok=True)
    space = os.statvfs(directory.parent)
    if space.f_bavail * space.f_frsize < plan['output_reserve_bytes']:
        raise ValueError('External output reserve unavailable')
    directory.mkdir(exist_ok=False)
    (directory / 'records').mkdir()
    shutil.copyfile(plan_path, directory / 'manifest.json')
    started = dict(experiment_id='EXP-011', **freeze, manifest_sha256=runtime.sha(plan_path),
        started_at=runtime.now(), parent_pid=os.getpid(), expected_records=24,
        platform=platform.platform(), python=sys.version, executable=sys.executable, argv=sys.argv, numpy_build_config=np.__config__.CONFIG,
        packages={name: importlib.metadata.version(name) for name in plan['required_runtime_packages']},
        thread_environment={name: os.environ[name] for name in runtime.THREAD_ENV}, workers=4,
        cpu_count=os.cpu_count(), load_average=list(os.getloadavg()), paid_compute=False, paid_review=False)
    write(directory / 'started.json', started)
    begin = time.monotonic()
    deadline = begin + plan['wall_clock_ceiling_seconds']
    handlers = {sig: signal.getsignal(sig) for sig in [signal.SIGTERM, signal.SIGALRM]}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Whole attempt ceiling')))
    signal.setitimer(signal.ITIMER_REAL, plan['wall_clock_ceiling_seconds'])
    error, records, timings = None, [], {}
    try:
        start = time.monotonic()
        bundle = features.load_features(root=ROOT)
        validate_feature_census(bundle)
        if feature_identity(bundle) != plan['feature_identity']:
            raise ValueError('Feature construction differs from frozen plan')
        timings['feature_load_and_construction_seconds'] = time.monotonic() - start
        timings['feature_adapter_detail'] = bundle['timings']
        write(directory / 'feature-identity.json', feature_identity(bundle))
        runtime.atomic_npz(directory / 'feature-arrays.npz', shared=bundle['shared'], geometry=bundle['geometry'])
        contexts = {task['case']['id']: dict(plan=plan, bundle=bundle) for task in tasks(plan)}
        start = time.monotonic()
        records = runtime.execute_tasks(tasks(plan), contexts, directory, None, workers=4,
            deadline=deadline, per_task=plan['per_task_ceiling_seconds'], worker_fn=fold_worker)
        timings['bounded_parallel_fit_prediction_selection_seconds'] = time.monotonic() - start
        check_records(plan, records, directory)
        start = time.monotonic()
        evaluation = evaluate(plan, bundle, records, directory, persist=True)
        write(directory / 'summary.json', summarize(plan, records, evaluation))
        timings['label_reveal_evaluation_summary_seconds'] = time.monotonic() - start
        start = time.monotonic()
        write(directory / 'model-replay.json', replay_models(plan, bundle, records, directory))
        if evaluate(plan, bundle, records, directory, persist=False) != evaluation:
            raise ValueError('Saved decision/label replay differs')
        timings['saved_model_and_score_replay_seconds'] = time.monotonic() - start
        if runtime.git('rev-parse', 'HEAD') != freeze['source_commit'] or runtime.git('status', '--porcelain', '--untracked-files=all'):
            raise ValueError('Source changed during the guarded attempt')
    except BaseException as exc:
        error = repr(exc)
        write(directory / 'failure.json', dict(error=error, ended_at=runtime.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if (directory / 'journal.jsonl').exists():
            records = [json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()]
        observed = {r['id'] for r in records}
        for task in tasks(plan):
            if task['id'] not in observed:
                records.append(dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                                    status='not_started_infrastructure_failure'))
        write(directory / 'terminal-records.json', records)
        timings['producer_and_builtin_replay_seconds_before_receipt_seal'] = time.monotonic() - begin
        write(directory / 'timings.json', timings)
        seal_start = time.monotonic()
        inventory = [runtime.binding(path, directory) for path in sorted(directory.rglob('*'))
                     if path.is_file() and path.name != 'receipt.json']
        write(directory / 'receipt.json', dict(**started, ended_at=runtime.now(),
            wall_seconds=time.monotonic() - begin, receipt_inventory_seconds=time.monotonic() - seal_start,
            observed_records=len(records),
            status='complete' if error is None and all(r['status'] == 'complete' for r in records) else 'incomplete',
            status_counts=dict(Counter(r['status'] for r in records)), error=error, artifacts=inventory))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'run', 'verify'])
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--data-root', type=Path, default=runtime.DEFAULT_DATA_ROOT)
    parser.add_argument('--attempt', default='attempt-001')
    args = parser.parse_args()
    if args.action == 'prepare':
        plan = build_plan()
        write(args.plan, plan)
        print(json.dumps(dict(status='prepared_without_fitting', **validate(plan))))
    elif args.action == 'validate':
        print(json.dumps(dict(status='validated_without_fitting', **validate(read(args.plan), check_bindings=True, reconstruct=True))))
    elif args.action == 'verify':
        # Bound the standalone same-platform replay too. It creates no workers
        # or output files; interrupted or timed-out verification remains failed.
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Verification ceiling')))
        signal.setitimer(signal.ITIMER_REAL, CONTRACT['wall_clock_ceiling_seconds'])
        print(json.dumps(verify(external_attempt(args.data_root, args.attempt))))
    else:
        directory = run(args.plan, args.data_root, args.attempt)
        complete = read(directory / 'receipt.json')['status'] == 'complete'
        print(json.dumps(dict(directory=str(directory), complete=complete)))
        if not complete:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
