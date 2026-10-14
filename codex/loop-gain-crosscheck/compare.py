"""Task18 sealed-output comparison. No model or producer-module execution.

The separately reviewed manifest supplies actual hashes and original closure
authority after the own worker has sealed. No scientific input is supplied by
this source preparation. Only saved arrays/scalars are compared; there is no
eigenvalue, singular-value, derivative, orbit, or dense-product recomputation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import sys
import time
import traceback

SCHEMA = 'codex-task18-sealed-comparison-v2'
ALIGNMENT_TOLERANCE = 1e-8
OUTPUT_BYTES = 16 * 1024**2
CLOSING_BYTES = 512 * 1024
INPUT_BYTES = 1024**3
TIMEOUT = 120
DIMENSION, PERIOD, VOCABULARY = 896, 8, 151936
ANALYSIS_SHA = 'a5553396985f947bb0d8e628f6f1190a683141b341b46538fdc06ceaeec9faf2'
MODEL_SHA = '493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'
PRODUCER_REFERENCE_SHA = '327539c1830d3275f097c81b40c8c762b99a4c01b9626a646397bdda2204f8f8'
RECOVERY_SOURCE_SHA = '49006cea46e3a4c5891fdb5d55a5146988b05a0ad29adb6fd67e5b70ac3a7648'
FAILED_ROOT = '/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002'
FAILED_SOURCE = '7918af949fa60cb8bb123504e78c2dc959943d9c'
FAILED_PLAN = '98a01706d756e3271f3f566dedf792719bf088055ba084c753b199b649b198c4'
FAILED_RECEIPT = '4a3153f174bec42f034e245b63d5fe48978962908cbf82478ace3498b632b4fa'
FAILED_CLOSURE = '2d3a679a6dabde4379f95c56c66b303d9a82a5e1ddad6b76f4a3d7296c66b047'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key: ' + key)
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + s)))


def same(value, wanted, name):
    if type(value) is not type(wanted) or value != wanted:
        raise ValueError('literal mismatch: ' + name)
    if isinstance(value, dict):
        for key in value: same(value[key], wanted[key], name+'/'+key)
    if isinstance(value, list):
        for index in range(len(value)): same(value[index], wanted[index], name+'/'+str(index))


def pointer(value, text):
    if text == '':
        return value
    if not text.startswith('/'):
        raise ValueError('JSON pointer required')
    for part in text[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def array(value, shape, dtype, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
        raise ValueError('wrong shape/dtype/nonfinite saved array: ' + name)
    return value


def metric(left, right, columns=False):
    """All entries contribute; only small error summaries are newly retained."""
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('equal finite comparison shapes required')
    delta = left - right
    flat = int(np.argmax(np.abs(delta)))
    index = tuple(int(i) for i in np.unravel_index(flat, delta.shape))
    error, scale = float(np.linalg.norm(delta)), float(np.linalg.norm(right))
    out = dict(shape=list(left.shape), entries=int(left.size),
               unequal_entries=int(np.count_nonzero(delta)),
               maximum_absolute=float(np.max(np.abs(delta))),
               maximum_index=list(index), signed_difference_at_maximum=float(delta[index]),
               own_at_maximum=float(left[index]), producer_at_maximum=float(right[index]),
               euclidean_or_frobenius_error=error, reference_norm=scale,
               relative_error=error/scale if scale > 0 else None,
               zero_reference=(scale == 0), exact_equal=bool(np.array_equal(left, right)))
    if columns:
        errors = np.linalg.norm(delta, axis=0)
        scales = np.linalg.norm(right, axis=0)
        out['columns'] = [dict(index=i, absolute=float(e), reference_norm=float(s),
                               relative=float(e/s) if s > 0 else None)
                          for i, (e, s) in enumerate(zip(errors, scales))]
    return out


def leaf_differences(own, producer, path=''):
    if isinstance(own, dict) and isinstance(producer, dict):
        return [r for key in sorted(set(own) | set(producer))
                for r in leaf_differences(own.get(key), producer.get(key), path+'/'+key)]
    if isinstance(own, list) and isinstance(producer, list):
        return [r for i in range(max(len(own), len(producer)))
                for r in leaf_differences(own[i] if i < len(own) else None,
                                          producer[i] if i < len(producer) else None, path+'/'+str(i))]
    numbers = all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in (own, producer))
    if numbers and not all(math.isfinite(v) for v in (own, producer)):
        raise ValueError('nonfinite scalar comparison')
    return [dict(path=path, own=own, producer=producer,
                 equal=((numbers or type(own) is type(producer)) and own == producer),
                 difference=own-producer if numbers else None,
                 relative_difference=(own-producer)/abs(producer) if numbers and producer != 0 else None)]


def alignment(own, producer):
    """Every forward rotation; no reversal/transpose choice or tuned threshold."""
    own_scale = float(np.linalg.norm(own[:8], axis=1).mean())
    producer_scale = float(np.linalg.norm(producer[:8], axis=1).mean())
    if min(own_scale, producer_scale) <= 0:
        raise ValueError('positive orbit scales required')
    scale = producer_scale
    distances = np.linalg.norm(own[:8, None, :] - producer[None, :8, :], axis=2)
    scores = []
    for shift in range(8):
        values = [float(distances[i, (i+shift) % 8]/scale) for i in range(8)]
        scores.append(dict(shift=shift, convention='own_i matches producer_(i+shift)%8',
                           producer_phases=[(i+shift) % 8 for i in range(8)],
                           phase_relative_errors=values, maximum_relative_error=max(values),
                           rms_relative_error=float(np.sqrt(np.mean(np.square(values)))),
                           qualifies=max(values) < ALIGNMENT_TOLERANCE))
    qualified = [s['shift'] for s in scores if s['qualifies']]
    selected = qualified[0] if len(qualified) == 1 else None
    return dict(scale='producer mean norm over its eight nonduplicate states',
                producer_mean_norm=producer_scale, own_mean_norm=own_scale,
                full_state_absolute_distance_grid=distances.tolist(), rotations=scores,
                unique_qualified_rotation=selected, qualifying_rotations=qualified,
                state_alignment_threshold=ALIGNMENT_TOLERANCE,
                disposition='aligned' if selected is not None else 'unaligned_or_ambiguous',
                own_closure_relative=float(np.linalg.norm(own[8]-own[0])/own_scale),
                producer_closure_relative=float(np.linalg.norm(producer[8]-producer[0])/producer_scale))


class Inputs:
    def __init__(self, manifest, check):
        self.m = manifest; self.check = check; self.loaded = {}; self.used = {}; self.bytes = 0
        if manifest['schema'] != SCHEMA or manifest['admission'] != 'reviewed':
            raise ValueError('reviewed exact input manifest required')
        if manifest['alignment_tolerance'] != ALIGNMENT_TOLERANCE:
            raise ValueError('prospective alignment rule changed')
        if manifest['source_sha256'] != sha(Path(__file__)):
            raise ValueError('comparison source differs from reviewed input manifest')
        if not isinstance(manifest['authority_scope'], str) or not manifest['authority_scope'].strip():
            raise ValueError('explicit finite authority scope required')
        for file_id, record in manifest['files'].items():
            if type(file_id) is not str or not file_id:
                raise ValueError('named literal files required')
            self.verify(record)
        if manifest['files'][manifest['producer_reference_source']]['sha256'] != PRODUCER_REFERENCE_SHA:
            raise ValueError('reference producer source differs; not an executed-source attestation')

    def verify(self, record):
        if set(record) != {'path', 'bytes', 'sha256'}:
            raise ValueError('literal path/byte/hash record required')
        path = Path(record['path'])
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise ValueError('canonical nonsymlink input required')
        if type(record['bytes']) is not int or not 0 <= record['bytes'] <= INPUT_BYTES:
            raise ValueError('bounded input bytes required')
        if re.fullmatch('[0-9a-f]{64}', record['sha256']) is None:
            raise ValueError('full input hash required')
        if str(path) not in self.used:
            self.bytes += record['bytes']
            if self.bytes > INPUT_BYTES:
                raise ValueError('selected input IO exceeds 1GiB')
        elif self.used[str(path)] != record:
            raise ValueError('conflicting input pins')
        self.check()
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('input bytes/hash changed: ' + str(path))
        self.used[str(path)] = dict(record)
        return path

    def file(self, name):
        return Path(self.m['files'][name]['path'])

    def json(self, name):
        if name not in self.loaded:
            self.loaded[name] = read_json(self.file(name))
        return self.loaded[name]

    def assertions(self):
        rows = self.m['assertions']
        if not rows:
            raise ValueError('actual original closure/source assertions required')
        result = []
        for row in rows:
            if set(row) != {'file', 'pointer', 'equals'}:
                raise ValueError('literal original authority assertions required')
            actual = pointer(self.json(row['file']), row['pointer'])
            same(actual, row['equals'], row['file']+row['pointer'])
            result.append(dict(**row, matched=True))
        return result

    def finish(self):
        for record in list(self.used.values()):
            self.verify(record)


def recovered_summary(inputs, receipt, root, inventory):
    """Admit a separate saved summary without converting original exit1 to exit0.

    Actual recovery and closure pins come from the reviewed input manifest.
    No file under the original attempt is rewritten, and its partial status is
    hashed as evidence rather than decoded as a successful worker status.
    """
    spec = inputs.m['own']['recovery']
    same(set(spec), {'report', 'manifest', 'source', 'card', 'original_closure',
                     'closure', 'closure_pointers'}, 'recovery interface')
    files = inputs.m['files']
    same(str(root), FAILED_ROOT, 'specific original failed attempt')
    same(files[inputs.m['own']['receipt']]['sha256'], FAILED_RECEIPT, 'failed receipt pin')
    for key, value in dict(source_commit=FAILED_SOURCE, plan_sha256=FAILED_PLAN).items():
        same(receipt[key], value, 'failed receipt/' + key)
    same(files[spec['original_closure']]['sha256'], FAILED_CLOSURE, 'original closure pin')
    original_closure = inputs.json(spec['original_closure'])
    for key, value in dict(passed=True, task=18, root=FAILED_ROOT,
                          tool_session_reaped_exit_code=1, receipt_sha256=FAILED_RECEIPT,
                          source_commit=FAILED_SOURCE, plan_sha256=FAILED_PLAN,
                          original_waiter_launcher_worker_pids_groups_sessions_absent=True,
                          uncertain_ownership=[]).items():
        same(original_closure[key], value, 'original closure/' + key)

    same(files[spec['source']]['sha256'], RECOVERY_SOURCE_SHA, 'reviewed recovery source pin')
    card = inputs.json(spec['card'])
    for key, value in dict(admission='reviewed', schema='task18-saved-summary-recovery-v1',
                          script_sha256=RECOVERY_SOURCE_SHA, source_commit=FAILED_SOURCE,
                          plan_sha256=FAILED_PLAN, receipt_sha256=FAILED_RECEIPT,
                          closure_sha256=FAILED_CLOSURE).items():
        same(card[key], value, 'recovery card/' + key)
    same(card['caps'], dict(cpu_threads=1, nice=10, memory_gib=1, timeout_seconds=120,
        maximum_output_mib=8, minimum_start_available_gib=20, minimum_runtime_available_gib=16),
        'reviewed recovery caps')
    report = inputs.json(spec['report'])
    for key, value in dict(status='saved_summary_recovered', recovery_completed=True,
                          original_exit_success=False, scientific_event_set_complete=True,
                          certificate_passed=False, model_rerun=False, matrix_eigen_svd_rerun=False,
                          recovery_source_sha256=RECOVERY_SOURCE_SHA,
                          recovery_card_sha256=files[spec['card']]['sha256']).items():
        same(report[key], value, 'separate recovered summary/' + key)
    if type(report['saved_numerical_gates_pass']) is not bool:
        raise ValueError('saved numerical disposition must remain a Boolean, including false')
    if not report['checks'] or any(row['passed'] is not True for row in report['checks']):
        raise ValueError('completed recovery must retain all passing evidence checks')
    if not isinstance(report['limitations'], list) or not report['limitations'] or any(
            not isinstance(text, str) or not text.strip() for text in report['limitations']):
        raise ValueError('original failure and recovery limitations must remain explicit')
    authority = report['original_authority']
    for key, value in dict(source_commit=FAILED_SOURCE, plan_sha256=FAILED_PLAN,
                          receipt_sha256=FAILED_RECEIPT, closure_sha256=FAILED_CLOSURE,
                          closure_path=str(inputs.file(spec['original_closure'])),
                          original_exit_code=1, original_failure_preserved=True,
                          original_elapsed_seconds=receipt['elapsed_seconds'],
                          original_peak_rss_kib=None, original_backend_identity=None,
                          partial_status_sha256=inventory['run/status.json']['sha256'],
                          stderr_sha256=inventory['stderr.log']['sha256']).items():
        same(authority[key], value, 'recovery original authority/' + key)
    # Authenticate the original failed status and stderr, without parsing or rewriting them.
    for relative in ('run/status.json', 'stderr.log'):
        row = inventory[relative]
        path = root/relative
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError('original reporting-failure evidence not sealed')
        inputs.verify(dict(path=str(path), bytes=row['bytes'], sha256=row['sha256']))

    # The recovery manifest is immutable authority for its report and retained
    # copies of the original partial status/stderr. No guessed success receipt.
    manifest = inputs.json(spec['manifest'])
    recovery_root = inputs.file(spec['manifest']).parent
    same(recovery_root, inputs.file(spec['report']).parent, 'recovery report/manifest directory')
    records = {}
    for row in manifest['files']:
        if set(row) != {'path', 'bytes', 'sha256'} or Path(row['path']).name != row['path']:
            raise ValueError('literal recovery artifact basename required')
        if row['path'] in records:
            raise ValueError('duplicate recovery manifest artifact')
        records[row['path']] = row
        path = recovery_root/row['path']
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError('recovery output not sealed read-only')
        inputs.verify(dict(path=str(path), bytes=row['bytes'], sha256=row['sha256']))
    same(set(records), {'report.json', 'original-partial-status.json.bytes', 'original-stderr.log'},
         'exact separate recovery output inventory')
    for name, wanted in [('report.json', files[spec['report']]['sha256']),
                         ('original-partial-status.json.bytes', authority['partial_status_sha256']),
                         ('original-stderr.log', authority['stderr_sha256'])]:
        same(records[name]['sha256'], wanted, 'recovery member/' + name)

    # Pointers are selected from the actual coordinator closure report and
    # reviewed before launch. Their roles and expected values are fixed here.
    closure = inputs.json(spec['closure'])
    expected = dict(passed=True, exit_code=0, all_processes_absent=True,
                    report_sha256=files[spec['report']]['sha256'])
    same(set(spec['closure_pointers']), set(expected), 'recovery closure pointer roles')
    closure_checks = []
    for role, wanted in expected.items():
        location = spec['closure_pointers'][role]
        if not isinstance(location, str) or not location.startswith('/'):
            raise ValueError('actual recovery closure JSON pointer required')
        same(pointer(closure, location), wanted, 'recovery closure/' + role)
        closure_checks.append(dict(role=role, file=spec['closure'], pointer=location,
                                   equals=wanted, matched=True))
    disposition = dict(mode='recovered_reporting_failure',
        original_status=receipt['status'], original_exit_code=receipt['exit_code'],
        original_numerical_gain_result_available=receipt['numerical_gain_result_available'],
        original_exit_success=False, original_receipt_sha256=FAILED_RECEIPT,
        recovery_report_sha256=files[spec['report']]['sha256'],
        recovery_manifest_sha256=files[spec['manifest']]['sha256'],
        recovery_source_sha256=RECOVERY_SOURCE_SHA,
        recovery_card_sha256=files[spec['card']]['sha256'],
        recovery_closure_sha256=files[spec['closure']]['sha256'],
        recovery_closure_checks=closure_checks, original_authority=authority,
        saved_numerical_gates_pass=report['saved_numerical_gates_pass'],
        recovery_reconciliation=report['reconciliation'], limitations=report['limitations'],
        recovery_seconds=report['recovery_seconds'], recovery_peak_rss_kib=report['recovery_peak_rss_kib'],
        recovery_backend=report['recovery_backend'])
    return report, disposition


class Own:
    def __init__(self, inputs):
        self.inputs = inputs; m = inputs.m; self.root = Path(m['own']['root'])
        if not self.root.is_absolute() or self.root.resolve() != self.root:
            raise ValueError('literal own attempt root required')
        self.receipt = receipt = inputs.json(m['own']['receipt'])
        same(inputs.file(m['own']['receipt']), self.root/'receipt.json', 'original own receipt path')
        mode = m['own']['summary_mode']
        if mode not in ('original_completed', 'recovered_reporting_failure'):
            raise ValueError('explicit own summary disposition required')
        recovered = mode == 'recovered_reporting_failure'
        for k, v in dict(status='loop_gain_failed' if recovered else 'loop_gain_completed',
                         exit_code=1 if recovered else 0, timed_out=False,
                         worker_termination_verified=True,
                         numerical_gain_result_available=not recovered).items():
            same(receipt[k], v, 'own receipt/' + k)
        same(receipt['uncertain_ownership'], [], 'own receipt/uncertain_ownership')
        self.inventory = {}
        for row in receipt['artifacts']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts or str(relative) != row['path']:
                raise ValueError('invalid original artifact path')
            if row['path'] in self.inventory:
                raise ValueError('duplicate original own artifact')
            self.inventory[row['path']] = row
        if recovered:
            self.status, self.disposition = recovered_summary(inputs, receipt, self.root, self.inventory)
        else:
            if 'recovery' in m['own']:
                raise ValueError('successful original summary must not carry recovery authority')
            self.status = self.json('run/status.json')
            for k, v in dict(status='loop_gain_completed', computational_completed=True,
                             producer_arrays_read=False, certificate_passed=False, exploratory=True).items():
                same(self.status[k], v, 'own worker/' + k)
            for key in ('source_commit', 'plan_sha256'):
                same(self.status[key], receipt[key], key)
            self.disposition = dict(mode=mode, original_status=receipt['status'],
                original_exit_code=0, original_exit_success=True,
                original_numerical_gain_result_available=True,
                original_receipt_sha256=m['files'][m['own']['receipt']]['sha256'])
        self.binding = self.json('run/input-binding.json')
        if recovered:
            same(self.status['original_input_binding'], self.binding, 'recovered original binding')
        same(self.binding['producer_arrays_read'], False, 'independent input binding')
        same(self.binding['steering_layer'], 12, 'own steering layer')
        same(self.binding['vector_key'], 'philosophy', 'own steering vector key')
        same(self.binding['dtype'], 'float64', 'own binary64 source binding')
        same(self.binding['tied_readout'], True, 'own tied readout source binding')
        if len(self.binding['prefix_ids']) != 37 or any(type(k) is not int for k in self.binding['prefix_ids']):
            raise ValueError('own full prefix IDs required')
        same(self.binding['source_commit'], receipt['source_commit'], 'binding source')
        same(self.binding['plan_sha256'], receipt['plan_sha256'], 'binding plan')
        same(self.binding['source_sha256']['codex/loop-gain-crosscheck/analysis.py'], ANALYSIS_SHA, 'analysis pin')
        same(self.binding['source_sha256']['codex/crosscheck/qwen_map.py'], MODEL_SHA, 'model pin')
        self.plan = self.json('run/original-plan.json')
        if sha(self.root/'run/original-plan.json') != receipt['plan_sha256']:
            raise ValueError('own actual plan hash differs')
        same(self.plan['parameters'], dict(alpha=.071674, log10_temperature=-.115, period=8,
            transient=1000, dimension=896, vocabulary=151936, batch_directions=16,
            random_directions=8, first_seed=18001), 'own actual recipe')
        self.states = self.saved(self.event('orbit')['states'], (9,896), 'float64')
        if len(self.status['phases']) != 8 or [p['index'] for p in self.status['phases']] != list(range(8)):
            raise ValueError('all ordered own phase summaries required')
        if len(self.status['cyclic_monodromies']) != 8 or [p['start_phase'] for p in self.status['cyclic_monodromies']] != list(range(8)):
            raise ValueError('all ordered own cyclic summaries required')
        self.matrices = []; self.spectra = []; self.readouts = []; self.loops = []
        for phase in range(8):
            readout = self.event(f'phase{phase}-readout')
            state = self.saved(readout['state'], (896,), 'float64')
            if not np.array_equal(state, self.states[phase]):
                raise ValueError('own phase readout state differs from complete orbit')
            self.readouts.append(readout)
            dense = self.event(f'phase{phase}-dense')
            same(dense['columns'], 896, 'own full dense columns')
            self.matrices.append(self.saved(dense['map_jacobian'], (896,896), 'float64'))
            spectrum = self.event(f'phase{phase}-spectra')
            self.spectra.append(dict(
                singular=self.saved(spectrum['map_singular_values'], (896,), 'float64'),
                covariance=self.saved(spectrum['covariance_eigenvalues'], (896,), 'float64')))
            loop = self.event(f'monodromy-phase{phase}')
            same(loop['start_phase'], phase, 'cyclic phase')
            # NumPy eigvals of a real matrix may return real dtype when all roots are real.
            eigen_dtype = loop['monodromy_eigenvalues']['dtype']
            if eigen_dtype not in ('float64','complex128'):
                raise ValueError('saved binary64 real/complex eigenvalues required')
            self.loops.append(dict(
                eigen=self.saved(loop['monodromy_eigenvalues'], (896,), eigen_dtype),
                singular=self.saved(loop['monodromy_singular_values'], (896,), 'float64'),
                product_step_norms=loop['product_step_norms']))

    def path(self, relative):
        row = self.inventory[relative]
        path = self.root/relative
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError('own output not sealed read-only: ' + str(path))
        self.inputs.verify(dict(path=str(path), bytes=row['bytes'], sha256=row['sha256']))
        return path

    def json(self, relative):
        return read_json(self.path(relative))

    def event(self, name):
        candidates = [p for p in self.inventory if re.fullmatch(r'run/\d{4}-'+re.escape(name)+r'\.json', p)]
        if len(candidates) != 1:
            raise ValueError('one original own event required: ' + name)
        return self.json(candidates[0])

    def saved(self, descriptor, shape, dtype):
        if set(descriptor) != {'path', 'dtype', 'shape', 'sha256'}:
            raise ValueError('native descriptor required')
        if Path(descriptor['path']).name != descriptor['path']:
            raise ValueError('native basename only')
        relative = 'run/'+descriptor['path']
        if self.inventory[relative]['sha256'] != descriptor['sha256']:
            raise ValueError('native descriptor is not original receipt artifact')
        same(descriptor['shape'], list(shape), 'native shape descriptor')
        same(descriptor['dtype'], dtype, 'native dtype descriptor')
        return array(np.load(self.path(relative), allow_pickle=False), shape, dtype, relative)


def producer_recipe(summary, mode):
    wanted = dict(mode=mode, model='qwen0.5b', layer=12, vectors='llm-phase-map/vectors-l12',
                  concept='philosophy', alpha=.071674, log_temperature=-.115,
                  transient=1000, dtype='float64')
    if mode == 'orbit': wanted['period'] = 8
    for key, value in wanted.items(): same(summary['args'][key], value, 'producer args/' + key)
    temperature = summary['temperature']
    if not isinstance(temperature, (int,float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite positive producer temperature required')
    same(temperature, 10.**(-.115), 'producer physical temperature from pinned log-temperature')


def producer_metadata(producer, own, inputs):
    """Only explicitly supplied original metadata can attest historical identity."""
    fields = ('dimension', 'vocabulary', 'prefix_ids', 'steering_layer', 'vector_key', 'asset_sha256')
    records = producer['metadata']
    if set(records) != set(fields):
        raise ValueError('explicit original metadata availability for every field required')
    expected = dict(dimension=896, vocabulary=151936,
                    **{k: own.binding[k] for k in fields if k not in ('dimension','vocabulary')})
    results = []
    for field in fields:
        record = records[field]
        if set(record) == {'unavailable_reason'} and isinstance(record['unavailable_reason'], str) and record['unavailable_reason'].strip():
            results.append(dict(field=field, disposition='not_recorded_or_not_independently_admitted',
                                own=expected[field], reason=record['unavailable_reason']))
        elif set(record) == {'file','pointer'}:
            actual = pointer(inputs.json(record['file']), record['pointer'])
            results.append(dict(field=field, disposition='compared_original_metadata',
                                evidence=record, differences=leaf_differences(expected[field],actual)))
        else:
            raise ValueError('original metadata pointer or explicit limitation required: '+field)
    return results


def compare_run(own, producer, inputs):
    summary = inputs.json(producer['summary'])
    producer_recipe(summary, 'orbit')
    if len(summary['steps']) != 8:
        raise ValueError('all eight producer step summaries required')
    result = dict(name=producer['name'], original_arguments=summary['args'],
                  historical_authority=producer['authority'],
                  metadata=producer_metadata(producer,own,inputs),
                  original_temperature=summary['temperature'], aligned_comparisons=[],
                  original_reported_closure=summary.get('closure_relative'))
    if producer['arrays'] is None:
        if not producer['arrays_unavailable_reason']:
            raise ValueError('absent original arrays require explicit disclosed reason')
        result.update(disposition='unaligned_original_arrays_unavailable',
                      unavailable_reason=producer['arrays_unavailable_reason'])
        return result
    with np.load(inputs.file(producer['arrays']), allow_pickle=False) as archive:
        if set(archive.files) != {'states','jacobians'}:
            raise ValueError('literal producer orbit archive schema differs')
        states = array(archive['states'], (9,896), 'float64', 'producer states')
        matrices = array(archive['jacobians'], (8,896,896), 'float64', 'producer Jacobians')
    align = alignment(own.states, states); result['alignment'] = align
    result['original_closure_check'] = dict(
        convention='producer denominator is norm of its phase0; own original gate uses mean phase norm',
        own_using_producer_convention=float(np.linalg.norm(own.states[8]-own.states[0])/np.linalg.norm(own.states[0])),
        producer_recomputed_from_saved_states=float(np.linalg.norm(states[8]-states[0])/np.linalg.norm(states[0])),
        original_reported=summary['closure_relative'])
    result['original_closure_check']['saved_minus_original'] = (
        result['original_closure_check']['producer_recomputed_from_saved_states']-summary['closure_relative'])
    # Every possible phase-pair matrix error is retained; no error threshold selects alignment.
    result['step_matrix_pair_grid'] = [dict(own_phase=i, producer_phase=j,
        **metric(own.matrices[i], matrices[j])) for i in range(8) for j in range(8)]
    shift = align['unique_qualified_rotation']
    if shift is None:
        result['disposition'] = 'unaligned_or_ambiguous'
        return result
    result['disposition'] = 'aligned_saved_comparison'
    for i in range(8):
        j = (i+shift) % 8; readout = own.readouts[i]
        probabilities = own.saved(readout['probabilities'], (151936,), 'float64')
        if np.any(probabilities < 0) or abs(float(probabilities.sum())-1) > 1e-12:
            raise ValueError('saved full readout is not normalized')
        saved_temperature = own.saved(readout['temperature'], (), 'float64')
        actual = dict(entropy_nats=float(-np.sum(probabilities*np.log(np.maximum(probabilities,1e-300)))),
            top3_p=np.sort(probabilities)[-3:][::-1].tolist(),
            sigma_p_eigenvalues=own.spectra[i]['covariance'][:4].tolist(),
            df_singular_values=own.spectra[i]['singular'][:4].tolist())
        old = summary['steps'][j]
        required = set(actual)
        if not required <= set(old):
            raise ValueError('producer step summary missing defined scientific field')
        result['aligned_comparisons'].append(dict(own_phase=i,producer_phase=j,
            state=metric(own.states[i],states[j]),
            step_matrix=metric(own.matrices[i],matrices[j],columns=True),
            scalar_differences=leaf_differences(actual,{k:old[k] for k in actual}),
            temperature_difference=float(saved_temperature)-summary['temperature'],
            own_leading_token_id=int(np.argmax(probabilities)), producer_decoded_token=old.get('token'),
            token_text_comparison='not independently decoded; no tokenizer/model loaded'))
    # Producer monodromy starts at producer phase0, which corresponds to own (-shift)%8.
    start = (-shift) % 8; loop = own.loops[start]
    singular = np.stack([s['singular'][:2] for s in own.spectra])
    actual = dict(multipliers=[[float(z.real),float(z.imag)] for z in loop['eigen'][:4]],
        multiplier_moduli=np.abs(loop['eigen'][:4]).tolist(),
        monodromy_singular_values=loop['singular'][:4].tolist(),
        prod_sigma1=loop['product_step_norms'],
        prod_sigma1_sigma2=float(np.prod(singular[:,0]*singular[:,1])),
        prod_sigma2_over_sigma1=float(np.prod(singular[:,1]/singular[:,0])))
    result['loop'] = dict(own_start_phase=start,producer_start_phase=0,
        differences=leaf_differences(actual,{k:summary.get(k) for k in actual}),
        fields_unavailable_in_original=[k for k in actual if k not in summary],
        eigenvalue_comparison='saved order by descending modulus, signed real/imag retained; no eigensolver replay',
        own_all_cyclic_summaries=own.status['cyclic_monodromies'])
    producer_eigen = np.asarray(summary['multipliers'],dtype=np.float64)
    if producer_eigen.shape != (4,2) or not np.isfinite(producer_eigen).all():
        raise ValueError('four original signed complex multipliers required')
    old_complex = producer_eigen[:,0]+1j*producer_eigen[:,1]
    result['loop']['leading_eigenvalue_pair_grid'] = [dict(own_index=i,producer_index=j,
        signed_real_difference=float((a-b).real), signed_imaginary_difference=float((a-b).imag),
        absolute_difference=float(abs(a-b))) for i,a in enumerate(loop['eigen'][:4]) for j,b in enumerate(old_complex)]
    result['loop']['pair_grid_disposition'] = 'all saved leading pairs retained; no permutation or conjugation selected'
    return result


def execute(manifest, check):
    inputs = Inputs(manifest, check); assertions = inputs.assertions(); own = Own(inputs)
    producers = manifest['producers']
    if [p['name'] for p in producers] != ['p8','p8b']:
        raise ValueError('both original producer runs must have explicit disposition')
    results = [compare_run(own,p,inputs) for p in producers]
    identity = manifest['producer_identity']
    identity_result = None
    if identity is not None:
        original = inputs.json(identity); producer_recipe(original,'identity')
        directions = original['relative_errors']
        if len(directions) != 8 or not all(math.isfinite(v) and v >= 0 for v in directions):
            raise ValueError('eight finite producer identity errors required')
        if original['max_relative_error'] != max(directions):
            raise ValueError('producer maximum does not match retained direction errors')
        identity_result = dict(original=original,
            independent_errors_by_phase=[dict(index=p['index'],checks=p['checks'],directions=p['directions'])
                                          for p in own.status['phases']],
            paired_random_direction_comparison=False,
            reason='different predeclared random seeds/directions; individual errors retained, not paired')
    else:
        if not manifest['producer_identity_unavailable_reason'].strip():
            raise ValueError('absent original identity receipt requires explicit reason')
        identity_result = dict(disposition='original_identity_receipt_unavailable',
                               reason=manifest['producer_identity_unavailable_reason'])
    inputs.finish()
    return dict(schema='codex-task18-saved-comparison-result-v2',
        computational_completed=True,certificate_passed=False,model_executed=False,
        eigensolver_or_svd_executed=False,producer_module_imported=False,
        producer_read_after_sealed_own_evidence_admission=True,source_sha256=sha(Path(__file__)),
        authority_scope=manifest['authority_scope'],original_assertions=assertions,
        own_source_commit=own.receipt['source_commit'], own_result_disposition=own.disposition,
        own_input_binding=own.binding,
        own_orbit_gates=own.status['orbit_gates'],own_phases=own.status['phases'],
        own_composed_controls=own.status['composed_controls'],producer_runs=results,
        factor_identity_receipt=identity_result,
        compared_original_files=list(inputs.used.values()),selected_input_bytes=inputs.bytes,
        all_computed_scalar_differences_retained=True,all_matrix_entries_contribute_to_error_summaries=True,
        original_full_arrays_remain_pinned=True,scientific_acceptance_pending=True,
        scope='finite sealed-output numerical comparison; no independent model, orbit, or dense spectral replay')


def main():
    began=time.monotonic(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.resolve()!=args.output: raise ValueError('canonical exclusive output path required')
    args.output.mkdir(exist_ok=False)
    def write(name, value, closing=False):
        data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        bound = OUTPUT_BYTES if closing else OUTPUT_BYTES-CLOSING_BYTES
        if sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())+len(data)>bound:
            raise RuntimeError('16MiB output ceiling')
        with (args.output/name).open('xb') as stream:stream.write(data)
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('external nice10 required')
        limit=4*1024**3; hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<limit:raise ValueError('inherited AS below approved4GiB')
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        resource.setrlimit(resource.RLIMIT_CPU,(120,125))
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='1'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        if re.fullmatch('[0-9a-f]{64}',args.manifest_sha256)is None or sha(args.manifest)!=args.manifest_sha256:
            raise ValueError('literal external input-manifest hash required')
        if args.manifest.stat().st_size>2*1024**2:raise ValueError('bounded manifest required')
        write('input-manifest.json',read_json(args.manifest))
        global np
        import numpy as np
        def check():
            if time.monotonic()-began>=TIMEOUT:raise TimeoutError('120s cooperative cap; outer timeout still required')
        result=execute(read_json(args.manifest),check);check()
        result.update(seconds=time.monotonic()-began,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      input_manifest_sha256=args.manifest_sha256,backend=dict(python=sys.version,numpy=np.__version__))
        write('report.json',result)
        if sha(args.manifest)!=args.manifest_sha256:raise ValueError('input manifest changed during comparison')
        write('status.json',dict(status='completed',exit_code=0,seconds=time.monotonic()-began))
        return 0
    except BaseException as error:
        write('failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
                                 seconds=time.monotonic()-began,certificate_passed=False),closing=True)
        return 1
    finally:
        write('output-manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(args.output.iterdir()) if p.is_file()]),closing=True)


if __name__=='__main__':sys.exit(main())
