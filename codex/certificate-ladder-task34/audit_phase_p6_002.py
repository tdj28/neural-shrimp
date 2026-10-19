"""Explicit actual002 Task34 cascade-P6 saved phase audit; no model or producer replay.

The saved ordered-product, residual, inverse-defect and diagnostic formulas are
those in accepted audit_phase.py. This wrapper binds the original named P6 run
and refuses missing phases/files/arrays. Import performs no evidence access.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

R = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D = R / 'codex/certificate-ladder-task34'
B = Path('/data2/neural-shrimp-data/codex/certificate-ladder')
SOURCE = '9028465dfb8e7a9001b34a091cc61067b464d80e'
ATTEMPT, CELL, PERIOD, DIMENSION = 'phase-cascade-p6-002', 'cascade-p6', 6, 896
RAW = B / ATTEMPT
SUP = B / (ATTEMPT + '-supervision')
CARD = D / 'phase-cascade-p3-001-card.json'  # Original shared five-cell card.
PLAN = D / 'phase-cascade-p6-002-plan.json'
RESULT = B / (ATTEMPT + '-audit-001') / 'result.json'
CARD_SHA = 'fad1beeb1eda8afaed84bcf315d27ffc1f1e446f3c2694b9690c1c702e87aa4a'
PLAN_SHA = '9a2697f832d6615cb5c0ec68870bcc4aef3ff6da4e4ac257be1e5c526ac2ea40'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
SCIENCE = {
    'codex/certificate-ladder-task34/phase_preconditioner.py': '71e3a6046a4f0857df83089a508c1bad49851c00145abb81713bf7227f6c106f',
    'codex/certificate-ladder-task34/derivative_profile.py': '63d17376d9d33a7fda7669dfba03f317013a0aee19564533280edf9c51fbaf6b',
    'codex/certify-llm/center_seed.py': '87cf27a05999641995d441704cd7a1ec3ea5aa0d2c0d200b0c05662e584572e5',
    'codex/crosscheck/qwen_map.py': '493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e',
    'codex/certify-chaos/supervise.py': '306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44',
    'codex/certificate-ladder-task34/phase-cascade-p3-001-card.json': CARD_SHA,
}
ARRAYS = {'A', 'B', 'centers', 'parameters', 'product', 'left_inverse_defect', 'right_inverse_defect'}
PRODUCT_ORDER = 'A[p-1] @ ... @ A[0]'
SCOPE = ('Saved six-phase finite binary64 proposals only. Phase images and derivatives inherit the '
         'authenticated reviewed original execution. No model replay, rigorous derivative enclosure, '
         'rigorous inverse, orbit certificate, trajectory or reachability proof.')


# Exact closed original metadata; Root completion carries historical runtime/asset
# authentication. This auditor rechecks saved data/card inputs, not every runtime
# or model preflight file, and does not attest native execution or prior asset use.
RECOVERY_CARD = D / 'phase-cascade-p6-002-card.json'
PINS = {
    'scientific_card': dict(path=str(CARD), bytes=6064, sha256=CARD_SHA),
    'plan': dict(path=str(PLAN), bytes=33562, sha256=PLAN_SHA),
    'resource_card': dict(path=str(RECOVERY_CARD), bytes=8582, sha256='7a850118506c8ef8bb1ed7b04f0c481a75367039e6d30f1be736d3326b88360d'),
    'receipt': dict(path=str(SUP/'receipt.json'), bytes=8391, sha256='68d5d3d6babee7db31fef948e38491c9e49d988d353668dc673ccd87d85244d1'),
    'admission': dict(path=str(SUP/'admission.json'), bytes=9442, sha256='bcd1b2f0e62b4b0843418b6b5bf39633efab4dc4341f4dece4f939047d82b2e0'),
    'worker_startup': dict(path=str(SUP/'worker-startup.json'), bytes=2644, sha256='e5da98b0f9a8bb440dcd3abc6c2c5f14c88fcf99ff38436153d868f434e0ea49'),
    'root_startup': dict(path=str(D/'phase-cascade-p6-002-startup.json'), bytes=10462, sha256='b26218fc1e7d513708a1cfa4e4005d0c7307414860242e2cb9e08ea15c506262'),
    'completion': dict(path=str(D/'phase-cascade-p6-002-completion.json'), bytes=43552, sha256='19e44e5816068dbab156a227e850f4ab30f859af1317783f2885fc2a48107eb7'),
    'host_closure': dict(path=str(D/'phase-cascade-p6-002-host-closure.json'), bytes=367, sha256='81fe09664d8d9464ffc05b82941278c117e2f9ef50bc8754be2063167f48bea2'),
}
SUPERVISOR = dict(pid=787987, starttime=237723364, pgid=787986, sid=787986)
WORKER = dict(pid=788101, starttime=237723667, pgid=788101, sid=788101)
ORIGINAL_LIMITS = dict(as_bytes=12884901888, cpu_seconds=5080, wall_seconds=5180, output_bytes=134217728)
FALSE_FLAGS = ('certificate_passed', 'phase_qualification_passed', 'derivative_qualification_admitted',
               'inverse_qualification_admitted', 'orbit_qualification_admitted')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    answer = {}
    for key, value in items:
        require(key not in answer, 'duplicate JSON key')
        answer[key] = value
    return answer


def metadata(record):
    """Decode the exact authenticated bytes, bounded to small JSON metadata."""
    require(type(record) is dict and set(record) == {'path', 'bytes', 'sha256'}, 'complete metadata pin required')
    require(type(record['bytes']) is int and 0 < record['bytes'] <= 2 * 1024**2, 'bounded metadata bytes required')
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
            'canonical metadata path required')
    with path.open('rb') as stream:
        data = stream.read(record['bytes'] + 1)
    require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
            'exact metadata bytes differ')
    value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
    def finite(item):
        if type(item) is float: require(math.isfinite(item), 'nonfinite JSON number')
        elif type(item) is dict:
            for child in item.values(): finite(child)
        elif type(item) is list:
            for child in item: finite(child)
    finite(value)
    return value


def pin(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'original regular absolute evidence file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(record):
    require(type(record) is dict and set(record) == {'path', 'bytes', 'sha256'}, 'complete file pin required')
    require(type(record['bytes']) is int and record['bytes'] >= 0
            and type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256']) is not None,
            'strict byte length and digest required')
    require(pin(record['path']) == record, 'original file bytes differ')
    return record


def inventory(records):
    require(type(records) is list, 'complete file inventory required')
    values = {record['path']: record for record in records}
    require(len(values) == len(records), 'duplicate original file paths')
    return values


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'source', 'cell', 'receipt-sha256', 'admission-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    require((args.attempt, args.source, args.cell) == (ATTEMPT, SOURCE, CELL), 'wrong original producer allocation')
    for value in (args.receipt_sha256, args.admission_sha256):
        require(re.fullmatch('[0-9a-f]{64}', value) is not None, 'explicit original receipt/admission SHA256 required')
    require(args.receipt_sha256 == PINS['receipt']['sha256']
            and args.admission_sha256 == PINS['admission']['sha256'], 'wrong exact002 original metadata hashes')
    require(args.output == str(RESULT), 'fresh separate named P6 saved-audit output required')
    return args


def expected_cell(card):
    require(card['repository'] == str(R), 'foreign shared card repository')
    named = [cell for cell in card['cells'] if cell['name'] == CELL]
    require(len(named) == 1, 'missing or duplicate named cascade-P6 cell')
    cell = named[0]
    require(type(cell['period']) is int and cell['period'] == PERIOD
            and cell['alpha'] == .065693 and cell['log10T'] == -.14798, 'wrong named cascade-P6 cell/period')
    require(cell['alpha_ratio'] == [147927485160425, 2251799813685248]
            and cell['log10T_ratio'] == [-333221336429143, 2251799813685248]
            and list(cell['alpha'].as_integer_ratio()) == cell['alpha_ratio']
            and list(cell['log10T'].as_integer_ratio()) == cell['log10T_ratio'], 'wrong exact binary64 cell ratios')
    require(cell['states'] == str(B / 'settle-001/cascade-p6-states.npy')
            and cell['settle_report'] == str(B / 'settle-001/cascade-p6-report.json')
            and cell['seed'] == 'p3' and cell['seed_path'] == '/data2/neural-shrimp-data/explore/sheets/seeds/p3.npy',
            'wrong original cascade-P6 history/seed authority')
    return cell


def expected_outputs():
    return {str(RAW / 'report.json'), str(RAW / CELL / 'preconditioner.npz'),
            *(str(RAW / CELL / f'phase-{phase}-matrix.npy') for phase in range(PERIOD)),
            *(str(RAW / CELL / f'phase-{phase}-residual.npz') for phase in range(PERIOD))}


def allocation(receipt, admission, plan, card, plan_pin):
    """Pure P6 authority delta; no target, source, Git or process access."""
    cell = expected_cell(card)
    lane = plan['lanes']['phase']
    command = ['/data2/neural-shrimp/.venv/bin/python', '-I', '-B', str(D / 'phase_preconditioner.py'),
               '--card', str(CARD), '--cell', CELL, '--out', str(RAW)]
    require(plan['status'] == 'reviewed_for_execution' and plan['repository'] == str(R)
            and plan['raw_root'] == str(B) and plan['cpu_threads'] == 1, 'wrong original phase plan')
    require([plan['python'], '-I', '-B', lane['entry'], *lane['arguments']] == command
            and receipt['argv'] == admission['argv'] == command
            and receipt['lane'] == admission['lane'] == 'phase', 'wrong exact original phase command/lane')
    require(lane['output'] == str(RAW) and lane['supervision'] == str(SUP), 'wrong original phase allocation')
    require(receipt['source_commit'] == admission['source_commit'] == SOURCE, 'wrong original P6 execution source')
    require(type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['failure'] is None,
            'original phase producer did not succeed')
    require(all(receipt[key] is True for key in ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')),
            'original closure flags incomplete')
    for pid in (receipt['worker_pid'], admission['supervisor']['pid']):
        require(type(pid) is int and pid > 0, 'original exact owned process identifiers required')
    ap = dict(admission['plan'])
    if not Path(ap['path']).is_absolute(): ap['path'] = str(R / ap['path'])
    require(ap == plan_pin and plan_pin['path'] == str(PLAN) and plan_pin['sha256'] == PLAN_SHA,
            'original exact plan pin differs')
    require(admission['limits'] == lane['limits'] and all(admission['threads'][name] == '1' for name in THREADS),
            'original limits/thread allocation differs')
    source_files = inventory(plan['source_files'])
    expected = {str(Path(path).relative_to(R)): {key: record[key] for key in ('bytes', 'sha256')}
                for path, record in source_files.items()}
    actual = {record['relative_path']: {key: record[key] for key in ('bytes', 'sha256')}
              for record in admission['source_blobs']}
    require(len(actual) == len(admission['source_blobs']) and actual == expected, 'original complete source inventory differs')
    for path, digest in SCIENCE.items():
        require(actual.get(path, {}).get('sha256') == digest, 'reviewed original scientific source/card differs')
    inputs = inventory(card['inputs']); preflight = inventory(lane['preflight_pins'])
    require(all(preflight.get(path) == value for path, value in inputs.items()), 'shared card inputs missing from original preflight')
    for path in (cell['states'], cell['settle_report']):
        require(path in inputs, 'original named-cell input pin absent')
    files = inventory(receipt['output_files'])
    require(set(files) == expected_outputs(), 'exact six-phase original output inventory required')
    return cell, files


def row_authority(report, cell):
    require(type(report['rows']) is list and len(report['rows']) == 1, 'exactly one original P6 row required')
    row = report['rows'][0]
    require(row['cell'] == cell and type(row['period']) is int and row['period'] == PERIOD,
            'saved row differs from original named cascade-P6 cell')
    require(row['certificate_passed'] is False and row['product_order'] == PRODUCT_ORDER,
            'finite proposal scope/order differs')
    require(type(row['phases']) is list and len(row['phases']) == PERIOD
            and all(type(phase['phase']) is int for phase in row['phases'])
            and [phase['phase'] for phase in row['phases']] == list(range(PERIOD)),
            'all six original phase rows in order required')
    for phase in row['phases']:
        require(type(phase['seconds']) in (int, float) and math.isfinite(phase['seconds']) and phase['seconds'] >= 0,
                'invalid original phase timing')
    return row


def check_saved_arrays(row, history, arrays, phase_files, *, dimension=DIMENSION):
    """Same accepted saved numerical formulas; dimension override is for tiny fixtures.

    Main never overrides the target dimension. B remains a supplied binary64
    proposal: no new solve, Jacobian, interval inverse or certificate is computed.
    """
    import numpy as np
    require(type(dimension) is int and dimension > 0, 'positive audit dimension required')
    def array(value, shape, label):
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all(), 'invalid complete binary64 ' + label)
    require(set(arrays) == ARRAYS, 'exactly seven original proposal arrays required')
    array(history, (513, dimension), 'original history')
    shapes = dict(A=(PERIOD, dimension, dimension), centers=(PERIOD, dimension), parameters=(2,),
                  B=(dimension, dimension), product=(dimension, dimension),
                  left_inverse_defect=(dimension, dimension), right_inverse_defect=(dimension, dimension))
    for name, shape in shapes.items(): array(arrays[name], shape, name)
    require(np.array_equal(arrays['centers'], history[-PERIOD:]), 'centres differ from original last six history rows')
    require(np.array_equal(arrays['parameters'], [row['cell']['alpha'], row['cell']['log10T']]),
            'proposal parameters differ from original exact named cell')
    require(type(phase_files) is list and len(phase_files) == PERIOD, 'all six original matrix/residual pairs required')
    identity = np.eye(dimension)
    product = identity.copy()
    closures = []
    for phase in range(PERIOD):
        matrix, values = phase_files[phase]
        array(matrix, (dimension, dimension), f'phase {phase} matrix')
        require(np.array_equal(matrix, arrays['A'][phase]), 'original per-phase matrix differs from stacked A')
        require(set(values) == {'center', 'image', 'residual'}, 'complete original phase residual fields required')
        for name in values: array(values[name], (dimension,), f'phase {phase} ' + name)
        following = arrays['centers'][(phase + 1) % PERIOD]
        require(np.array_equal(values['center'], arrays['centers'][phase])
                and np.array_equal(values['residual'], values['image'] - following),
                'original phase centre/image/residual relation differs')
        denominator = float(np.linalg.norm(following))
        require(denominator > 0 and math.isfinite(denominator), 'nonzero finite original closure normalization required')
        relative = float(np.linalg.norm(values['residual']) / np.linalg.norm(following))
        require(math.isfinite(relative) and relative == row['phases'][phase]['relative_l2_closure'] < 1e-10,
                'original phase closure diagnostic differs or failed')
        closures.append(relative)
        product = matrix @ product
        require(np.isfinite(product).all(), 'nonfinite saved ordered product replay')
    require(np.array_equal(product, arrays['product']), 'original A[5] through A[0] ordered product differs')
    denominator = identity - product
    left = identity - arrays['B'] @ denominator
    right = identity - denominator @ arrays['B']
    require(np.array_equal(left, arrays['left_inverse_defect'])
            and np.array_equal(right, arrays['right_inverse_defect']), 'original left/right inverse-defect arrays differ')
    diagnostics = row['diagnostics']
    recomputed = {}
    for name, value in [('left_inverse_defect_inf', left), ('right_inverse_defect_inf', right),
                        ('product_norm_inf', product), ('inverse_norm_inf', arrays['B'])]:
        recomputed[name] = float(np.linalg.norm(value, ord=np.inf))
    recomputed['condition_2_estimate'] = float(np.linalg.cond(denominator))
    require(all(math.isfinite(value) and value == diagnostics[name] for name, value in recomputed.items()),
            'original saved proposal diagnostics differ')
    return dict(phases_checked=PERIOD, dimension=dimension, seven_proposal_arrays_checked=True,
                ordered_product=PRODUCT_ORDER, phase_relative_l2_closures=closures, diagnostics=recomputed,
                certificate_passed=False, rigorous_inverse_proved=False)


def original_execution(records, receipt, admission, plan, card, files):
    """Join authenticated actual002 metadata; no live observation or arithmetic."""
    recovery, startup, root, completion, host = [records[key] for key in
        ('resource_card', 'worker_startup', 'root_startup', 'completion', 'host_closure')]
    require(recovery['status'] == 'reviewed_for_execution' and recovery['attempt'] == ATTEMPT
            and recovery['scientific_card'] == PINS['scientific_card']
            and recovery['cell'] == expected_cell(card) and recovery['new_output'] == str(RAW)
            and recovery['new_supervision'] == str(SUP), 'wrong recovery002 resource card')
    for limits in (recovery['limits'], plan['lanes']['phase']['limits'], admission['limits']):
        require(limits == ORIGINAL_LIMITS and all(type(value) is int for value in limits.values()),
                'wrong original002 limits')
    require(type(plan['cpu_threads']) is int and plan['cpu_threads'] == 1 and plan['cpu_ids'] == [0]
            and all(type(cpu) is int for cpu in plan['cpu_ids']), 'wrong original002 CPU allocation')
    require(startup['planned_argv'] == receipt['argv'] and startup['affinity'] == [0]
            and all(type(cpu) is int for cpu in startup['affinity']), 'wrong original worker startup command/affinity')
    for field, limit in (('address_space', 'as_bytes'), ('cpu_seconds', 'cpu_seconds'), ('file_size', 'output_bytes')):
        require(startup[field] == [ORIGINAL_LIMITS[limit]] * 2
                and all(type(value) is int for value in startup[field]), 'wrong actual startup rlimits')
    def identity(record, expected, runtime=False):
        require(all(type(record[key]) is int and record[key] == value for key, value in expected.items()),
                'wrong original process birth/group identity')
        if runtime:
            require(type(record['nice']) is int and record['nice'] == 10
                    and type(record['threads']) is int and record['threads'] == 1
                    and record['uid_line'].split() == ['Uid:', '1000', '1000', '1000', '1000'],
                    'wrong original UID/nice/thread metadata')
    for record in (admission['supervisor'], receipt['supervisor']): identity(record, SUPERVISOR, True)
    for record in (startup['identity'], receipt['worker_observed_after_exec']): identity(record, WORKER, True)
    require(receipt['worker_observed_after_exec']['cmdline'].split() == receipt['argv'],
            'wrong original postexec worker command')
    require(root['source_commit'] == SOURCE and root['admission'] == PINS['admission']
            and root['worker_startup'] == PINS['worker_startup'], 'wrong Root original startup binding')
    identity(root['root_supervisor_postexec'], SUPERVISOR)
    identity(root['root_worker_postexec'], WORKER)
    require(completion['schema'] == 'codex-task34-phase002-operational-completion-v1'
            and completion['status'] == 'closed_exit0_pending_independent_saved_audit'
            and completion['source_commit'] == SOURCE and completion['failure'] is None
            and type(completion['session_exit_code']) is int and completion['session_exit_code'] == 0
            and type(completion['original_returncode']) is int and completion['original_returncode'] == 0,
            'wrong closed002 operational completion')
    for key in ('receipt', 'admission', 'scientific_card', 'resource_card', 'plan', 'worker_startup', 'root_startup'):
        require(completion[key] == PINS[key], 'completion original metadata pin differs')
    require(all(completion[key] is False for key in FALSE_FLAGS)
            and completion['saved_numerical_consistency_admitted'] is False,
            'operational completion must not pre-admit science')
    require(completion['source_blobs_checked'] == admission['source_blobs']
            and inventory(completion['original_preflight_pins_checked']) == inventory(plan['lanes']['phase']['preflight_pins'])
            and inventory(completion['original_card_input_pins_checked']) == inventory(card['inputs']),
            'Root original source/input authentication differs')
    require(inventory(completion['outputs']) == files and completion['complete_output_inventory_checked'] is True
            and completion['all14_output_bytes_authenticated'] is True, 'Root whole output authentication differs')
    total = sum(record['bytes'] for record in files.values())
    require(all(type(record['bytes']) is int and record['bytes'] >= 0 for record in files.values())
            and type(receipt['actual_output_bytes']) is int and type(completion['output_bytes']) is int
            and receipt['actual_output_bytes'] == completion['output_bytes'] == total <= ORIGINAL_LIMITS['output_bytes'],
            'original output byte allocation differs')
    closure = completion['original_host_closure']
    groups = sorted({SUPERVISOR['pgid'], WORKER['pgid']})
    sessions = sorted({SUPERVISOR['sid'], WORKER['sid']})
    require(closure['pin'] == PINS['host_closure'] and closure['observation'] == host
            and closure['group_ids_checked'] == groups and closure['session_ids_checked'] == sessions
            and all(type(value) is int for value in closure['group_ids_checked'] + closure['session_ids_checked']),
            'Root original host group/record binding differs')
    require(host['checked_utc'] == '2026-10-19T18:15:54.306291+00:00'
            and host['identities'] == [dict(pid=value['pid'], expected_starttime=value['starttime'], present=False)
                                       for value in (SUPERVISOR, WORKER)]
            and all(item['present'] is False and type(item['pid']) is int and type(item['expected_starttime']) is int
                    for item in host['identities'])
            and host['group_session_members'] == [] and host['scan_permission_errors'] == [],
            'actual Root host closure incomplete')
    return dict(supervisor=SUPERVISOR, worker=WORKER, group_ids=groups, session_ids=sessions,
                root_observation=PINS['host_closure'])


def original_host_closed(identity, members):
    """Current namespace recheck only; pinned Root record supplies actual-host scope."""
    for expected in (SUPERVISOR, WORKER):
        require(identity(expected['pid']) is None, 'original process identity is present')
    for group in sorted({SUPERVISOR['pgid'], SUPERVISOR['sid'], WORKER['pgid'], WORKER['sid']}):
        require(not members(group), 'original recorded group/session closure pending')


def whole_inventory(files):
    """Exact two-level original layout, including directories, without following links."""
    require(RAW.is_absolute() and RAW.resolve(strict=True) == RAW and not RAW.is_symlink(),
            'canonical original output root required')
    expected = {RAW/'report.json', RAW/CELL}
    require(set(RAW.iterdir()) == expected, 'whole original root inventory differs')
    require((RAW/CELL).is_dir() and not (RAW/CELL).is_symlink(), 'original cell directory differs')
    actual = {str(RAW/'report.json')}
    for path in (RAW/CELL).iterdir():
        require(path.is_file() and not path.is_symlink(), 'original output contains nonregular entry')
        actual.add(str(path))
    require(actual == set(files) == expected_outputs(), 'whole original output inventory differs')


def git_blob(path):
    return subprocess.check_output(['git', 'show', SOURCE + ':' + str(Path(path).relative_to(R))], cwd=R, timeout=10)


def pinned_source(path, digest):
    record = pin(path)
    require(record['sha256'] == digest, 'frozen original source/card hash differs')
    data = git_blob(path)
    require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == digest,
            'original committed source/card bytes differ')
    return record


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def run(args):
    require(all(os.environ.get(name) == '1' for name in THREADS)
            and os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'offline CPU one-thread environment required before NumPy import')
    require(args.receipt_sha256 == PINS['receipt']['sha256']
            and args.admission_sha256 == PINS['admission']['sha256'], 'wrong exact002 original metadata hashes')
    import numpy as np
    records = {key: metadata(record) for key, record in PINS.items()}
    card, plan, receipt, admission = [records[key] for key in ('scientific_card', 'plan', 'receipt', 'admission')]
    for key in ('scientific_card', 'plan', 'resource_card'):
        require(pinned_source(PINS[key]['path'], PINS[key]['sha256']) == PINS[key], 'original Git metadata pin differs')
    cell, files = allocation(receipt, admission, plan, card, PINS['plan'])
    execution = original_execution(records, receipt, admission, plan, card, files)
    for record in files.values(): verify(record)
    whole_inventory(files)
    for record in admission['source_blobs']:
        data = git_blob(R / record['relative_path'])
        require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                'original producer source blob differs')
    for record in card['inputs']: verify(record)
    pinned_source(R / 'codex/certify-chaos/supervise.py', SCIENCE['codex/certify-chaos/supervise.py'])
    sys.path.insert(0, str(R / 'codex/certify-chaos'))
    import supervise
    require(Path(supervise.__file__).resolve() == R / 'codex/certify-chaos/supervise.py', 'wrong host closure helper')
    original_host_closed(supervise.ident, supervise.members)
    row = row_authority(metadata(files[str(RAW/'report.json')]), cell)
    history = np.load(cell['states'], allow_pickle=False)
    folder = RAW / CELL
    with np.load(folder / 'preconditioner.npz', allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'duplicate original proposal archive fields')
        arrays = {name: archive[name] for name in archive.files}
    phase_files = []
    for phase in range(PERIOD):
        matrix = np.load(folder / f'phase-{phase}-matrix.npy', allow_pickle=False)
        with np.load(folder / f'phase-{phase}-residual.npz', allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'duplicate original residual archive fields')
            values = {name: archive[name] for name in archive.files}
        phase_files.append((matrix, values))
    checked = check_saved_arrays(row, history, arrays, phase_files)
    for record in [*PINS.values(), *files.values(), *card['inputs']]: verify(record)
    whole_inventory(files)
    original_host_closed(supervise.ident, supervise.members)
    return dict(schema='codex-task34-cascade-p6-002-saved-phase-audit-v1', status='saved_phase_proposals_consistent',
                original_metadata=PINS, source_commit=SOURCE, cell=CELL, period=PERIOD,
                source_blobs_checked=len(admission['source_blobs']), output_files=list(files.values()),
                outputs_checked=len(files), whole_inventory_before_after=True, original_execution_binding=execution,
                current_namespace_closure_rechecked=True, actual_host_closure_authority=PINS['host_closure'],
                original_preflight_authentication_inherited_from=PINS['completion'],
                all_original_runtime_asset_preflight_independently_rechecked=False,
                checked=checked, scientific_row=row, original_external_elapsed_seconds=receipt['external_elapsed_seconds'],
                audit_backend=dict(python=sys.version, numpy=np.__version__, threads={name: os.environ[name] for name in THREADS}),
                **{key: False for key in FALSE_FLAGS}, scope=SCOPE)


def main():
    args = arguments()
    require(not RESULT.exists() and not RESULT.is_symlink() and not RESULT.parent.exists()
            and not RESULT.parent.is_symlink(), 'fresh absent audit output and parent required')
    require(RESULT.parent.parent.resolve(strict=True) == RESULT.parent.parent, 'canonical existing audit base required')
    RESULT.parent.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args)
        result['audit_elapsed_seconds'] = time.monotonic() - started
        save(RESULT, result)
        print(json.dumps(dict(status=result['status'], phases=PERIOD, outputs=result['outputs_checked'], certificate_passed=False)))
    except BaseException as error:
        save(RESULT.parent / 'failure.json', dict(schema='codex-task34-cascade-p6-phase-audit-refusal-v1',
             status='audit_refused', source_commit=SOURCE, cell=CELL, period=PERIOD, original_metadata=PINS,
             requested_receipt_sha256=args.receipt_sha256, requested_admission_sha256=args.admission_sha256,
             error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
             audit_elapsed_seconds=time.monotonic() - started, **{key: False for key in FALSE_FLAGS},
             scope='Original saved evidence is unchanged. This refusal is not a failed numerical proposal or certificate.'))
        raise


if __name__ == '__main__':
    main()
