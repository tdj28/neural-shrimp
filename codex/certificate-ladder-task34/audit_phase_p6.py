"""Explicit Task34 cascade-P6 saved phase audit; no model or producer replay.

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
SOURCE = '36366f7fb9b81e5f90d61aee1d09ae5730dff666'
ATTEMPT, CELL, PERIOD, DIMENSION = 'phase-cascade-p6-001', 'cascade-p6', 6, 896
RAW = B / ATTEMPT
SUP = B / (ATTEMPT + '-supervision')
CARD = D / 'phase-cascade-p3-001-card.json'  # Original shared five-cell card.
PLAN = D / 'phase-cascade-p6-001-plan.json'
RESULT = B / (ATTEMPT + '-audit-001') / 'result.json'
CARD_SHA = 'fad1beeb1eda8afaed84bcf315d27ffc1f1e446f3c2694b9690c1c702e87aa4a'
PLAN_SHA = '07713794955b1a7ca18de10ff8b0038ff4dd8eb9eb398c7967a1cebae2322177'
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    answer = {}
    for key, value in items:
        require(key not in answer, 'duplicate JSON key')
        answer[key] = value
    return answer


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))


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
    require(Path(args.output) == RESULT, 'fresh separate named P6 saved-audit output required')
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


def original_host_closed(receipt, admission, identity, members):
    for pid in (admission['supervisor']['pid'], receipt['worker_pid']):
        require(identity(pid) is None and not members(pid), 'original host process/group closure pending')


def git_blob(path):
    return subprocess.check_output(['git', 'show', SOURCE + ':' + str(Path(path).relative_to(R))], cwd=R)


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
    require(all(os.environ.get(name) == '1' for name in THREADS), 'one BLAS/OpenMP thread required before NumPy import')
    import numpy as np
    card_pin = pinned_source(CARD, CARD_SHA); plan_pin = pinned_source(PLAN, PLAN_SHA)
    receipt_pin = pin(SUP / 'receipt.json'); admission_pin = pin(SUP / 'admission.json')
    require(receipt_pin['sha256'] == args.receipt_sha256 and admission_pin['sha256'] == args.admission_sha256,
            'explicit original receipt/admission pin differs')
    card, plan, receipt, admission = [read(path) for path in (CARD, PLAN, SUP / 'receipt.json', SUP / 'admission.json')]
    cell, files = allocation(receipt, admission, plan, card, plan_pin)
    for record in files.values(): verify(record)
    require({str(path) for path in RAW.rglob('*') if path.is_file()} == set(files), 'whole original directory differs from receipt')
    for record in admission['source_blobs']:
        data = git_blob(R / record['relative_path'])
        require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                'original producer source blob differs')
    for record in card['inputs']: verify(record)
    pinned_source(R / 'codex/certify-chaos/supervise.py', SCIENCE['codex/certify-chaos/supervise.py'])
    sys.path.insert(0, str(R / 'codex/certify-chaos'))
    import supervise
    require(Path(supervise.__file__).resolve() == R / 'codex/certify-chaos/supervise.py', 'wrong host closure helper')
    original_host_closed(receipt, admission, supervise.ident, supervise.members)
    row = row_authority(read(RAW / 'report.json'), cell)
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
    for record in [receipt_pin, admission_pin, card_pin, plan_pin, *files.values(), *card['inputs']]: verify(record)
    original_host_closed(receipt, admission, supervise.ident, supervise.members)
    return dict(schema='codex-task34-cascade-p6-saved-phase-audit-v1', status='saved_phase_proposals_consistent',
                receipt=receipt_pin, admission=admission_pin, producer_card=card_pin, producer_plan=plan_pin,
                source_commit=SOURCE, cell=CELL, period=PERIOD, source_blobs_checked=len(admission['source_blobs']),
                output_files=list(files.values()), outputs_checked=len(files), host_closure_rechecked=True,
                checked=checked, scientific_row=row, original_external_elapsed_seconds=receipt['external_elapsed_seconds'],
                audit_backend=dict(python=sys.version, numpy=np.__version__, threads={name: os.environ[name] for name in THREADS}),
                certificate_passed=False, scope=SCOPE)


def main():
    args = arguments()
    require(not RESULT.exists() and not RESULT.is_symlink() and not RESULT.parent.exists()
            and not RESULT.parent.is_symlink(), 'fresh absent audit output and parent required')
    RESULT.parent.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args)
        result['audit_elapsed_seconds'] = time.monotonic() - started
        save(RESULT, result)
        print(json.dumps(dict(status=result['status'], phases=PERIOD, outputs=result['outputs_checked'], certificate_passed=False)))
    except BaseException as error:
        save(RESULT.parent / 'failure.json', dict(schema='codex-task34-cascade-p6-phase-audit-refusal-v1',
             status='audit_refused', source_commit=SOURCE, cell=CELL, period=PERIOD,
             requested_receipt_sha256=args.receipt_sha256, requested_admission_sha256=args.admission_sha256,
             error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
             audit_elapsed_seconds=time.monotonic() - started, certificate_passed=False,
             scope='Original saved evidence is unchanged. This refusal is not a failed numerical proposal or certificate.'))
        raise


if __name__ == '__main__':
    main()
