"""One exact-dyadic multiple-shooting centre proposal and fresh residual check.

This CPU1 setup driver evaluates all eight phases before and after exactly one
proposal correction. All images/residuals are Arb enclosures; numerical A/B are
fixed exact dyadic proposals, not derivative enclosures or inverse proofs.
No model/checkpoint access occurs on import. A separately pushed plan and bounded
launcher are required for actual target execution; this driver issues no proof.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ball_qwen import BallQwen, exact
from ball_linear import DyadicWeightCache
from cache_io import load_prefix
from shooting_correction import correct

OWNED = Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
PERIOD, DIMENSION = 8, 896
ALPHA, LOG10_TEMPERATURE = .071674, -.115
ARRAY_KEYS = {'A', 'B', 'centers', 'parameters', 'product',
              'left_inverse_defect', 'right_inverse_defect'}
HASH_KEYS = ('prefix_receipt_sha256', 'prefix_map_source_sha256',
             'center_seed_receipt_sha256', 'center_seed_driver_sha256',
             'center_seed_map_source_sha256', 'preconditioner_sha256')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def packed(value):
    if not isinstance(value, arb) or not value.is_finite():
        raise ValueError('finite Arb enclosure required for persistence')
    return dict(mid=[int(v) for v in value.mid().man_exp()],
                rad=[int(v) for v in value.rad().man_exp()])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def valid_hash(value, length=64):
    return (isinstance(value, str) and len(value) == length
            and all(c in '0123456789abcdef' for c in value))


def validate_plan(plan, *, owned=OWNED):
    required = {'schema', 'alpha', 'log10_temperature', 'period', 'dimension',
                'precision', 'max_corrections', 'evaluation_passes', 'envelope',
                'prefix_attempt', 'center_seed_attempt', *HASH_KEYS}
    if not isinstance(plan, dict) or not required <= set(plan) or set(plan)-required-{'purpose', 'basis'}:
        raise ValueError('complete known refinement plan fields required')
    if plan['schema'] != 'codex-llm-center-refine-v1':
        raise ValueError('unsupported refinement schema')
    for key, expected in [('period', PERIOD), ('dimension', DIMENSION),
                          ('precision', 2048), ('max_corrections', 1), ('evaluation_passes', 2)]:
        if type(plan[key]) is not int or plan[key] != expected:
            raise ValueError('unsupported refinement scope: '+key)
    for key, expected in [('alpha', ALPHA), ('log10_temperature', LOG10_TEMPERATURE)]:
        if type(plan[key]) is not float or plan[key] != expected:
            raise ValueError('different assigned parameter: '+key)
    if plan['envelope'] != dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=3600):
        raise ValueError('only the admitted CPU1 one-hour envelope is supported')
    for key in HASH_KEYS:
        if not valid_hash(plan[key]):
            raise ValueError('full lowercase SHA-256 required: '+key)
    paths = []
    for key in ['prefix_attempt', 'center_seed_attempt']:
        if not isinstance(plan[key], str):
            raise ValueError('owned attempt path required')
        p = Path(plan[key]).resolve()
        if p.parent != Path(owned).resolve() or not p.is_dir():
            raise ValueError('existing direct-child Task1 attempt required: '+key)
        paths.append(p)
    if paths[0] == paths[1]:
        raise ValueError('prefix and numerical seed must be distinct attempts')
    return tuple(paths)


def checked_attempt(attempt, receipt_sha256):
    """Hash-check the entire frozen receipt inventory without trusting filenames."""
    attempt = Path(attempt).resolve()
    path = attempt/'receipt.json'
    if not valid_hash(receipt_sha256) or digest(path) != receipt_sha256:
        raise ValueError('receipt hash mismatch')
    receipt = json.loads(path.read_text())
    if not valid_hash(receipt.get('source_commit'), 40):
        raise ValueError('missing source commit in receipt')
    rows = receipt.get('artifacts')
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty artifact inventory required')
    files = {}
    for row in rows:
        name = row['path']
        if not isinstance(name, str) or name in files:
            raise ValueError('duplicate/invalid artifact name')
        p = (attempt/name).resolve()
        if not p.is_relative_to(attempt) or not p.is_file():
            raise ValueError('artifact escapes the owned attempt')
        if p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
            raise ValueError('artifact size/hash mismatch: '+name)
        files[name] = p
    if receipt.get('exit_code') != 0 or receipt.get('timed_out') is not False:
        raise ValueError('upstream setup did not complete successfully')
    return receipt, files


def validate_preconditioner_arrays(arrays, *, period=PERIOD, dimension=DIMENSION):
    """Validate bytes as finite binary64 proposals, without a proof claim."""
    if set(arrays) != ARRAY_KEYS:
        raise ValueError('exact seven-array preconditioner inventory required')
    shapes = dict(A=(period, dimension, dimension), B=(dimension, dimension),
                  centers=(period, dimension), parameters=(2,),
                  product=(dimension, dimension), left_inverse_defect=(dimension, dimension),
                  right_inverse_defect=(dimension, dimension))
    for name, shape in shapes.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError('invalid finite binary64 preconditioner array: '+name)
    if not np.array_equal(arrays['parameters'], np.array([ALPHA, LOG10_TEMPERATURE], dtype=np.float64)):
        raise ValueError('preconditioner parameters differ from assigned map')
    return arrays


def load_bound_inputs(plan, expected_assets, prefix_ids):
    prefix, seed = validate_plan(plan)
    prefix_receipt, prefix_files = checked_attempt(prefix, plan['prefix_receipt_sha256'])
    seed_receipt, seed_files = checked_attempt(seed, plan['center_seed_receipt_sha256'])
    pi = json.loads(prefix_files['run/inputs.json'].read_text())
    si = json.loads(seed_files['run/inputs.json'].read_text())
    ps = json.loads(prefix_files['run/status.json'].read_text())
    ss = json.loads(seed_files['run/status.json'].read_text())
    for inputs, receipt in [(pi, prefix_receipt), (si, seed_receipt)]:
        if (inputs['source_commit'] != receipt['source_commit']
                or inputs['pinned_assets'] != expected_assets
                or tuple(inputs['prefix_ids']) != tuple(prefix_ids)
                or inputs['alpha_ratio'] != list(ALPHA.as_integer_ratio())
                or inputs['log10_temperature_ratio'] != list(LOG10_TEMPERATURE.as_integer_ratio())):
            raise ValueError('upstream input/source/map contract mismatch')
    if (pi['precision'] != 2048 or pi['arb_context'] != dict(precision=2048, threads=1)
            or pi.get('skip_box') is not True or ps['status'] != 'point_setup_completed'
            or ps['certificate_passed'] is not False):
        raise ValueError('a completed fresh 2048-bit prefix/point attempt is required')
    if digest(prefix_files['source/codex/certify-llm/ball_qwen.py']) != plan['prefix_map_source_sha256']:
        raise ValueError('unreviewed original prefix map source')
    if (digest(seed_files['source/codex/certify-llm/center_seed.py']) != plan['center_seed_driver_sha256']
            or si['driver_sha256'] != plan['center_seed_driver_sha256']
            or digest(seed_files['source/codex/crosscheck/qwen_map.py']) != plan['center_seed_map_source_sha256']
            or si['independent_map_sha256'] != plan['center_seed_map_source_sha256']):
        raise ValueError('unreviewed numerical seed construction source')
    if (si['period'] != PERIOD or si['dimension'] != DIMENSION
            or si['parameter_array_order'] != ['alpha', 'log10_temperature']
            or ss['status'] != 'proposal_completed' or ss['certificate_passed'] is not False):
        raise ValueError('incomplete/wrong numerical seed scope')
    metadata = json.loads(seed_files['run/preconditioner.json'].read_text())
    phases = ss['phase_artifacts']
    if (len(phases) != PERIOD or [v['index'] for v in phases] != list(range(PERIOD))
            or any(v['passed'] is not True for v in phases)
            or metadata['phase_artifacts'] != phases or metadata['certificate_passed'] is not False):
        raise ValueError('every numerical phase gate must be retained and passed')
    for phase in phases:
        saved = json.loads(seed_files[f"run/phase-{phase['index']:02d}.json"].read_text())
        if saved != phase or digest(seed_files['run/'+phase['path']]) != phase['sha256']:
            raise ValueError('numerical phase receipt mismatch')
    preconditioner = seed_files['run/preconditioner.npz']
    if (digest(preconditioner) != plan['preconditioner_sha256']
            or metadata['arrays_sha256'] != plan['preconditioner_sha256']
            or ss['preconditioner_sha256'] != plan['preconditioner_sha256']):
        raise ValueError('preconditioner hash mismatch')
    with np.load(preconditioner, allow_pickle=False) as archive:
        if len(archive.files) != len(ARRAY_KEYS) or set(archive.files) != ARRAY_KEYS:
            raise ValueError('duplicate or unexpected preconditioner NPZ entries')
        arrays = {k: archive[k].copy() for k in archive.files}
    validate_preconditioner_arrays(arrays)
    return arrays, dict(prefix_source=prefix_receipt['source_commit'],
                        seed_source=seed_receipt['source_commit'],
                        prefix_artifacts_checked=len(prefix_files), seed_artifacts_checked=len(seed_files))


def exact_matrix(value):
    n, m = value.shape
    matrix = arb_mat(n, m)
    for i in range(n):
        for j in range(m):
            matrix[i, j] = exact(value[i, j])
    return matrix


def residual_summary(centers, images):
    p, n = len(centers), len(centers[0])
    residuals = [[images[i][j]-centers[(i+1)%p][j] for j in range(n)] for i in range(p)]
    if any(not v.is_finite() for row in residuals for v in row):
        raise ValueError('nonfinite complete residual enclosure')
    upper = max(v.abs_upper() for row in residuals for v in row)
    lower = max(v.abs_lower() for row in residuals for v in row)
    return upper, lower, dict(residuals=[[packed(v) for v in row] for row in residuals],
                              inf_upper=packed(upper), inf_lower=packed(lower))


def evaluate_pass(model, centers, out, label, event, outcomes):
    """Persist each exact input and complete image immediately; no batched calls."""
    if label not in ('before', 'after') or not centers or not centers[0]:
        raise ValueError('complete named evaluation pass required')
    n = len(centers[0])
    if any(len(row) != n or any(not isinstance(v, arb) or not v.is_finite() or not v.is_exact() for v in row) for row in centers):
        raise ValueError('exact finite dyadic phase centres required')
    images = []
    for index, center in enumerate(centers):
        input_path = out/f'{label}-phase-{index:02d}-input.json'
        point_path = out/f'{label}-phase-{index:02d}-point.json'
        write_json(input_path, dict(center=[packed(v) for v in center],
                                   next_center=[packed(v) for v in centers[(index+1)%len(centers)]]))
        row = dict(evaluation_pass=label, phase_index=index, status='running',
                   input_path=input_path.name, input_sha256=digest(input_path))
        outcomes.append(row); write_json(out/'phase-outcomes.json', outcomes)
        event('phase_started', evaluation_pass=label, phase_index=index)
        began = time.monotonic()
        value, tangent = model.forward(center)
        if (len(value) != n or len(tangent) != n
                or any(not isinstance(v, arb) or not v.is_finite() for v in value+tangent)
                or any(not v.is_exact() or v != 0 for v in tangent)):
            raise ValueError('finite full-dimensional primal and exact zero tangent required')
        write_json(point_path, dict(value=[packed(v) for v in value], tangent=[packed(v) for v in tangent]))
        row.update(status='completed', point_path=point_path.name, point_sha256=digest(point_path),
                   seconds=time.monotonic()-began, maximum_radius=packed(max(v.rad() for v in value)))
        write_json(out/'phase-outcomes.json', outcomes); event('phase_completed', **row)
        images.append(value)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(OWNED) or out == OWNED:
        raise ValueError('exclusive Task1 output required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); phase = 'admission'; outcomes = []
    def event(name, **fields):
        row = dict(event=name, phase=phase, elapsed_seconds=time.monotonic()-started,
                   utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    try:
        source = os.environ.get('CERT_SOURCE_COMMIT', '')
        if not valid_hash(source, 40):
            raise ValueError('exact source commit from bounded launcher required')
        if args.plan.resolve().parent != HERE or args.plan.suffix != '.json':
            raise ValueError('plan must be in the committed source snapshot')
        plan = json.loads(args.plan.read_text()); validate_plan(plan)
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        import flint
        from threadpoolctl import threadpool_limits
        sys.path.insert(0, str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap, PINNED_SHA256, PREFIX_IDS
        torch.set_num_threads(1); torch.set_num_interop_threads(1); ctx.prec = 2048; ctx.threads = 1
        with threadpool_limits(limits=1):
            arrays, binding = load_bound_inputs(plan, PINNED_SHA256, PREFIX_IDS)
            write_json(out/'inputs.json', dict(plan=plan, plan_sha256=digest(args.plan),
                source_commit=source, driver_sha256=digest(__file__), source_binding=binding,
                pinned_assets=PINNED_SHA256, precision=ctx.prec, threads=1, pid=os.getpid(),
                versions=dict(python=sys.version, python_flint=flint.__version__, flint=flint.__FLINT_VERSION__, numpy=np.__version__, torch=torch.__version__),
                statement='one multiple-shooting midpoint proposal and two fresh eight-phase point evaluations; no proof'))
            phase = 'exact_dyadic_proposals'; event('proposal_conversion_started')
            centers = [[exact(v) for v in row] for row in arrays['centers']]
            derivatives = [exact_matrix(a) for a in arrays['A']]; inverse = exact_matrix(arrays['B'])
            write_json(out/'centers-before.json', [[packed(v) for v in row] for row in centers])
            phase = 'model_load'; event('model_load_started')
            model = ManualQwenSoftMap.from_pinned_files()
            weights = {name: tensor.numpy() for name, tensor in model.weights.items()}
            for weight in weights.values(): weight.flags.writeable = False
            vector = model.vector.numpy(); vector.flags.writeable = False
            ball = BallQwen(model.config, weights, PREFIX_IDS, vector, alpha=ALPHA,
                log10_temperature=LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64, chunk_cols=896, cache=DyadicWeightCache(max_elements=262144)),
                progress=lambda kind, index, values: event(kind, layer=index))
            phase = 'prefix_restore'
            prefix_binding = load_prefix(plan['prefix_attempt'], ball,
                expected_receipt_sha256=plan['prefix_receipt_sha256'], expected_map_source_sha256=plan['prefix_map_source_sha256'],
                expected_assets=PINNED_SHA256, expected_config=model.config, expected_weights=weights,
                expected_vector=vector, expected_steering_layer=model.steering_layer)
            write_json(out/'prefix-binding.json', prefix_binding); event('prefix_restored')
            phase = 'before_evaluation'
            images = evaluate_pass(ball, centers, out, 'before', event, outcomes)
            before_upper, before_lower, before_record = residual_summary(centers, images)
            write_json(out/'residual-before.json', before_record)
            phase = 'single_correction'; event('correction_started')
            proposal = correct(centers, images, derivatives, inverse)
            write_json(out/'correction-witnesses.json', proposal['record'])
            # The entire correction enclosure is persisted before selecting the
            # returned exact midpoint centres for any new model evaluation.
            centers = proposal['centers']
            write_json(out/'centers-after.json', [[packed(v) for v in row] for row in centers])
            event('correction_saved')
            phase = 'after_evaluation'
            after_images = evaluate_pass(ball, centers, out, 'after', event, outcomes)
            after_upper, after_lower, after_record = residual_summary(centers, after_images)
            write_json(out/'residual-after.json', after_record)
            comparison = dict(before_inf_upper=packed(before_upper), after_inf_upper=packed(after_upper),
                before_inf_lower=packed(before_lower), after_inf_lower=packed(after_lower),
                upper_bound_decreased=bool(after_upper < before_upper),
                strict_residual_norm_decrease_established=bool(after_upper < before_lower),
                note='Residual comparison is a setup diagnostic, not existence, uniqueness, primitive period, or attraction.')
            write_json(out/'residual-comparison.json', comparison)
            if len(outcomes) != 16 or any(v['status'] != 'completed' for v in outcomes):
                raise ValueError('both complete eight-phase evaluations are required')
        write_json(out/'status.json', dict(status='proposal_setup_completed', certificate_passed=False,
            corrections=1, phase_outcomes=outcomes, comparison=comparison,
            elapsed_seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reason='Exact midpoint proposal plus fresh residual enclosures; no root or stability proof gates run.'))
        event('completed')
    except Exception as error:
        for outcome in outcomes:
            if outcome['status'] == 'running':
                outcome.update(status='failed', error=repr(error))
        write_json(out/'phase-outcomes.json', outcomes)
        write_json(out/'status.json', dict(status='proposal_setup_failed', certificate_passed=False,
            phase=phase, phase_outcomes=outcomes, error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__ == '__main__':
    main()
