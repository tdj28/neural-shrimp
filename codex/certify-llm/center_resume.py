"""Bounded fixed-R continuation from a fully completed refinement attempt.

Saved exact inputs/images are reused only after receipt, source, backend and
full-phase admission. Every correction receives eight fresh phase evaluations.
This is proposal setup, never an orbit certificate. No target calls on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import flint
from flint import arb, ctx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import center_refine as base
from cache_io import unpack

OWNED = base.OWNED
SCHEMA = 'codex-llm-center-resume-v1'
# Strict same-source reuse. Changes require a separately reviewed importer;
# declaring a new hash does not make a changed evaluator compatible.
COMMON_SOURCES = (
    'codex/certify-llm/center_refine.py',
    'codex/certify-llm/ball_qwen.py',
    'codex/certify-llm/ball_linear.py',
    'codex/certify-llm/cache_io.py',
    'codex/certify-llm/shooting_correction.py',
    'codex/certify-llm/proof_gates.py',
    'codex/crosscheck/qwen_map.py',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan, *, owned=None):
    owned = OWNED if owned is None else Path(owned)
    fields = {'schema', 'base_plan', 'parent_attempt', 'parent_kind',
              'parent_receipt_sha256', 'parent_source_sha256',
              'max_corrections', 'envelope'}
    if not isinstance(plan, dict) or not fields <= set(plan) or set(plan)-fields-{'purpose', 'basis'}:
        raise ValueError('complete known resume plan fields required')
    if plan['schema'] != SCHEMA or plan['parent_kind'] not in ('center_refine', 'center_resume'):
        raise ValueError('unsupported completed-parent kind or resume schema')
    base.validate_plan(plan['base_plan'], owned=owned)
    if type(plan['max_corrections']) is not int or not 1 <= plan['max_corrections'] <= 3:
        raise ValueError('exactly one to three new fixed-R corrections required')
    envelope = plan['envelope']
    if (not isinstance(envelope, dict)
            or envelope != dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=3600)
            or any(type(v) is not int for v in envelope.values())):
        raise ValueError('only CPU1, 18 GiB, one-hour continuation is admitted')
    if not isinstance(plan['parent_attempt'], str):
        raise ValueError('owned parent path required')
    parent = Path(plan['parent_attempt']).resolve()
    if (parent.parent != owned.resolve() or not parent.is_dir()
            or parent in (Path(plan['base_plan'][k]).resolve() for k in ('prefix_attempt', 'center_seed_attempt'))):
        raise ValueError('separate existing direct-child completed parent required')
    sources = set(COMMON_SOURCES)
    if plan['parent_kind'] == 'center_resume':
        sources.add('codex/certify-llm/center_resume.py')
    if (not base.valid_hash(plan['parent_receipt_sha256'])
            or not isinstance(plan['parent_source_sha256'], dict)
            or set(plan['parent_source_sha256']) != sources
            or any(not base.valid_hash(v) for v in plan['parent_source_sha256'].values())):
        raise ValueError('receipt and complete explicit parent source hashes required')
    return parent


def dyadic(pair):
    if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
        raise ValueError('integer dyadic pairs required')
    return Fraction(pair[0]) * Fraction(2)**pair[1]


def decoded_vector(records, dimension, *, exact=False):
    if not isinstance(records, list) or len(records) != dimension:
        raise ValueError('every full state coordinate must be retained')
    values = []
    for record in records:
        value = unpack(record)
        midpoint, radius = dyadic(record['mid']), dyadic(record['rad'])
        actual = base.packed(value)
        am, ar = dyadic(actual['mid']), dyadic(actual['rad'])
        if am-ar > midpoint-radius or am+ar < midpoint+radius:
            raise ValueError('saved ball narrowed while decoding')
        if exact and (radius != 0 or not value.is_exact() or am != midpoint):
            raise ValueError('exact dyadic centre lost precision or has a radius')
        values.append(value)
    return values


def decoded_centers(records, period, dimension):
    if not isinstance(records, list) or len(records) != period:
        raise ValueError('complete phase tuple required')
    return [decoded_vector(row, dimension, exact=True) for row in records]


def center_records(centers):
    return [[base.packed(v) for v in row] for row in centers]


def same_centers(records, centers):
    got = decoded_centers(records, len(centers), len(centers[0]))
    if any(a != b for left, right in zip(got, centers) for a, b in zip(left, right)):
        raise ValueError('saved phase input is not the exact claimed centre tuple')


def inspect_pass(files, directory, label, centers):
    """Read all complete images; validate exact input and successor bindings."""
    period, dimension = len(centers), len(centers[0])
    outcomes = read_json(files[directory+'/phase-outcomes.json'])
    selected = [v for v in outcomes if v.get('evaluation_pass') == label]
    if (len(selected) != period or [v.get('phase_index') for v in selected] != list(range(period))
            or any(v.get('status') != 'completed' for v in selected)):
        raise ValueError('every ordered phase must be completed')
    images, reuse = [], []
    for index, row in enumerate(selected):
        input_name, point_name = f'{label}-phase-{index:02d}-input.json', f'{label}-phase-{index:02d}-point.json'
        if row.get('input_path') != input_name or row.get('point_path') != point_name:
            raise ValueError('wrong phase artifact path')
        input_path, point_path = files[directory+'/'+input_name], files[directory+'/'+point_name]
        if base.digest(input_path) != row['input_sha256'] or base.digest(point_path) != row['point_sha256']:
            raise ValueError('phase artifact hash mismatch')
        inputs, result = read_json(input_path), read_json(point_path)
        same_centers([inputs['center'], inputs['next_center']], [centers[index], centers[(index+1)%period]])
        values = decoded_vector(result['value'], dimension)
        tangent = decoded_vector(result['tangent'], dimension, exact=True)
        if any(v != 0 for v in tangent):
            raise ValueError('point evaluation must have exact zero tangent')
        images.append(values)
        reuse.append(dict(phase_index=index, input_path=directory+'/'+input_name,
                          input_sha256=row['input_sha256'], point_path=directory+'/'+point_name,
                          point_sha256=row['point_sha256'], stored_image=result['value'],
                          actual_decoded_image=[base.packed(v) for v in values]))
    return images, selected, reuse


def _comparison_before_after(record, before, after):
    """Check full saved norm summaries, without importing an old residual."""
    for key in ('before_inf_upper', 'before_inf_lower', 'after_inf_upper', 'after_inf_lower'):
        decoded_vector([record[key]], 1, exact=True)
    for key in ('upper_bound_decreased', 'strict_residual_norm_decrease_established'):
        if type(record.get(key)) is not bool:
            raise ValueError('both improvement and non-improvement must be recorded')
    # Encoding may expand on load. The saved booleans bind to original saved
    # exact norm bounds, not a newly widened recomputation.
    bup, blo = unpack(record['before_inf_upper']), unpack(record['before_inf_lower'])
    aup, alo = unpack(record['after_inf_upper']), unpack(record['after_inf_lower'])
    if (record['upper_bound_decreased'] != bool(aup < bup)
            or record['strict_residual_norm_decrease_established'] != bool(aup < blo)
            or blo < 0 or alo < 0 or bup < blo or aup < alo):
        raise ValueError('inconsistent saved residual comparison')
    for name, summary in [('before', before), ('after', after)]:
        if record[name+'_inf_upper'] != summary['inf_upper'] or record[name+'_inf_lower'] != summary['inf_lower']:
            raise ValueError('comparison does not bind retained residual summaries')


def _residual_record(record, centers, image_records):
    """Check saved residuals enclose original encoded image minus exact successor.

    Imported residuals are never used for a new correction: decoding a saved
    image can widen it, so a fresh residual is always computed from live balls.
    """
    p, n = len(centers), len(centers[0])
    rows = record.get('residuals')
    if not isinstance(rows, list) or len(rows) != p:
        raise ValueError('full residual record required')
    max_upper, max_lower = Fraction(0), Fraction(0)
    for i in range(p):
        decoded_vector(rows[i], n)
        for j in range(n):
            value = image_records[i][j]
            im, ir = dyadic(value['mid']), dyadic(value['rad'])
            cm = dyadic(base.packed(centers[(i+1)%p][j])['mid'])
            rm, rr = dyadic(rows[i][j]['mid']), dyadic(rows[i][j]['rad'])
            if rm-rr > im-ir-cm or rm+rr < im+ir-cm:
                raise ValueError('retained residual narrows the retained image difference')
            max_upper = max(max_upper, abs(rm)+rr)
            max_lower = max(max_lower, max(Fraction(0), abs(rm)-rr))
    for key in ('inf_upper', 'inf_lower'):
        decoded_vector([record[key]], 1, exact=True)
    if dyadic(record['inf_upper']['mid']) < max_upper or not 0 <= dyadic(record['inf_lower']['mid']) <= max_lower:
        raise ValueError('invalid residual norm enclosure')


def _encloses_images(records, original):
    if len(records) != len(original): raise ValueError('missing source image phase')
    for left, right in zip(records, original):
        if len(left) != len(right): raise ValueError('missing source image coordinate')
        for saved, source in zip(left, right):
            sm, sr = dyadic(saved['mid']), dyadic(saved['rad'])
            im, ir = dyadic(source['mid']), dyadic(source['rad'])
            if sm-sr > im-ir or sm+sr < im+ir:
                raise ValueError('correction images narrow their source phase images')


def _iteration_files(files, directory, previous, previous_images=None, *, full_before=False):
    before = decoded_centers(read_json(files[directory+'/centers-before.json']), base.PERIOD, base.DIMENSION)
    if previous is not None:
        same_centers(center_records(before), previous)
    after = decoded_centers(read_json(files[directory+'/centers-after.json']), base.PERIOD, base.DIMENSION)
    if full_before:
        before_images, before_rows, before_reuse = inspect_pass(files, directory, 'before', before)
        previous_images = [v['stored_image'] for v in before_reuse]
    else:
        before_images, before_rows = None, []
    images, after_rows, reuse = inspect_pass(files, directory, 'after', after)
    all_rows = read_json(files[directory+'/phase-outcomes.json'])
    if all_rows != before_rows+after_rows:
        raise ValueError('omitted, extra or unfinished parent phase rows')
    correction = read_json(files[directory+'/correction-witnesses.json'])
    if correction['schema'] != 'codex-shooting-midpoint-proposal-v1' or correction['certificate_passed'] is not False:
        raise ValueError('incorrect correction witness scope')
    same_centers(correction['original_centers'], before); same_centers(correction['next_centers'], after)
    for field in ('images', 'corrections'):
        if len(correction[field]) != base.PERIOD:
            raise ValueError('missing full correction witness phase')
        for row in correction[field]: decoded_vector(row, base.DIMENSION)
    _encloses_images(correction['images'], previous_images)
    residual_before = read_json(files[directory+'/residual-before.json'])
    residual_after = read_json(files[directory+'/residual-after.json'])
    raw_images = [read_json(files[row['point_path']])['value'] for row in reuse]
    _residual_record(residual_after, after, raw_images)
    # The retained correction witness includes the exact image balls used by
    # its producer, which can be wider than a previous serialized image.
    _residual_record(residual_before, before, correction['images'])
    comparison = read_json(files[directory+'/residual-comparison.json'])
    _comparison_before_after(comparison, residual_before, residual_after)
    return after, images, all_rows, reuse, comparison


def load_completed_parent(plan, expected_assets, prefix_ids, *, binding, seen=()):
    """Admit a complete same-evaluator lineage; never a failed/partial parent."""
    parent = validate_plan(plan)
    if ctx.prec != 2048 or ctx.threads != 1:
        raise ValueError('parent reuse requires the planned 2048-bit CPU1 arithmetic context')
    if parent in seen:
        raise ValueError('cyclic parent lineage')
    receipt, files = base.checked_attempt(parent, plan['parent_receipt_sha256'])
    for relative, expected in plan['parent_source_sha256'].items():
        if base.digest(files['source/'+relative]) != expected or base.digest(ROOT/relative) != expected:
            raise ValueError('parent source differs from the explicitly reviewed current source: '+relative)
    inputs, status = read_json(files['run/inputs.json']), read_json(files['run/status.json'])
    parent_plan = inputs['plan']
    original = parent_plan if plan['parent_kind'] == 'center_refine' else parent_plan['base_plan']
    if original != plan['base_plan']:
        raise ValueError('changed original prefix, parameter or fixed-R seed plan')
    matches = [path for name, path in files.items() if name.startswith('source/codex/certify-llm/')
               and name.endswith('.json') and base.digest(path) == inputs['plan_sha256']]
    if len(matches) != 1 or read_json(matches[0]) != parent_plan:
        raise ValueError('parent plan is not bound to exactly one saved source file')
    driver_path = 'codex/certify-llm/'+plan['parent_kind']+'.py'
    import torch  # Version metadata only; never constructs or loads a model.
    versions = inputs['versions']
    if (inputs['source_commit'] != receipt['source_commit']
            or inputs['driver_sha256'] != plan['parent_source_sha256'][driver_path]
            or inputs['pinned_assets'] != expected_assets or inputs['precision'] != 2048 or inputs['threads'] != 1
            or versions['python_flint'] != flint.__version__ or versions['flint'] != flint.__FLINT_VERSION__
            or versions['numpy'] != np.__version__ or versions['python'] != sys.version
            or versions['torch'] != str(torch.__version__)):
        raise ValueError('parent source, assets or arithmetic backend mismatch')
    for field in ('prefix_source', 'seed_source'):
        if inputs['source_binding'][field] != binding[field]:
            raise ValueError('different original setup lineage')
    prefix = read_json(files['run/prefix-binding.json'])
    bp = plan['base_plan']
    if (prefix['receipt_sha256'] != bp['prefix_receipt_sha256']
            or prefix['graph_sha256'] != bp['prefix_map_source_sha256']
            or prefix['source_commit'] != binding['prefix_source'] or prefix['precision'] != 2048
            or prefix['receiver_construction_checked'] is not True or prefix['steering_layer'] != 12
            or prefix['prefix_sha256'] != base.digest(Path(bp['prefix_attempt'])/'run/prefix-cache.json.gz')):
        raise ValueError('parent prefix receiver binding mismatch')
    if status.get('certificate_passed') is not False:
        raise ValueError('setup parent cannot claim a certificate')
    seconds = receipt.get('elapsed_seconds')
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('finite retained parent resource use required')
    ancestors, previous = [], None
    if plan['parent_kind'] == 'center_refine':
        base.validate_plan(parent_plan)
        if status['status'] != 'proposal_setup_completed' or status['corrections'] != 1:
            raise ValueError('complete one-correction base attempt required')
        centers, images, rows, reuse, comparison = _iteration_files(files, 'run', None, full_before=True)
        if rows != status['phase_outcomes'] or comparison != status['comparison']:
            raise ValueError('base completion marker does not bind every phase')
        count, cumulative_seconds = 2*base.PERIOD, float(seconds)
    else:
        validate_plan(parent_plan)
        earlier = load_completed_parent(parent_plan, expected_assets, prefix_ids, binding=binding, seen=seen+(parent,))
        ancestors = earlier['lineage']; previous = earlier['centers']
        previous_images = [v['stored_image'] for v in earlier['reuse']]
        manifest = read_json(files['run/reuse-manifest.json'])
        if (manifest['parent_attempt'] != parent_plan['parent_attempt']
                or manifest['parent_receipt_sha256'] != parent_plan['parent_receipt_sha256']
                or manifest['lineage'] != earlier['lineage'] or manifest['phases'] != earlier['reuse']
                or manifest['cumulative_prior_phase_evaluations'] != earlier['cumulative_phase_evaluations']
                or manifest['cumulative_prior_elapsed_seconds'] != earlier['cumulative_elapsed_seconds']
                or manifest['residual_recomputed'] is not True or manifest['no_image_radius_discarded'] is not True):
            raise ValueError('resume parent reuse manifest does not bind admitted lineage')
        same_centers(manifest['exact_centers'], previous)
        if status['status'] != 'proposal_resume_completed' or status['corrections'] != parent_plan['max_corrections']:
            raise ValueError('all planned parent corrections must be completed')
        iterations = read_json(files['run/iterations.json'])
        if (iterations != status['iterations'] or len(iterations) != parent_plan['max_corrections']
                or [v['correction_index'] for v in iterations] != list(range(1, len(iterations)+1))):
            raise ValueError('every planned parent comparison must be retained')
        collected = []
        for iteration in iterations:
            number = iteration['correction_index']; directory = f'run/correction-{number:02d}'
            centers, images, rows, reuse, comparison = _iteration_files(files, directory, previous, previous_images)
            if (iteration['status'] != 'completed' or iteration['comparison'] != comparison
                    or iteration['phase_outcomes'] != rows):
                raise ValueError('incomplete or inconsistent parent correction')
            collected += [dict(correction_index=number, **row) for row in rows]
            previous = centers; previous_images = [v['stored_image'] for v in reuse]
        if collected != status['phase_outcomes']:
            raise ValueError('resume completion marker omits phase rows')
        count = earlier['cumulative_phase_evaluations']+base.PERIOD*len(iterations)
        cumulative_seconds = earlier['cumulative_elapsed_seconds']+float(seconds)
    final = dict(attempt=str(parent), receipt_sha256=plan['parent_receipt_sha256'],
                 source_commit=receipt['source_commit'], kind=plan['parent_kind'], elapsed_seconds=float(seconds),
                 artifacts_checked=len(files))
    return dict(centers=centers, images=images, reuse=reuse, lineage=ancestors+[final],
                cumulative_phase_evaluations=count, cumulative_elapsed_seconds=cumulative_seconds)


def run_corrections(model, centers, images, derivatives, inverse, out, count, event, iterations):
    """Evaluate every planned correction, including unsuccessful improvements."""
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError('one to three fixed-R corrections required')
    for number in range(1, count+1):
        directory = out/f'correction-{number:02d}'; directory.mkdir()
        row = dict(correction_index=number, status='running', phase_outcomes=[])
        iterations.append(row); base.write_json(out/'iterations.json', iterations)
        base.write_json(directory/'centers-before.json', center_records(centers))
        before_upper, before_lower, before = base.residual_summary(centers, images)
        base.write_json(directory/'residual-before.json', before)
        event('correction_started', correction_index=number)
        proposal = base.correct(centers, images, derivatives, inverse)
        base.write_json(directory/'correction-witnesses.json', proposal['record'])
        centers = proposal['centers']
        base.write_json(directory/'centers-after.json', center_records(centers))
        images = base.evaluate_pass(model, centers, directory, 'after',
            lambda name, **fields: event(name, correction_index=number, **fields), row['phase_outcomes'])
        after_upper, after_lower, after = base.residual_summary(centers, images)
        base.write_json(directory/'residual-after.json', after)
        comparison = dict(before_inf_upper=base.packed(before_upper), before_inf_lower=base.packed(before_lower),
            after_inf_upper=base.packed(after_upper), after_inf_lower=base.packed(after_lower),
            upper_bound_decreased=bool(after_upper < before_upper),
            strict_residual_norm_decrease_established=bool(after_upper < before_lower),
            note='Setup diagnostic only; every result is retained and no certificate is issued.')
        base.write_json(directory/'residual-comparison.json', comparison)
        row.update(status='completed', comparison=comparison)
        base.write_json(out/'iterations.json', iterations)
        event('correction_completed', correction_index=number, comparison=comparison)
    return centers, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if not out.is_relative_to(OWNED) or out == OWNED:
        raise ValueError('exclusive Task1 output required')
    # Validate output ancestry before creating anything: completed parents are immutable.
    if out.name != 'run' or out.parent.parent != OWNED.resolve():
        raise ValueError('output must be a fresh owned attempt/run directory')
    if args.plan.resolve().parent != HERE or args.plan.suffix != '.json':
        raise ValueError('plan must belong to committed source snapshot')
    plan = read_json(args.plan); parent = validate_plan(plan)
    if out.is_relative_to(parent): raise ValueError('parent attempt is immutable')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); phase = 'admission'; iterations = []
    def event(name, **fields):
        row = dict(event=name, phase=phase, elapsed_seconds=time.monotonic()-started,
                   utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (out/'events.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    try:
        source = os.environ.get('CERT_SOURCE_COMMIT', '')
        if not base.valid_hash(source, 40): raise ValueError('bounded-launcher source commit required')
        if args.plan.resolve().parent != HERE or args.plan.suffix != '.json':
            raise ValueError('plan must belong to committed source snapshot')
        plan = read_json(args.plan); parent = validate_plan(plan)
        if out.is_relative_to(parent): raise ValueError('parent attempt is immutable')
        if os.getpriority(os.PRIO_PROCESS, 0) < 10: os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0, str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap, PINNED_SHA256, PREFIX_IDS
        torch.set_num_threads(1); torch.set_num_interop_threads(1); ctx.prec = 2048; ctx.threads = 1
        with threadpool_limits(limits=1):
            arrays, binding = base.load_bound_inputs(plan['base_plan'], PINNED_SHA256, PREFIX_IDS)
            inherited = load_completed_parent(plan, PINNED_SHA256, PREFIX_IDS, binding=binding)
            base.write_json(out/'inputs.json', dict(plan=plan, plan_sha256=base.digest(args.plan), source_commit=source,
                driver_sha256=base.digest(__file__), source_binding=binding, pinned_assets=PINNED_SHA256,
                precision=ctx.prec, threads=1, pid=os.getpid(),
                versions=dict(python=sys.version, python_flint=flint.__version__, flint=flint.__FLINT_VERSION__, numpy=np.__version__, torch=torch.__version__),
                statement='Fixed-R continuation with complete saved images and fresh full phase passes; no proof.'))
            centers, images = inherited['centers'], inherited['images']
            base.write_json(out/'reuse-manifest.json', dict(parent_attempt=str(parent), parent_receipt_sha256=plan['parent_receipt_sha256'],
                phases=inherited['reuse'], lineage=inherited['lineage'], exact_centers=center_records(centers),
                cumulative_prior_phase_evaluations=inherited['cumulative_phase_evaluations'],
                cumulative_prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'],
                residual_recomputed=True, no_image_radius_discarded=True))
            phase = 'exact_fixed_R'; derivatives = [base.exact_matrix(a) for a in arrays['A']]; inverse = base.exact_matrix(arrays['B'])
            phase = 'model_load'; model = ManualQwenSoftMap.from_pinned_files()
            weights = {name: tensor.numpy() for name, tensor in model.weights.items()}
            for weight in weights.values(): weight.flags.writeable = False
            vector = model.vector.numpy(); vector.flags.writeable = False
            ball = base.BallQwen(model.config, weights, PREFIX_IDS, vector, alpha=base.ALPHA, log10_temperature=base.LOG10_TEMPERATURE,
                linear_options=dict(chunk_rows=64, chunk_cols=896, cache=base.DyadicWeightCache(max_elements=262144)),
                progress=lambda kind, index, values: event(kind, layer=index))
            phase = 'prefix_restore'; bp = plan['base_plan']
            prefix_binding = base.load_prefix(bp['prefix_attempt'], ball, expected_receipt_sha256=bp['prefix_receipt_sha256'],
                expected_map_source_sha256=bp['prefix_map_source_sha256'], expected_assets=PINNED_SHA256,
                expected_config=model.config, expected_weights=weights, expected_vector=vector, expected_steering_layer=model.steering_layer)
            base.write_json(out/'prefix-binding.json', prefix_binding)
            phase = 'fixed_R_corrections'
            centers, images = run_corrections(ball, centers, images, derivatives, inverse, out, plan['max_corrections'], event, iterations)
            rows = [dict(correction_index=v['correction_index'], **row) for v in iterations for row in v['phase_outcomes']]
            if len(rows) != base.PERIOD*plan['max_corrections'] or any(v['status'] != 'completed' for v in rows):
                raise ValueError('all planned fresh phase evaluations required')
        elapsed = time.monotonic()-started
        base.write_json(out/'status.json', dict(status='proposal_resume_completed', certificate_passed=False,
            corrections=plan['max_corrections'], iterations=iterations, phase_outcomes=rows,
            new_phase_evaluations=len(rows), cumulative_phase_evaluations=inherited['cumulative_phase_evaluations']+len(rows),
            prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'], elapsed_seconds=elapsed,
            cumulative_driver_elapsed_seconds=inherited['cumulative_elapsed_seconds']+elapsed,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reason='Fixed-R exact midpoint proposals and fresh residual enclosures only; all proof gates remain separate.'))
        event('completed')
    except Exception as error:
        for iteration in iterations:
            if iteration['status'] == 'running': iteration.update(status='failed', error=repr(error))
            for row in iteration['phase_outcomes']:
                if row['status'] == 'running': row.update(status='failed', error=repr(error))
            directory = out/f"correction-{iteration['correction_index']:02d}"
            base.write_json(directory/'phase-outcomes.json', iteration['phase_outcomes'])
        base.write_json(out/'iterations.json', iterations)
        base.write_json(out/'status.json', dict(status='proposal_resume_failed', certificate_passed=False, phase=phase,
            iterations=iterations, error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__ == '__main__': main()
