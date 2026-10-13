"""Float64 planning estimates for the factored multiple-shooting bound.

This module never evaluates a model and never issues an enclosure or proof gate.
For equal phase radii and a hypothetical common discrepancy epsilon >= ||J_j-A_j||,
its scalar formula is q_i <= ||P_i D|| + epsilon * sum_j Rbound_ij. Here every
matrix, norm, sum, and threshold is computed in ordinary binary64; rounding is
NOT enclosed. The reported inequality is a formula to budget a later rigorous
calculation, not an established bound for the proposed matrices or derivatives.

Signed P_i=A_(i-1)...A_0, Q_j=A_(p-1)...A_(j+1), and
D=I-B(I-P_p) are formed before norms. Rbound_ij is the factored formula in
shooting_gates._factored_bounds, not a dense (p*n)-square inverse. All phase-pair
terms and thresholds are retained, including a failed threshold. Only a separately
committed plan and bounded launcher authorize loading the actual proposal arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import center_refine as base

OWNED = base.OWNED
SCHEMA = 'codex-llm-proposal-budget-v1'
SOURCE_PATHS = ('codex/certify-llm/center_refine.py',
                'codex/certify-llm/shooting_gates.py',
                'codex/crosscheck/qwen_map.py')
ENVELOPE = dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=120)


class PlanningFailure(ValueError):
    """Failed planning arithmetic with all completed diagnostics preserved."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _finite_matrix(value, label, shape=None):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.ndim != 2 or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all()):
        raise ValueError('finite binary64 matrix required: '+label)
    return value


def _matrix_hash(matrix):
    """Hash shape plus canonical little-endian binary64 bytes, not an enclosure."""
    h = hashlib.sha256(json.dumps(list(matrix.shape), separators=(',', ':')).encode())
    h.update(np.asarray(matrix, dtype='<f8', order='C').tobytes())
    return h.hexdigest()


def _norm(matrix, label):
    _finite_matrix(matrix, label)
    value = float(np.max(np.sum(np.abs(matrix), axis=1)))
    if not math.isfinite(value) or value < 0:
        raise ValueError('nonfinite induced infinity norm: '+label)
    return value


def _scalar(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('nonfinite/negative scalar factor: '+label)
    return value


def estimate_budget(A, B):
    """Pure small/large-matrix planning kernel; no input discovery or model use.

    On failure a PlanningFailure carries a JSON-safe partial report. All rows are
    completed before rejecting a nonpositive/overflowing epsilon threshold.
    Zero/negative thresholds are not replaced by zero, absolute values, or a
    different preconditioner. The limiting threshold is strict, never inclusive.
    """
    report = dict(schema='codex-llm-proposal-budget-estimate-v1',
        status='planning_incomplete', certificate_passed=False, rigorous_enclosures=False,
        arithmetic='ordinary float64; all values are planning estimates, not upper enclosures',
        hypothesis='equal nominal phase radii; one common absolute induced-infinity discrepancy epsilon for every J_j-A_j',
        formula='q_i estimate = ||P_i D||_inf + epsilon * sum_j Rbound_ij',
        epsilon_condition='0 <= epsilon < minimum positive row threshold; this is a floating planning condition only',
        phase_rows=[], invalid_threshold_rows=[])
    try:
        if (not isinstance(A, np.ndarray) or A.dtype != np.float64 or A.ndim != 3
                or A.shape[0] < 1 or A.shape[1] < 1 or A.shape[1] != A.shape[2]
                or not np.isfinite(A).all()):
            raise ValueError('nonempty finite binary64 square phase matrices required')
        p, n, _ = A.shape
        _finite_matrix(B, 'B', (n, n))
        report.update(period=p, dimension=n,
                      A_sha256=_matrix_hash(A), B_sha256=_matrix_hash(B))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            identity = np.eye(n, dtype=np.float64)
            prefixes = [identity]
            for i in range(p):
                prefixes.append(_finite_matrix(A[i] @ prefixes[-1], 'P_'+str(i+1), (n,n)))
            D = _finite_matrix(identity-B @ (identity-prefixes[-1]), 'D', (n,n))
            a_norms = [_norm(matrix, 'A_'+str(i)) for i,matrix in enumerate(A)]
            prefix_norms = [_norm(matrix, 'P_'+str(i)) for i,matrix in enumerate(prefixes[:-1])]
            defect_norms, defect_hashes = [], []
            for i, matrix in enumerate(prefixes[:-1]):
                value = _finite_matrix(matrix @ D, 'P_i D', (n,n))
                defect_norms.append(_norm(value, 'P_i D'))
                defect_hashes.append(_matrix_hash(value))
            suffix_norms, suffix_hashes = [None]*p, [None]*p
            suffix = identity
            for j in range(p-1, -1, -1):
                suffix_norms[j] = _norm(suffix, 'Q_'+str(j))
                suffix_hashes[j] = _matrix_hash(suffix)
                if j:
                    suffix = _finite_matrix(suffix @ A[j], 'Q_'+str(j-1), (n,n))
            b_norm = _norm(B, 'B')
            report.update(A_norm_estimates=a_norms, B_norm_estimate=b_norm,
                prefix_norm_estimates=prefix_norms, suffix_norm_estimates=suffix_norms,
                propagated_inverse_defect_estimates=defect_norms,
                cycle_product_norm_estimate=_norm(prefixes[-1], 'P_p'),
                inverse_defect_norm_estimate=_norm(D, 'D'),
                signed_matrix_hashes=dict(prefixes=[_matrix_hash(v) for v in prefixes],
                    suffixes=suffix_hashes, inverse_defect=_matrix_hash(D),
                    propagated_inverse_defects=defect_hashes))
            for i in range(p):
                pairs = []
                for j in range(p):
                    segment = 0.0
                    if j < i:
                        segment = 1.0
                        for k in range(j+1, i):
                            segment = _scalar(segment*a_norms[k], 'segment product')
                    product_term = _scalar(prefix_norms[i]*b_norm*suffix_norms[j], 'prefix/inverse/suffix product')
                    factor = _scalar(product_term+segment, 'Rbound_ij')
                    pairs.append(dict(row_phase=i, column_phase=j,
                        prefix_norm_estimate=prefix_norms[i], inverse_norm_estimate=b_norm,
                        suffix_norm_estimate=suffix_norms[j], product_term_estimate=product_term,
                        remainder_segment_estimate=segment, R_factor_estimate=factor,
                        inverse_defect_constant=defect_norms[i] if j == 0 else 0.0))
                coefficient = _scalar(math.fsum(row['R_factor_estimate'] for row in pairs), 'row coefficient')
                row = dict(phase_index=i, pairs=pairs,
                    inverse_defect_constant=defect_norms[i], epsilon_coefficient=coefficient,
                    epsilon_threshold=None, threshold_error=None)
                report['phase_rows'].append(row)
                numerator = 1.0-defect_norms[i]
                if coefficient == 0.0:
                    row['threshold_error'] = 'zero coefficient: no finite positive threshold is reported'
                else:
                    threshold = numerator/coefficient
                    if math.isfinite(threshold):
                        row['epsilon_threshold'] = threshold
                    if not math.isfinite(threshold) or threshold <= 0.0:
                        row['threshold_error'] = 'nonfinite or nonpositive epsilon threshold'
                if row['threshold_error']:
                    report['invalid_threshold_rows'].append(i)
            report['phase_pair_count'] = p*p
            if report['invalid_threshold_rows']:
                raise ValueError('no valid finite positive threshold in rows '+str(report['invalid_threshold_rows']))
            thresholds = [row['epsilon_threshold'] for row in report['phase_rows']]
            threshold = min(thresholds)
            report.update(status='planning_completed', nominal_uniform_epsilon_threshold=threshold,
                limiting_phase_indices=[i for i,t in enumerate(thresholds) if t == threshold],
                phase_pair_count=p*p)
        return report
    except (ValueError, ArithmeticError) as error:
        report.update(status='planning_failed', error=str(error))
        raise PlanningFailure(str(error), report) from error


def validate_plan(plan, *, owned=None):
    owned = OWNED if owned is None else Path(owned).resolve()
    required = {'schema', 'base_plan', 'source_sha256', 'envelope'}
    if (type(plan) is not dict or not required <= set(plan)
            or set(plan)-required-{'purpose','basis'} or plan['schema'] != SCHEMA):
        raise ValueError('complete known proposal-budget plan required')
    if (type(plan['envelope']) is not dict or plan['envelope'] != ENVELOPE
            or any(type(v) is not int for v in plan['envelope'].values())):
        raise ValueError('exact CPU1/nice10/18GiB/120-second envelope required')
    base.validate_plan(plan['base_plan'], owned=owned)
    hashes = plan['source_sha256']
    if (type(hashes) is not dict or set(hashes) != set(SOURCE_PATHS)
            or any(not base.valid_hash(v) for v in hashes.values())):
        raise ValueError('explicit input-loader/formula/metadata source hashes required')
    return plan


def load_inputs(plan, expected_assets, prefix_ids):
    validate_plan(plan)
    for name, expected in plan['source_sha256'].items():
        if base.digest(ROOT/name) != expected:
            raise ValueError('current reviewed source mismatch: '+name)
    # This authenticates complete prefix and seed receipts and the exact seven-
    # array preconditioner inventory. It never instantiates/evaluates a model.
    return base.load_bound_inputs(plan['base_plan'], expected_assets, prefix_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.name != 'run' or out.parent.parent != OWNED.resolve():
        raise ValueError('new direct-child Task1 attempt/run output required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); phase = 'admission'
    try:
        source = os.environ.get('CERT_SOURCE_COMMIT', '')
        if not base.valid_hash(source, 40):
            raise ValueError('exact source commit from bounded launcher required')
        if args.plan.resolve().parent != HERE or args.plan.suffix != '.json':
            raise ValueError('plan must be in the committed source snapshot')
        plan = json.loads(args.plan.read_text()); validate_plan(plan)
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'
        from threadpoolctl import threadpool_limits, threadpool_info
        sys.path.insert(0, str(ROOT/'codex/crosscheck'))
        from qwen_map import PINNED_SHA256, PREFIX_IDS  # Metadata only.
        captured = dict(plan=plan, plan_sha256=base.digest(args.plan), source_commit=source,
            driver_sha256=base.digest(__file__), certificate_passed=False, rigorous_enclosures=False,
            source_sha256=plan['source_sha256'], pinned_assets=PINNED_SHA256,
            prefix_ids=list(PREFIX_IDS), pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
            versions=dict(python=sys.version, numpy=np.__version__), threads=1,
            statement='Floating planning estimates from existing proposals only; no model, derivative, enclosure, root, or attraction computation.')
        base.write_json(out/'inputs.json', captured)
        with threadpool_limits(limits=1):
            phase = 'bound_input_admission'
            arrays, binding = load_inputs(plan, PINNED_SHA256, PREFIX_IDS)
            captured.update(source_binding=binding, threadpools=threadpool_info(),
                preconditioner_sha256=plan['base_plan']['preconditioner_sha256'],
                array_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
            base.write_json(out/'inputs.json', captured)
            phase = 'floating_planning_arithmetic'
            report = estimate_budget(arrays['A'], arrays['B'])
            base.write_json(out/'estimates.json', report)
        base.write_json(out/'status.json', dict(status='proposal_budget_completed',
            certificate_passed=False, rigorous_enclosures=False, phase_pair_count=report['phase_pair_count'],
            phase_rows_completed=len(report['phase_rows']), estimates_sha256=base.digest(out/'estimates.json'),
            elapsed_seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except Exception as error:
        if isinstance(error, PlanningFailure):
            base.write_json(out/'estimates.json', error.report)
        base.write_json(out/'status.json', dict(status='proposal_budget_failed',
            certificate_passed=False, rigorous_enclosures=False, phase=phase,
            error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__ == '__main__':
    main()
