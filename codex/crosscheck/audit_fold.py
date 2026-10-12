"""Read saved fold outputs only; no model or research-tool imports."""
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np

ATTEMPT = Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck/fold-001')
RUN = ATTEMPT/'run'
DATA = Path('/data2/neural-shrimp-data/explore/llm-soft-loop')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def raw(name):
    return json.loads((RUN/name).read_text())


def load(value):
    if isinstance(value, dict):
        if 'array' in value and 'file' in value:
            path = RUN/value['file']
            if path.parent.resolve() != RUN.resolve():
                raise ValueError('unexpected artifact path')
            with np.load(path, allow_pickle=False) as archive:
                result = archive[value['array']].copy()
            assert list(result.shape) == value['shape']
            assert str(result.dtype) == value['dtype']
            return result
        return {key: load(v) for key,v in value.items()}
    if isinstance(value, list):
        return [load(v) for v in value]
    return value


def norm(vector):
    return np.sqrt(np.sum(np.abs(vector)**2))


def scalar(value):
    return float(value)


def scalar_prediction(u_in, u_out, epsilon, a, b):
    """Independent Decimal60 strict discrete crossing and float64 arctan formula."""
    if a <= 0 or b <= 0 or epsilon <= 0:
        raise ValueError('positive-oriented coefficients required')
    with localcontext() as context:
        context.prec = 60
        u, bound = Decimal(str(u_in)), Decimal(str(u_out))
        aa, bb, ee = map(lambda v: Decimal(str(v)), (a,b,epsilon))
        for count in range(1,100001):
            u = u+aa*ee+bb*u*u
            if u > bound:
                break
        else:
            raise ValueError('scalar prediction is censored')
    scale = math.sqrt(b/(a*epsilon))
    integral = (math.atan(u_out*scale)-math.atan(u_in*scale))/math.sqrt(a*b*epsilon)
    return dict(discrete_G=count, continuous_G=integral)


result = load(raw('fold-result.json'))
solver, nf = result['solver'], result['normal_form']
matrix = solver['matrix']
ld = np.longdouble
mm, rr, ll = matrix.astype(ld), nf['right'].astype(ld), nf['left'].astype(ld)
points = result['points'].astype(ld)
returns = [scalar(norm(p-points[0])/norm(points[0])) for p in points[1:]]
eig = np.linalg.eigvals(matrix)
critical = int(np.argmin(np.abs(eig-1)))
complement = np.delete(eig, critical)
parameter_files = sorted(RUN.glob('parameter-derivative-*.json'))
parameter = load(json.loads(parameter_files[-1].read_text()))
second_files = sorted(RUN.glob('directional-second-*.json'))
second = load(json.loads(second_files[-1].read_text()))
assert np.array_equal(parameter['state'], solver['state'])
assert parameter['alpha'] == solver['alpha']
assert np.array_equal(second['state'], solver['state'])
assert second['alpha'] == solver['alpha']
assert np.array_equal(second['right'], nf['right'])
a = ll @ parameter['derivative'].astype(ld)
b = (ll @ second['derivative'].astype(ld))/2
manifest = json.loads((ATTEMPT/'manifest.json').read_text())
source_mismatches = [row['path'] for row in manifest['source']
    if sha(ATTEMPT/'source'/row['path']) != row['sha256']]
receipt = json.loads((ATTEMPT/'receipt.json').read_text())
artifact_mismatches = [row['path'] for row in receipt['artifacts']
    if (ATTEMPT/row['path']).stat().st_size != row['bytes'] or sha(ATTEMPT/row['path']) != row['sha256']]
ad = result['ad_coefficients']
original_passage_path = DATA/'passage-p2-t-0.115/passage.json'
independent_passage_path = ATTEMPT.parent/'passages-001/run/passages-summary.json'
original_passage = json.loads(original_passage_path.read_text())
independent_passage = json.loads(independent_passage_path.read_text())
old_a,old_b = abs(original_passage['normal_form']['a']),abs(original_passage['normal_form']['b'])
new_a,new_b = abs(ad['a']),abs(ad['b'])
passage_rows = []
assert len(original_passage['rows']) == len(independent_passage['rows']) == 18
for old_row,model_row in zip(original_passage['rows'],independent_passage['rows']):
    assert old_row['eps'] == model_row['eps'] and old_row['entrance'] == model_row['entrance']
    assert old_row['u_in'] == model_row['u_in'] and old_row['u_out'] == model_row['u_out'] == original_passage['U']
    old_prediction = scalar_prediction(old_row['u_in'],old_row['u_out'],old_row['eps'],old_a,old_b)
    new_prediction = scalar_prediction(old_row['u_in'],old_row['u_out'],old_row['eps'],new_a,new_b)
    passage_rows.append(dict(epsilon=old_row['eps'],entrance=old_row['entrance'],u_in=old_row['u_in'],
        original_full_model_G=old_row['measured_G'],independent_full_model_G=model_row['measured_G'],
        original_quadratic_G=old_row['predicted_discrete_G'],
        reconstructed_original_quadratic_G=old_prediction['discrete_G'],new_ad_quadratic_G=new_prediction['discrete_G'],
        original_continuous_G=old_row['predicted_arctan_G'],
        reconstructed_original_continuous_G=old_prediction['continuous_G'],new_ad_continuous_G=new_prediction['continuous_G'],
        new_minus_original_quadratic_G=new_prediction['discrete_G']-old_row['predicted_discrete_G'],
        new_quadratic_minus_independent_model_G=new_prediction['discrete_G']-model_row['measured_G'],
        new_minus_original_continuous_G=new_prediction['continuous_G']-old_row['predicted_arctan_G']))
audit_gates = dict(
    all_sources_match=not source_mismatches, all_receipted_artifacts_match=not artifact_mismatches,
    worker_completed=raw('status.json')['status']=='complete' and receipt['exit_code']==0 and not receipt['timed_out'],
    result_passed=result['passed'], matrix_dimension=matrix.shape==(896,896),
    modes_normalized=abs(norm(rr)-1)<1e-10 and abs(ll@rr-1)<1e-10,
    neutral_critical_mode=abs(eig[critical]-1)<1e-10,
    complementary_contraction=np.max(np.abs(complement))<1,
    mode_residuals=norm(mm@rr-rr)<1e-10 and norm(mm.T@ll-ll)/norm(ll)<1e-10,
    primitive_and_closed=returns[0]>1e-6 and returns[1]<1e-11,
    both_phases_closed=max(result['phase_closures'])<1e-11,
    saved_coefficient_dots_match=abs(a-ad['a'])<1e-10 and abs(b-ad['b'])<1e-10 and abs(a*b-ad['ab'])<1e-10,
    old_scalar_counts_reproduced=all(r['original_quadratic_G']==r['reconstructed_original_quadratic_G'] for r in passage_rows),
    old_continuous_predictions_reproduced=all(abs(r['original_continuous_G']-r['reconstructed_original_continuous_G'])<1e-10 for r in passage_rows),
    new_ad_preserves_all_18_scalar_counts=all(r['new_minus_original_quadratic_G']==0 for r in passage_rows),
    independent_full_model_counts_match=all(r['original_full_model_G']==r['independent_full_model_G'] for r in passage_rows))
audit_gates = {key:bool(value) for key,value in audit_gates.items()}
out = dict(scope='Independent saved-array arithmetic and eigensystem audit, without model calls; no interval certificate.',
    audit_source_sha256=sha(Path(__file__).resolve()),
    source_commit=manifest['source_commit'], source_hashes={r['path']:r['sha256'] for r in manifest['source']},
    source_mismatches=source_mismatches, receipt_artifact_mismatches=artifact_mismatches,
    receipt_sha256=sha(ATTEMPT/'receipt.json'), result_json_sha256=sha(RUN/'fold-result.json'),
    result_arrays_sha256=sha(RUN/'fold-result.npz'), worker_status=raw('status.json'),
    solver=dict(converged=solver['converged'], reason=solver['reason'], alpha=solver['alpha'],
        augmented_residual=solver['residual'], history=solver['history']),
    recomputed=dict(matrix_shape=list(matrix.shape), norm_right=scalar(norm(rr)), left_dot_right=scalar(ll@rr),
        right_plus_one_residual=scalar(norm(mm@rr-rr)),
        left_plus_one_relative_residual=scalar(norm(mm.T@ll-ll)/norm(ll)),
        critical_multiplier=[float(eig[critical].real),float(eig[critical].imag)],
        complementary_radius=float(np.max(np.abs(complement))),
        critical_separation=float(np.min(np.abs(complement-eig[critical]))),
        relative_returns=returns, a=scalar(a), b=scalar(b), ab=scalar(a*b),
        a_minus_saved=scalar(a-ad['a']),b_minus_saved=scalar(b-ad['b']),ab_minus_saved=scalar(a*b-ad['ab'])),
    prerequisite_gates=result['prerequisite_gates'], derivative_gates=result['derivative_gates'],
    phase_closures=result['phase_closures'], state_shift_relative=result['state_shift_relative'],
    alpha_difference=result['alpha_difference'],
    a_steps=[dict(row,relative_error_to_ad=abs(row['a']-float(a))/abs(float(a))) for row in nf['a_steps']],
    b_steps=[dict(row,relative_error_to_ad=abs(row['b']-float(b))/abs(float(b))) for row in nf['b_steps']],
    derivative_discrepancy=result['derivative_discrepancy'], reference_comparison=result['reference_comparison'],
    result_passed=result['passed'],
    scalar_passage_check=dict(scope='Same saved U, scalar entrances, epsilons and strict first exit; new AD coefficients only. Original fold alpha retained in existing full-model runs; this is not a new full-state passage experiment.',
        original_passage=dict(path=str(original_passage_path),sha256=sha(original_passage_path)),
        independent_passage=dict(path=str(independent_passage_path),sha256=sha(independent_passage_path)),
        U=original_passage['U'],positive_a=new_a,positive_b=new_b,rows=passage_rows,
        largest_continuous_change=max(abs(r['new_minus_original_continuous_G']) for r in passage_rows),
        largest_quadratic_minus_model_count=max(abs(r['new_quadratic_minus_independent_model_G']) for r in passage_rows)),
    audit_gates=audit_gates, audit_passed=all(audit_gates.values()),
    interpretation='Fold has a neutral +1 critical mode to numerical precision and contracting complement; a computed eigenvalue slightly below one does not establish fold attraction.')
print(json.dumps(out,indent=2,allow_nan=False))
if not out['audit_passed']:
    raise SystemExit(1)
