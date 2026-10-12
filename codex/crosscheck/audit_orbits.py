"""Read-only audit of Task2 stored orbit matrices and trajectories.

No forward evaluations or repository scientific modules are imported. Final
reports require complete receipts for every run. Matrix spectra and algebraic
checks are recomputed; unpersisted phase-return/linear-system vectors remain
explicitly metadata-only checks. This is a floating-point audit, not a proof.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment


DEFAULT_BASE = Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck')
REFERENCE_BASE = Path('/data2/neural-shrimp-data/explore/llm-soft-loop')
CASES = (
    dict(label='p2', run='orbits-001', period=2, alpha=.0624, log_temperature=-.115,
         reference='branch64-p2-t-0.115-alpha-up/branch.json',
         seed='branch64-p2-t-0.115-alpha-up/states.npy', seed_index=0,
         map_relation='Same current CPU float64 map; stable-branch point0.'),
    dict(label='p8', run='orbits-001', period=8, alpha=.0717, log_temperature=-.1205,
         reference='orbit64-zr-p8/orbit.json', seed='orbit64-zr-p8/orbit-state.npy',
         seed_index=None, map_relation='Historical CUDA seed/spectrum used float32 rotary tables; newly corrected under the current fully-float64 map. Differences are map-version comparisons.'),
    dict(label='p2-legacy', run='legacy-p2-001', period=2, alpha=.061579, log_temperature=-.17263,
         reference='orbit64-zr-p2/orbit.json', seed='orbit64-zr-p2/orbit-state.npy',
         seed_index=None, map_relation='Historical CUDA seed/spectrum used float32 rotary tables; newly corrected under the current fully-float64 map. Differences are map-version comparisons.'),
)


def identity(path):
    path = Path(path)
    digest = sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(data)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load_artifact(path):
    """Decode only local, shape/dtype-qualified NPZ references and complex scalars."""
    path = Path(path)
    metadata = json.loads(path.read_text())
    handles = {}
    def unpack(value):
        if isinstance(value, list):
            return [unpack(v) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {'real', 'imag'}:
            return complex(value['real'], value['imag'])
        if {'array','file','shape','dtype'} <= value.keys():
            filename = value['file']
            if Path(filename).name != filename:
                raise ValueError('array reference is not an adjacent NPZ')
            if filename not in handles:
                handles[filename] = np.load(path.parent/filename, allow_pickle=False)
            array = handles[filename][value['array']].copy()
            if list(array.shape) != value['shape'] or str(array.dtype) != value['dtype']:
                raise ValueError('stored array metadata mismatch')
            if not np.isfinite(array).all():
                raise ValueError('orbit artifact contains nonfinite array entries')
            return array
        return {k:unpack(v) for k,v in value.items()}
    try:
        return unpack(metadata)
    finally:
        for handle in handles.values():
            handle.close()


def relative(delta, reference):
    size = np.linalg.norm(reference)
    return float(np.linalg.norm(delta)/size) if size else float(np.linalg.norm(delta))


def close(left, right, *, atol=1e-15, rtol=1e-10):
    return bool(np.allclose(left, right, atol=atol, rtol=rtol))


def encoded(value):
    return [float(value.real), float(value.imag)]


def match_values(reference, values):
    rows, columns = linear_sum_assignment(np.abs(reference[:,None]-values[None,:]))
    return [dict(reference_index=int(i), recomputed_index=int(j), reference=encoded(reference[i]),
                 recomputed=encoded(values[j]), absolute_difference=float(abs(values[j]-reference[i])),
                 relative_difference=float(abs(values[j]-reference[i])/abs(reference[i])) if reference[i] else None)
            for i,j in zip(rows,columns)]


def audit_run(path, repository, hash_cache):
    """Require complete receipt and bind archive bytes to its stated Git commit."""
    path, repository = Path(path), Path(repository)
    if not (path/'receipt.json').exists() or not (path/'run/status.json').exists():
        raise RuntimeError(f'run not complete: {path.name}')
    receipt = json.loads((path/'receipt.json').read_text())
    status = json.loads((path/'run/status.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    if receipt['exit_code'] != 0 or receipt['timed_out'] or status['status'] != 'complete':
        raise RuntimeError(f'run did not complete successfully: {path.name}')
    artifacts = []
    for entry in receipt['artifacts']:
        file = path/entry['path']
        current = identity(file)
        artifacts.append(dict(relative_path=entry['path'], **current,
                              passed=current['bytes']==entry['bytes'] and current['sha256']==entry['sha256']))
    listed = {r['path'] for r in receipt['artifacts']}
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p != path/'receipt.json'}
    source = []
    for entry in manifest['source']:
        current = identity(path/'source'/entry['path'])
        committed = subprocess.run(['git','cat-file','blob',manifest['source_commit']+':'+entry['path']],
                                   cwd=repository,check=True,capture_output=True).stdout
        source.append(dict(relative_path=entry['path'], **current,
                           committed_sha256=sha256(committed).hexdigest(),
                           passed=current['bytes']==entry['bytes'] and current['sha256']==entry['sha256']
                                  and sha256(committed).hexdigest()==entry['sha256']))
    inputs = []
    for entry in json.loads((path/'source/codex/crosscheck/reference-inputs.json').read_text()):
        if entry['path'] not in hash_cache:
            hash_cache[entry['path']] = identity(entry['path'])
        current = hash_cache[entry['path']]
        inputs.append(dict(**current,passed=current['sha256']==entry['sha256'] and current['bytes']==entry['bytes']))
    gates = dict(source_commit_agrees=receipt['source_commit']==manifest['source_commit'],
                 exact_artifact_inventory=listed==actual,all_artifact_hashes=all(r['passed'] for r in artifacts),
                 all_sources_git_bound=all(r['passed'] for r in source),all_reference_hashes=all(r['passed'] for r in inputs))
    return dict(run=str(path),source_commit=manifest['source_commit'],receipt=identity(path/'receipt.json'),
                manifest=identity(path/'manifest.json'),status=status,exit_code=receipt['exit_code'],
                artifacts=artifacts,source=source,inputs=inputs,gates=gates,passed=all(gates.values()))


def audit_case(base, case):
    label,period=case['label'],case['period']
    directory=Path(base)/case['run']/'run'
    result=load_artifact(directory/(label+'-result.json'))
    initial_return=load_artifact(directory/(label+'-initial-return-001.json'))
    initial_record=load_artifact(directory/(label+'-newton-initial-001.json'))
    newton_record=load_artifact(directory/(label+'-newton-001.json'))
    seed=np.load(REFERENCE_BASE/case['seed'],allow_pickle=False)
    if case['seed_index'] is not None:
        seed=seed[case['seed_index']]
    seed=seed.reshape(-1)
    reference=json.loads((REFERENCE_BASE/case['reference']).read_text())
    if case['label']=='p2':
        reference=reference['points'][0]
    state,points,matrix=result['state'],result['points'],result['matrix']
    if state.shape!=(896,) or points.shape!=(period+1,896) or matrix.shape!=(896,896):
        raise ValueError('all full-state dimensions must be896')
    values,vectors=eig(matrix,check_finite=True)
    order=np.argsort(-np.abs(values));values,vectors=values[order],vectors[:,order]
    residuals=np.linalg.norm(matrix@vectors-vectors*values[None,:],axis=0)/np.linalg.norm(vectors,axis=0)
    radius=float(np.max(np.abs(values)))
    stored=result['diagnostics']
    stored_values,stored_vectors=stored['eigenvalues'],stored['eigenvectors']
    if stored_values.shape!=(896,) or stored_vectors.shape!=(896,896):
        raise ValueError('incomplete stored eigensystem')
    stored_residuals=np.linalg.norm(matrix@stored_vectors-stored_vectors*stored_values[None,:],axis=0)/np.linalg.norm(stored_vectors,axis=0)
    eigen_matches=match_values(stored_values,values)
    reference_values=np.array([complex(v[0],v[1]) for v in reference['multipliers']])
    reference_matches=match_values(reference_values,values)
    stored_comparisons=result['spectral_comparison']
    comparison_checks=[]
    for expected,c in zip(reference_values,stored_comparisons):
        matched=stored_values[c['independent_index']]
        comparison_checks.append(c['reference']==expected and close(c['independent'],matched)
                                 and close(c['absolute_difference'],abs(matched-expected),atol=1e-18))
    scale=np.linalg.norm(state)
    returns=np.linalg.norm(points[1:]-points[0],axis=1)/scale
    divisors=[d for d in range(1,period) if period%d==0]
    proper={str(d):float(returns[d-1]) for d in divisors}
    original_closure=relative(initial_return['points'][-1]-seed,seed)
    shift=relative(state-seed,seed);absolute_shift=float(np.linalg.norm(state-seed))
    phase_shifts=np.array([relative(x-seed,seed) for x in points[:-1]])
    if period==2:
        displacement=np.sin(np.arange(1,897,dtype=np.float64));displacement*=1e-8/np.linalg.norm(displacement)
    else:
        displacement=np.zeros(896)
    expected_initial=seed+displacement
    tangent=result['tangent_comparison'];mv=matrix@tangent['direction']
    tangent_error=relative(tangent['phase_product']-mv,mv)
    tangent_absolute=float(np.linalg.norm(tangent['phase_product']-mv))
    phase_checks=[]
    for j,row in enumerate(result['phase_closures']):
        stage=load_artifact(directory/(label+f'-phase-closure-{j+1:03d}.json'))
        phase_checks.append(dict(phase=j,reported_relative_closure=row['relative_closure'],
                                 scalar_repeated_in_stage=stage['phase']==row['phase']==j and stage['relative_closure']==row['relative_closure'],
                                 below_1e_10=0<=row['relative_closure']<1e-10,
                                 independent_forward_recomputed=False))
    history=result['solver']['history'];linear_checks=[]
    for row in history:
        if 'linear' in row:
            linear=row['linear']
            # newton_cycle's committed linear_rtol default1e-10 admits actual
            # residual<=max(5*rtol,1e-12). Vector inputs were not persisted.
            passed=linear['info']==0 and linear['passed'] and 0<=linear['relative_residual']<=5e-10
            linear_checks.append(dict(iteration=row['iteration'],reported_actual_relative_residual=linear['relative_residual'],
                                      declared_gate_5e_10=bool(passed),recomputed_from_vectors=False,
                                      damping=row.get('damping'),iterations=linear['iterations']))
    history_equal=newton_record['history']==history
    gates=dict(
        raw_seed_identity=np.array_equal(initial_return['state'],seed) and np.array_equal(initial_record['reference_seed'],seed),
        source_target_parameters=initial_return['alpha']==case['alpha'] and initial_return['log_temperature']==case['log_temperature'] and initial_return['period']==period,
        initial_return_shape=initial_return['points'].shape==(period+1,896) and np.array_equal(initial_return['points'][0],seed),
        intended_initial_perturbation=np.array_equal(result['initial_state'],expected_initial) and np.array_equal(initial_record['initial_state'],expected_initial),
        state_points_and_solver_identity=np.array_equal(points[0],state) and np.array_equal(result['solver']['state'],state) and np.array_equal(newton_record['state'],state),
        matrix_recomputed_eigen_residuals=bool(np.max(residuals)<1e-8),
        stored_eigen_residuals=bool(np.max(stored_residuals)<1e-8),
        recorded_eigen_residuals_agree=close(stored_residuals,stored['absolute_eigenpair_residuals'],atol=1e-14),
        complete_spectrum_agrees=max(r['absolute_difference'] for r in eigen_matches)<1e-8,
        spectral_radius_agrees=close(radius,stored['spectral_radius'],atol=1e-12),
        spectral_contraction=radius<1,
        recorded_returns_agree=close(returns,stored['relative_returns']),
        closure=float(returns[-1])<1e-11,
        primitive_period=all(v>1e-6 for v in proper.values()),
        recorded_divisor_separations_agree=proper.keys()==stored['proper_divisor_returns'].keys() and all(close(v,stored['proper_divisor_returns'][k]) for k,v in proper.items()),
        recorded_phase_closure_gates=len(phase_checks)==period and all(r['scalar_repeated_in_stage'] and r['below_1e_10'] for r in phase_checks),
        phase_zero_closure_consistency=close(returns[-1],result['phase_closures'][0]['relative_closure']),
        original_closure_agrees=close(original_closure,result['initial_relative_residual']),
        state_displacement_agrees=close(shift,result['seed_shift_relative']) and close(absolute_shift,result['seed_shift_absolute']),
        phase_alignment_agrees=int(np.argmin(phase_shifts))==result['aligned_phase'] and close(float(np.min(phase_shifts)),result['phase_aligned_seed_shift_relative']),
        newton_history_identity=history_equal,
        newton_final_state_closure=close(history[-1]['relative_residual'],returns[-1]) and close(history[-1]['absolute_residual'],np.linalg.norm(points[-1]-state)),
        newton_status=result['solver']['converged'] and result['solver']['reason']=='residual_passed',
        recorded_newton_linear_gates=all(r['declared_gate_5e_10'] and r['damping'] is not None and 0<r['damping']<=1 for r in linear_checks),
        tangent_direction_unit=abs(np.linalg.norm(tangent['direction'])-1)<1e-12,
        tangent_dense_action=close(mv,tangent['composed'],atol=1e-13),
        tangent_chain_agreement=tangent_absolute<1e-10 or tangent_error<1e-8,
        tangent_reported_errors_agree=close(tangent_absolute,tangent['absolute_error'],atol=1e-15) and close(tangent_error,tangent['relative_error'],atol=1e-12),
        every_reference_comparison_retained=len(stored_comparisons)==len(reference_values) and all(comparison_checks))
    gates={key:bool(value) for key,value in gates.items()}
    return dict(label=label,period=period,map_relation=case['map_relation'],passed=bool(all(gates.values())),gates=gates,
                result_json=identity(directory/(label+'-result.json')),
                result_npz=identity(directory/(label+'-result.npz')),
                recomputed_spectral_radius=radius,max_recomputed_eigen_residual=float(np.max(residuals)),
                max_stored_eigen_residual=float(np.max(stored_residuals)),
                maximum_stored_recomputed_eigenvalue_difference=max(r['absolute_difference'] for r in eigen_matches),
                recomputed_eigenvalues=[encoded(v) for v in values],
                eigenpair_residuals=residuals.tolist(),
                all_reference_multiplier_discrepancies=reference_matches,
                relative_returns=returns.tolist(),proper_divisor_returns=proper,
                original_seed_current_map_closure=original_closure,
                original_seed_current_map_absolute_return=float(np.linalg.norm(initial_return['points'][-1]-seed)),
                refined_closure=float(returns[-1]),seed_shift_relative=shift,seed_shift_absolute=absolute_shift,
                initial_perturbation_norm=float(np.linalg.norm(expected_initial-seed)),
                phase_aligned_seed_shift_relative=float(np.min(phase_shifts)),
                reference_reported_radius=reference.get('spectral_radius'),
                reference_reported_closure=reference.get('residual', (reference.get('newton_residuals') or [None])[-1]),
                reference_reported_newton_residuals=reference.get('newton_residuals'),
                phase_checks=phase_checks,newton_history=history,linear_checks=linear_checks,
                tangent_absolute_error=tangent_absolute,tangent_relative_error=tangent_error,
                limits=['No model forward or derivative reevaluation.',
                        'Stored phase-return and linear-solver scalar gates checked against stage records; their vectors were not persisted.',
                        'Tangent-chain saved output compared with dense matrix action; per-step tangent derivatives were not rerun.',
                        'Small floating-point eigenpair residuals are not interval bounds on eigenvalue error.'])


def audit_all(base=DEFAULT_BASE, repository=None):
    base=Path(base)
    repository=Path(repository) if repository else Path(__file__).resolve().parents[2]
    for name in sorted({c['run'] for c in CASES}):
        if not (base/name/'receipt.json').exists():
            raise RuntimeError('Final audit waits for both complete receipts; missing '+name)
    cache={}
    runs=[audit_run(base/name,repository,cache) for name in sorted({c['run'] for c in CASES})]
    cases=[audit_case(base,c) for c in CASES]
    return dict(audit_kind='Offline independent recomputation of saved orbit evidence',
                auditor_source=identity(Path(__file__).resolve()),
                no_model_evaluations=True,scope='Numerical full-state cycle and matrix consistency audit, not a computer-assisted rigorous proof.',
                runs=runs,cases=cases,passed=all(r['passed'] for r in runs+cases))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--repository',type=Path,default=None)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit_all(args.base,args.repository)
    if args.output.exists():
        raise RuntimeError('refusing to overwrite a prior final audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],cases=[dict(label=c['label'],passed=c['passed'],
          radius=c['recomputed_spectral_radius'],closure=c['refined_closure']) for c in report['cases']]),indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
