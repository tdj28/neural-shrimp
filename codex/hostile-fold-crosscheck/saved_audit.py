"""Independent Task7 saved-array audit, without model imports or replay.

Authenticating captured bytes does not prove the implementation or backend.
This numerical audit is evidence rung2; certificate_passed is always false.
The external caller pins the completed receipt, producing commit and run plan.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

OWNED=Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
AUDIT_ENVELOPE=dict(cpu_threads=1,nice=10,memory_gib=4,timeout_seconds=180)
EXPECTED_SOURCES=('codex/hostile-fold-crosscheck/fold_runtime.py',
 'codex/hostile-fold-crosscheck/fold_worker.py','codex/hostile-fold-crosscheck/assets.py',
 'codex/hostile-fold-crosscheck/resource_pilot.py','codex/confirm-fold/fold_setup.py',
 'codex/crosscheck/qwen_map.py','codex/crosscheck/numerics.py')
EXPECTED_CAPS=dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=1800,
 minimum_start_available_gib=28,minimum_runtime_available_gib=16)
EXPECTED_TOLERANCES=dict(augmented=1e-11,closure=1e-10,primitive=1e-6,
 critical=1e-8,imaginary=1e-10,complementary_limit=.95,isolation=1e-3,
 eigen_residual=1e-8,dense_jvp_relative=1e-10,composed_fd_relative=1e-5,
 composed_fd_floor=1e-7,composed_fd_reduction=.3,coefficient_floor=1e-10,
 coefficient_fd_relative=1e-3)
PARAMETER_STEPS=(7e-6,2e-6,7e-7)
STATE_STEPS=(3e-4,1e-4,3e-5)
CHECKPOINT_ASSETS={'model.safetensors':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
 'config.json':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'}
SEED_SHA='d574d0740da3d026276c508527b6001e642c35daace57da7755ec48fd4036123'
VECTOR_SHA='d18395239dba63c7deacb8aa9431d768269617c9cc44a98f89cc1e22423b5958'
PREFIX_IDS=[151644,8948,198,2610,525,458,15235,7404,448,2441,15235,13,1446,614,4583,11290,13,31733,1910,311,22729,8820,498,1366,13,151645,198,151644,872,198,9707,0,151645,198,151644,77091,198]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 return h.hexdigest()


def plain(value):
 import numpy as np
 if isinstance(value,np.ndarray):return plain(value.tolist())
 if isinstance(value,np.generic):return plain(value.item())
 if isinstance(value,complex):return {'real':value.real,'imag':value.imag}
 if isinstance(value,dict):return {str(k):plain(v)for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [plain(v)for v in value]
 return value


def same_json(a,b):return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)


def inventory(attempt,receipt_sha,source_commit,plan_sha):
 if not re.fullmatch('[0-9a-f]{40}',source_commit)or any(not re.fullmatch('[0-9a-f]{64}',p)for p in (receipt_sha,plan_sha)):
  raise ValueError('external full source/receipt/plan pins required')
 if sha(attempt/'receipt.json')!=receipt_sha:raise ValueError('external receipt hash differs')
 receipt=json.loads((attempt/'receipt.json').read_text())
 if (receipt.get('source_commit')!=source_commit or receipt.get('plan_sha256')!=plan_sha
  or type(receipt.get('exit_code'))is not int or receipt['exit_code']!=0
  or receipt.get('timed_out')is not False or receipt.get('status')!='fold_attempt_completed'
  or receipt.get('worker_termination_verified')is not True or receipt.get('uncertain_ownership')!=[]
  or receipt.get('certificate_passed')is not False or receipt.get('resource_probe_only')is not False
  or receipt.get('scientific_fold_result')is not True):raise ValueError('successful complete owned fold attempt required')
 files={}
 for row in receipt['artifacts']:
  name=row['path'];path=attempt/name
  if (type(name)is not str or Path(name).is_absolute()or '..'in Path(name).parts
      or Path(name).as_posix()!=name or name in files or path.resolve()!=path or path.is_symlink()
      or not path.is_file()or path.stat().st_size!=row['bytes']or sha(path)!=row['sha256']):
   raise ValueError('invalid original artifact inventory: '+str(name))
  files[name]=row
 actual={str(p.relative_to(attempt))for p in attempt.rglob('*')if p.is_file()and p!=attempt/'receipt.json'}
 if actual!=set(files):raise ValueError('missing or extra original artifacts')
 plan_path=attempt/'run/original-plan.json'
 if sha(plan_path)!=plan_sha:raise ValueError('actual run plan differs')
 plan=json.loads(plan_path.read_text())
 if (plan.get('schema')!='codex-hostile-fold-run-v1'or plan.get('scope')!='fold_only_no_passages_rung2'
  or not same_json(plan.get('envelope'),EXPECTED_CAPS)
  or not same_json(plan.get('numerics'),dict(chunk=128,max_newton=4,max_backtracks=2))
  or set(plan.get('source_sha256',{}))!=set(EXPECTED_SOURCES)
  or not same_json(plan.get('gates'),dict(tolerances=EXPECTED_TOLERANCES,
      parameter_fd_steps=list(PARAMETER_STEPS),state_fd_steps=list(STATE_STEPS)))):
  raise ValueError('frozen fold target/gate/resource scope differs')
 expected_target=dict(period=2,alpha=.083,active_parameter='log10T',initial_parameter=-.04624653570443442,
  bounds=[-.05,-.043],vector_key='hostility-vs-kindness',seed_index=0)
 if (not same_json(plan.get('target'),expected_target)or plan['seed_sha256']!=SEED_SHA or plan['vector_sha256']!=VECTOR_SHA):raise ValueError('wrong assigned target/assets')
 manifest=json.loads((attempt/'manifest.json').read_text())
 if manifest['source_commit']!=source_commit or manifest['plan_sha256']!=plan_sha:
  raise ValueError('launcher source/plan binding differs')
 expected_capture=set(EXPECTED_SOURCES)|{'codex/certify-llm/derivative_campaign.py',
  'codex/hostile-fold-crosscheck/launch.py','codex/hostile-fold-crosscheck/fold-001-plan.json',
  'codex/hostile-fold-crosscheck/fold-001-run-card.md'}
 captured={row['path']:row for row in manifest['source']}
 if len(captured)!=len(manifest['source'])or set(captured)!=expected_capture:
  raise ValueError('complete captured source bundle required')
 for name,row in captured.items():
  if files['source/'+name]['sha256']!=row['sha256']or files['source/'+name]['bytes']!=row['bytes']:
   raise ValueError('captured source manifest differs')
  if name in plan['source_sha256']and row['sha256']!=plan['source_sha256'][name]:
   raise ValueError('scientific source pin differs')
 if files['source/codex/hostile-fold-crosscheck/fold-001-plan.json']['sha256']!=plan_sha:
  raise ValueError('captured original plan differs')
 inputs=json.loads((attempt/'run/inputs.json').read_text())
 if inputs['source_commit']!=source_commit or inputs['plan_sha256']!=plan_sha or not same_json(inputs['plan'],plan):
  raise ValueError('worker inputs differ')
 if (inputs.get('certificate_passed')is not False or inputs.get('passage_trajectories_evaluated')is not False
  or inputs.get('producer_seed_only')is not True or inputs.get('seed_index')!=0
  or inputs.get('vector_key')!='hostility-vs-kindness'
  or inputs.get('steering_artifact')!=dict(path=inputs['vector_path'],key=inputs['vector_key'],sha256=plan['vector_sha256'])
  or inputs.get('checkpoint_assets')!=CHECKPOINT_ASSETS or inputs.get('prefix_ids')!=PREFIX_IDS):
  raise ValueError('actual asset/seed/scope metadata differs')
 if inputs['resource_admission_sha256']!=files['run/resource-admission.json']['sha256']:
  raise ValueError('resource admission hashlink differs')
 resource_admission=json.loads((attempt/'run/resource-admission.json').read_text())
 if (resource_admission['profile_receipt_sha256']!=plan['profile_authority']['receipt_sha256']
  or resource_admission['historical_receipt_sha256']!=plan['historical_memory_authority']['receipt_sha256']
  or resource_admission['profile']['backend']!=inputs['backend']):raise ValueError('measured profile authority differs')
 status=json.loads((attempt/'run/status.json').read_text())
 for k in ('passed','computational_completed','solver_history_complete'):
  if status.get(k)is not True:raise ValueError('complete qualified worker required: '+k)
 for k in ('certificate_passed','resource_probe_only','passage_trajectories_evaluated'):
  if status.get(k)is not False:raise ValueError('worker scope differs: '+k)
 if status['status']!='fold_qualified'or status['stage']!='qualified':raise ValueError('worker qualification absent')
 return dict(receipt_sha256=receipt_sha,source_commit=source_commit,plan_sha256=plan_sha,
  artifact_count=len(files),captured_source_count=len(captured),source_sha256=plan['source_sha256'],
  captured_source_bytes_authenticated=True,git_commit_blob_verification_performed=False,
  source_trust='External receipt/commit/plan pins plus captured bytes; caller independently verifies Git provenance.',
  plan=plan,inputs=inputs,worker_status=status),files


def load_record(attempt,name,files):
 """Recover exact original numeric arrays; refuse stale shapes/dtypes/hashlinks."""
 import numpy as np
 metadata=json.loads((attempt/'run'/f'{name}.json').read_text())
 binary=attempt/'run'/f'{name}.npz'
 if metadata['array_file_sha256']!=files[f'run/{name}.npz']['sha256']or sha(binary)!=metadata['array_file_sha256']:
  raise ValueError('result array hashlink differs')
 used=set()
 with np.load(binary,allow_pickle=False)as arrays:
  def restore(value):
   if isinstance(value,dict)and set(value)=={'file','key','shape','dtype'}:
    if value['file']!=binary.name or value['key'] in used:raise ValueError('array descriptor differs or repeats')
    used.add(value['key']);array=arrays[value['key']].copy()
    if list(array.shape)!=value['shape']or str(array.dtype)!=value['dtype']or array.dtype.kind not in 'fciub'or not np.isfinite(array).all():
     raise ValueError('nonfinite or wrong-shape/dtype saved array')
    return array
   if isinstance(value,dict):return {k:restore(v)for k,v in value.items()}
   if isinstance(value,list):return [restore(v)for v in value]
   return value
  record=restore(metadata)
  if used!=set(arrays.files):raise ValueError('unreferenced result arrays')
 return record


def numerical_audit(record,initial_seed,*,dimension):
 """Independent linear algebra/contractions; consumes no model or worker helper."""
 import numpy as np
 from scipy.linalg import eig
 from scipy.optimize import linear_sum_assignment
 errors=[];computed={};t=EXPECTED_TOLERANCES
 def array(value,shape,name,complex_ok=False):
  value=np.asarray(value)
  if value.shape!=shape or value.dtype not in ((np.dtype('float64'),np.dtype('complex128'))if complex_ok else(np.dtype('float64'),))or not np.isfinite(value).all():
   raise ValueError('full finite binary64 array required: '+name)
  return value
 def close(name,saved,actual,atol=1e-12,rtol=1e-9):
  if not np.allclose(saved,actual,atol=atol,rtol=rtol,equal_nan=False):errors.append(name)
 def rel(delta,reference):return float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30))
 def dot(a,b):return float(np.sum(np.asarray(a,dtype=np.longdouble)*np.asarray(b,dtype=np.longdouble),dtype=np.longdouble))
 def complex_array(value):
  return np.asarray([complex(x['real'],x['imag'])if isinstance(x,dict)else complex(x)for x in value])
 if (record['schema']!='codex-task7-hostile-fold-v1'or record.get('passed')is not True
  or record.get('certificate_passed')is not False or record.get('passage_trajectories_evaluated')is not False
  or record.get('solver_history_complete')is not True or record.get('stage')!='qualified'
  or record.get('active_parameter_units')!='log10T' or record.get('return_map')!='G=F^2'
  or not same_json(record.get('tolerances'),t)):
  raise ValueError('completed numerical fold record scope differs')
 n=dimension;x=array(record['x_c'],(n,),'x_c');M=array(record['matrix'],(n,n),'matrix')
 seed=array(initial_seed,(n,),'initial_seed')
 p=record['parameter_c']
 if (type(p)not in (int,float)or not np.isfinite(p)or not -.05<=p<=-.043
  or record['alpha']!=.083 or record['log10T']!=p or record['clock']['forward_steps_per_return']!=2):
  raise ValueError('assigned parameter/clock differs')
 target=dict(label='hostile',period=2,coordinate='log10T',fixed_coordinate='alpha',fixed_value=.083,
  initial_parameter=-.04624653570443442,bounds=[-.05,-.043])
 if not same_json(record['target'],target):raise ValueError('target recipe differs')
 values,left_new,right_new=eig(M,left=True,right=True,check_finite=True)
 index=int(np.argmin(abs(values-1)));critical=values[index];others=np.delete(values,index)
 radius=float(np.max(abs(others)))if others.size else 0.
 isolation=float(np.min(abs(others-critical)))if others.size else None
 new_right_errors=np.linalg.norm(M@right_new-right_new*values[None,:],axis=0)
 new_left_errors=np.linalg.norm(M.T@left_new-left_new*np.conjugate(values)[None,:],axis=0)
 spectrum=record['spectrum'];saved_values=array(spectrum['eigenvalues'],(n,),'eigenvalues',complex_ok=True)
 saved_modes=array(spectrum['eigenvectors'],(n,n),'eigenvectors',complex_ok=True)
 saved_left_values=array(spectrum['left_eigenvalues'],(n,),'left_eigenvalues',complex_ok=True)
 saved_left_modes=array(spectrum['left_eigenvectors'],(n,n),'left_eigenvectors',complex_ok=True)
 right_errors=np.linalg.norm(M@saved_modes-saved_modes*saved_values[None,:],axis=0)
 left_errors=np.linalg.norm(M.T@saved_left_modes-saved_left_modes*saved_left_values[None,:],axis=0)
 rows,cols=linear_sum_assignment(abs(values[:,None]-saved_values[None,:]))
 spectral_difference=float(np.max(abs(values[rows]-saved_values[cols])))
 close('saved right eigenpair residuals',spectrum['eigenpair_residuals'],right_errors)
 saved_critical=complex(spectrum['critical_multiplier']['real'],spectrum['critical_multiplier']['imag'])if isinstance(spectrum['critical_multiplier'],dict)else complex(spectrum['critical_multiplier'])
 close('critical multiplier',saved_critical,critical,atol=1e-10,rtol=0)
 close('critical modulus',spectrum['critical_modulus'],abs(critical),atol=1e-10,rtol=0)
 close('critical argument',spectrum['critical_argument_turns'],np.angle(critical)/(2*np.pi),atol=1e-10,rtol=0)
 close('complementary radius',spectrum['next_modulus'],radius,atol=1e-10,rtol=0)
 rr=array(record['raw_r'],(n,),'raw_r');ll=array(record['raw_l'],(n,),'raw_l')
 r=array(record['r'],(n,),'r');l=array(record['l'],(n,),'l')
 critical_error=max(float(np.linalg.norm(M@rr-rr)),rel(M.T@ll-ll,ll))
 modes_ok=bool(abs(np.linalg.norm(rr)-1)<1e-12 and abs(dot(ll,rr)-1)<1e-12
  and rr[int(np.argmax(abs(rr)))]>0 and critical_error<t['eigen_residual'])
 close('spectrum/raw right',spectrum['right'],rr,atol=0,rtol=0);close('spectrum/raw left',spectrum['left'],ll,atol=0,rtol=0)
 close('left/right pairing',spectrum['left_right_pairing'],dot(ll,rr))
 close('left/right cosine',spectrum['left_right_cosine'],1/np.linalg.norm(ll))
 close('critical eigen residual',spectrum['critical_eigen_residual'],critical_error)
 points=array(record['orbitpoints'],(3,n),'orbitpoints')
 phase=array(record['phase_return_points'],(2,n),'phase_return_points')
 close('initial orbit point',points[0],x,atol=0,rtol=0)
 returns=[rel(y-x,x)for y in points[1:]]
 phase_errors=[rel(y-z,z)for y,z in zip(phase,points[:2])]
 diagnostics=record['orbit_diagnostics']
 close('stored relative returns',diagnostics['relative_returns'],returns)
 close('stored phase closures',record['phase_returns'],phase_errors)
 close('proper divisor1',diagnostics['proper_divisor_returns'].get(1,diagnostics['proper_divisor_returns'].get('1')),returns[0])
 if diagnostics.get('numerically_attracting')is not None:errors.append('neutral fold reported attracting')
 solver=record['solver'];close('solver state',solver['state'],x,atol=0,rtol=0);close('solver matrix',solver['matrix'],M,atol=0,rtol=0)
 close('legacy scalar parameter',solver['alpha'],p,atol=0,rtol=0)
 augmented=float(np.hypot(np.linalg.norm(phase[0]-x)/max(np.linalg.norm(seed),1e-30),critical.real-1))
 close('augmented residual',solver['residual'],augmented,atol=1e-12,rtol=1e-5)
 orbit_gates=dict(closure=returns[-1]<t['closure'],primitive=returns[0]>t['primitive'],
  eigenpairs=bool(max(np.max(right_errors),np.max(left_errors),np.max(new_right_errors),np.max(new_left_errors))<t['eigen_residual']),
  all_phase_closure=max(phase_errors)<t['closure'])
 independent_gates=dict(real_plus_one=bool(abs(critical.imag)<=t['imaginary']and abs(critical.real-1)<=t['critical']),
  complementary=radius<t['complementary_limit'],isolation=isolation is None or isolation>t['isolation'],
  full_spectrum_match=spectral_difference<1e-8,
  saved_basis_normalized=bool(np.max(abs(np.linalg.norm(saved_modes,axis=0)-1))<1e-10 and np.max(abs(np.linalg.norm(saved_left_modes,axis=0)-1))<1e-10),
  gauge=modes_ok,
  augmented=solver.get('converged')is True and augmented<t['augmented'])
 precision=[]
 if len(record['composed_precision'])!=2:raise ValueError('two composed precision directions required')
 probe=np.arange(1,n+1,dtype=np.float64);probe/=np.linalg.norm(probe)
 for row,expected_label,expected_direction in zip(record['composed_precision'],('critical','deterministic'),(rr,probe)):
  if row['direction_label']!=expected_label:raise ValueError('composed direction order differs')
  direction=array(row['direction'],(n,),'composed direction');close('prescribed '+expected_label+' direction',direction,expected_direction,atol=1e-14,rtol=0)
  dense=M@direction;jvp=array(row['jvp'],(n,),'composed JVP');error=rel(jvp-dense,dense)
  close('saved dense action '+expected_label,row['dense_action'],dense)
  close('dense/JVP error '+expected_label,row['dense_jvp_relative_error'],error)
  fd_errors=[]
  if len(row['finite_difference'])!=3:raise ValueError('three composed FD rows required')
  for estimate,hrel in zip(row['finite_difference'],STATE_STEPS):
   if estimate['relative_h']!=hrel:raise ValueError('composed FD ladder differs')
   close('composed h',estimate['h'],hrel*max(np.linalg.norm(x),1.))
   fd=array(estimate['derivative'],(n,),'composed finite difference');fd_error=rel(fd-dense,dense)
   close('composed FD error',estimate['relative_error'],fd_error);fd_errors.append(fd_error)
  passed=bool(error<t['dense_jvp_relative']and fd_errors[-1]<t['composed_fd_relative']and
   (fd_errors[-1]<t['composed_fd_floor']or fd_errors[-1]<t['composed_fd_reduction']*fd_errors[0]))
  if row['passed']is not passed:errors.append('composed row gate differs')
  precision.append(dict(direction=expected_label,dense_jvp_error=error,fd_errors=fd_errors,passed=passed))
 critical_estimate=dot(rr,record['composed_precision'][0]['jvp'])
 critical_jvp_error=abs(critical_estimate-critical.real)
 close('critical JVP multiplier',record['critical_jvp_multiplier'],critical_estimate)
 close('critical JVP discrepancy',record['critical_jvp_multiplier_discrepancy'],critical_jvp_error,atol=1e-10,rtol=0)
 gp=array(record['parameter_derivative'],(n,),'parameter AD');hrr=array(record['directional_second'],(n,),'state AD')
 a=dot(ll,gp);b=.5*dot(ll,hrr);orientation=float(np.sign(b));ab=a*b
 for name,saved,actual in (('raw_a',record['raw_a'],a),('raw_b',record['raw_b'],b),
  ('a',record['a'],orientation*a),('b',record['b'],abs(b)),('ab',record['ab'],ab),
  ('absolute_ab',record['absolute_ab'],abs(ab)),('orientation',record['orientation'],orientation),
  ('ghost side',record['ghost_parameter_direction'],float(np.sign(ab))),
  ('slope',record['asymptotic_inverse_passage_coefficient'],float(np.sqrt(abs(ab))/np.pi)),
  ('U',record['U'],.1/abs(b))):close(name,saved,actual)
 close('oriented r',r,orientation*rr,atol=1e-14,rtol=0);close('oriented l',l,orientation*ll,atol=1e-14,rtol=0)
 fd_tables={}
 for name,field,steps,ad,vector,factor in (('parameter_fd','a',PARAMETER_STEPS,a,gp,1.),('state_fd','b',STATE_STEPS,b,hrr,.5)):
  rows=record[name]
  if len(rows)!=3:raise ValueError('complete coefficient FD ladder required')
  values_fd=[]
  for row,step in zip(rows,steps):
   if row['h'if name=='parameter_fd'else'relative_h']!=step:raise ValueError('coefficient FD ladder differs')
   if name=='state_fd':close('state FD h',row['h'],step*max(np.linalg.norm(x),1.))
   derivative=array(row['derivative'],(n,),name);value=factor*dot(ll,derivative)
   close(name+' contraction',row[field],value);close(name+' vector error',row['vector_relative_error'],rel(derivative-vector,vector))
   values_fd.append(value)
  agreement=[bool(np.sign(value)==np.sign(ad)and abs(value-ad)<=t['coefficient_fd_relative']*max(abs(ad),t['coefficient_floor']))for value in values_fd]
  fd_tables[name]=dict(values=values_fd,relative_discrepancies=[abs(value-ad)/max(abs(ad),t['coefficient_floor'])for value in values_fd],individual_agreement=agreement,passed=all(agreement[-2:]))
 gates=dict(orbit_gates,composed_precision=all(row['passed']for row in precision),
  critical_multiplier_jvp=critical_jvp_error<t['eigen_residual'],
  nonzero_a=abs(a)>t['coefficient_floor'],nonzero_b=abs(b)>t['coefficient_floor'],
  parameter_ad_fd=fd_tables['parameter_fd']['passed'],state_ad_fd=fd_tables['state_fd']['passed'])
 if set(record['gates'])!=set(gates):errors.append('reported gate set differs')
 for name,value in gates.items():
  if record['gates'].get(name)is not bool(value):errors.append('reported gate differs: '+name)
 computed=dict(parameter_c=p,alpha=record['alpha'],active_parameter_units='log10T',forward_steps_per_return=2,
  critical_multiplier=critical,critical_modulus=float(abs(critical)),critical_argument_turns=float(np.angle(critical)/(2*np.pi)),
  complementary_radius=radius,critical_isolation=isolation,spectrum=values,
  recomputed_right_eigen_residuals=new_right_errors,recomputed_left_eigen_residuals=new_left_errors,
  saved_right_eigen_residuals=right_errors,saved_left_eigen_residuals=left_errors,
  maximum_spectrum_difference=spectral_difference,raw_right_norm=float(np.linalg.norm(rr)),left_dot_right=dot(ll,rr),
  critical_eigen_residual=critical_error,relative_returns=returns,phase_closures=phase_errors,
  augmented_residual=augmented,raw_a=a,raw_b=b,a=orientation*a,b=abs(b),ab=ab,absolute_ab=abs(ab),
  ghost_parameter_direction=float(np.sign(ab)),slope=float(np.sqrt(abs(ab))/np.pi),
  composed_precision=precision,finite_difference=fd_tables)
 return plain(dict(scope='Independent saved-array arithmetic and eigenanalysis; no model replay or interval proof.',
  numerical_audit_passed=bool(not errors and all(gates.values())and all(independent_gates.values())),
  certificate_passed=False,recomputed=computed,gates=gates,independent_gates=independent_gates,
  saved_metadata_disagreements=errors))


def audit_attempt(attempt,*,receipt_sha,source_commit,plan_sha,dimension=896):
 provenance,files=inventory(attempt,receipt_sha,source_commit,plan_sha)
 record=load_record(attempt,'fold',files);initial=load_record(attempt,'initial-001',files)
 if not same_json(initial['target'],record['target']):raise ValueError('original initial target differs')
 result=numerical_audit(record,initial['seed'],dimension=dimension)
 result['provenance']=provenance
 result['fold_json_sha256']=files['run/fold.json']['sha256']
 result['fold_array_sha256']=files['run/fold.npz']['sha256']
 return result


def admit_resources():
 """Before numerical imports, enforce the separately approved audit envelope."""
 if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('CPU audit requires nice10')
 if any(os.environ.get(k)!='1'for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
  raise ValueError('CPU1 thread limits must precede imports')
 soft,hard=resource.getrlimit(resource.RLIMIT_AS);cap=AUDIT_ENVELOPE['memory_gib']*1024**3
 if hard!=resource.RLIM_INFINITY and hard<cap:raise ValueError('inherited hard AS below audit envelope')
 # Retain a stricter inherited soft cap instead of silently expanding it.
 admitted_soft=cap if soft==resource.RLIM_INFINITY else min(soft,cap)
 resource.setrlimit(resource.RLIMIT_AS,(admitted_soft,cap))
 return dict(AUDIT_ENVELOPE,actual_soft_address_space_bytes=admitted_soft,actual_hard_address_space_bytes=cap)


def deadline_expired(*_):raise TimeoutError('saved-array audit fixed180-second deadline')


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--receipt-sha256',required=True)
 parser.add_argument('--source-commit',required=True);parser.add_argument('--plan-sha256',required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 if args.attempt.resolve()!=args.attempt or args.attempt.parent!=OWNED:
  raise ValueError('owned original attempt required')
 if args.output.resolve()!=args.output or args.output.parent!=OWNED or args.output==args.attempt:
  raise ValueError('fresh separate owned audit directory required')
 envelope=admit_resources()
 args.output.mkdir(exist_ok=False);started=time.monotonic()
 old_alarm=signal.signal(signal.SIGALRM,deadline_expired)
 signal.setitimer(signal.ITIMER_REAL,AUDIT_ENVELOPE['timeout_seconds'])
 report=dict(numerical_audit_passed=False,certificate_passed=False,model_replayed=False)
 try:
  report=audit_attempt(args.attempt,receipt_sha=args.receipt_sha256,source_commit=args.source_commit,plan_sha=args.plan_sha256)
  report['model_replayed']=False
 except BaseException as error:
  report.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
  report['audit_envelope']=envelope
  report.update(audit_source_sha256=sha(__file__),seconds=time.monotonic()-started,
   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (args.output/'audit.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k]for k in ('numerical_audit_passed','certificate_passed','seconds','peak_rss_kib')}))
 return 0 if report['numerical_audit_passed']else 2


if __name__=='__main__':sys.exit(main())
