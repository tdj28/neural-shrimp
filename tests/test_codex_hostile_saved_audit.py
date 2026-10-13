"""Real saved-format synthetic fold audits; no target checkpoint or model replay."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'codex/hostile-fold-crosscheck'


def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


A=module('task7_saved_audit_test',TASK/'saved_audit.py')
W=module('task7_saved_audit_worker',TASK/'fold_worker.py')
C=module('task7_saved_audit_synthetic_chart',ROOT/'tests/test_codex_hostile_fold_worker.py')


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def reseal(attempt):
 p=attempt/'receipt.json';receipt=json.loads(p.read_text())if p.exists()else{}
 receipt['artifacts']=[dict(path=str(f.relative_to(attempt)),bytes=f.stat().st_size,sha256=A.sha(f))
  for f in sorted(attempt.rglob('*'))if f.is_file()and f!=p]
 write(p,receipt);return A.sha(p)


@pytest.fixture
def record(tmp_path):
 setup,_=W.dependencies();seed=np.array([3.000005,1e-7]);events=[]
 result=W.qualify_fold(C.Nonnormal(W.target(),b=-3.),seed,chunk=128,max_newton=4,max_backtracks=2,
  save=lambda n,p:events.append((n,p)),check=lambda:None)
 assert result['passed']
 setup.persist(tmp_path,'fold',result)
 files={'run/fold.npz':dict(sha256=A.sha(tmp_path/'fold.npz'))}
 # Load through the exact saved NPZ descriptor decoder (not in-memory shortcuts).
 run=tmp_path/'attempt/run';run.mkdir(parents=True)
 shutil.copyfile(tmp_path/'fold.json',run/'fold.json');shutil.copyfile(tmp_path/'fold.npz',run/'fold.npz')
 decoded=A.load_record(run.parent,'fold',files)
 return decoded,seed,result,events,setup


def test_independent_full_spectrum_contractions_gauge_and_phase_checks(record):
 decoded,seed,_,_,_=record
 result=A.numerical_audit(decoded,seed,dimension=2)
 assert result['numerical_audit_passed'],result
 assert not result['certificate_passed']
 assert all(result['gates'].values())and all(result['independent_gates'].values())
 assert result['recomputed']['ghost_parameter_direction']==-1
 assert result['recomputed']['a']<0<result['recomputed']['b']
 assert result['recomputed']['left_dot_right']==pytest.approx(1.)
 assert len(result['recomputed']['spectrum'])==2
 assert len(result['recomputed']['phase_closures'])==2
 assert len(result['recomputed']['composed_precision'])==2
 assert result['recomputed']['finite_difference']['parameter_fd']['individual_agreement']==[True]*3


@pytest.mark.parametrize('fault',['phase1','primitive','leftgauge','raw_a','jvp','fd','stored_gate','mode_zero'])
def test_resealed_scientific_inconsistency_never_passes(record,fault):
 decoded,seed,_,_,_=record;x=copy.deepcopy(decoded)
 if fault=='phase1':x['phase_return_points'][1,0]+=.01
 if fault=='primitive':x['orbitpoints'][1]=x['orbitpoints'][0]
 if fault=='leftgauge':x['raw_l']*=2
 if fault=='raw_a':x['raw_a']+=.1
 if fault=='jvp':x['composed_precision'][0]['jvp'][0]+=.02
 if fault=='fd':
  for row,sign in zip(x['parameter_fd'][-2:],[1.,-1.]):row['derivative']*=1+sign*.02
 if fault=='stored_gate':x['gates']['primitive']=False
 if fault=='mode_zero':x['spectrum']['left_eigenvectors'][:,1]=0
 result=A.numerical_audit(x,seed,dimension=2)
 assert not result['numerical_audit_passed']


@pytest.mark.parametrize('fault',['missing_phase','nonfinite','complex_state','short_ladder','wrong_coordinate'])
def test_unsupported_or_incomplete_arrays_refuse(record,fault):
 decoded,seed,_,_,_=record;x=copy.deepcopy(decoded)
 if fault=='missing_phase':x['phase_return_points']=x['phase_return_points'][:1]
 if fault=='nonfinite':x['parameter_derivative'][0]=np.nan
 if fault=='complex_state':x['x_c']=x['x_c'].astype(np.complex128)
 if fault=='short_ladder':x['state_fd'].pop()
 if fault=='wrong_coordinate':x['active_parameter_units']='temperature'
 with pytest.raises(ValueError):A.numerical_audit(x,seed,dimension=2)


def complete_attempt(tmp_path,record):
 _,seed,result,events,setup=record
 attempt=tmp_path/'complete';run=attempt/'run';run.mkdir(parents=True)
 for name,payload in events:setup.persist(run,name,payload)
 setup.persist(run,'fold',result)
 plan=json.loads((TASK/'fold-001-plan.json').read_text())
 write(run/'original-plan.json',plan);plan_sha=A.sha(run/'original-plan.json')
 # The serializer above matches the committed pretty-printed source bytes.
 assert plan_sha==A.sha(TASK/'fold-001-plan.json')
 source_commit='b'*40
 extra=('codex/certify-llm/derivative_campaign.py','codex/hostile-fold-crosscheck/launch.py',
  'codex/hostile-fold-crosscheck/fold-001-plan.json','codex/hostile-fold-crosscheck/fold-001-run-card.md')
 source=[]
 for name in (*A.EXPECTED_SOURCES,*extra):
  path=attempt/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
  source.append(dict(path=name,bytes=path.stat().st_size,sha256=A.sha(path)))
 write(attempt/'manifest.json',dict(source_commit=source_commit,plan_sha256=plan_sha,source=source))
 backend=dict(synthetic_only=True)
 write(run/'resource-admission.json',dict(profile_receipt_sha256=plan['profile_authority']['receipt_sha256'],
  historical_receipt_sha256=plan['historical_memory_authority']['receipt_sha256'],profile=dict(backend=backend)))
 vector_path='/synthetic/disclosed-hostility-vector.npz'
 write(run/'inputs.json',dict(source_commit=source_commit,plan_sha256=plan_sha,plan=plan,
  certificate_passed=False,passage_trajectories_evaluated=False,producer_seed_only=True,seed_index=0,
  vector_key='hostility-vs-kindness',vector_path=vector_path,
  steering_artifact=dict(path=vector_path,key='hostility-vs-kindness',sha256=plan['vector_sha256']),
  checkpoint_assets=A.CHECKPOINT_ASSETS,prefix_ids=A.PREFIX_IDS,backend=backend,
  resource_admission_sha256=A.sha(run/'resource-admission.json')))
 write(run/'status.json',dict(passed=True,computational_completed=True,solver_history_complete=True,
  certificate_passed=False,resource_probe_only=False,passage_trajectories_evaluated=False,status='fold_qualified',stage='qualified'))
 write(attempt/'receipt.json',dict(source_commit=source_commit,plan_sha256=plan_sha,exit_code=0,timed_out=False,
  status='fold_attempt_completed',worker_termination_verified=True,uncertain_ownership=[],certificate_passed=False,
  resource_probe_only=False,scientific_fold_result=True))
 receipt_sha=reseal(attempt)
 return attempt,dict(receipt_sha=receipt_sha,source_commit=source_commit,plan_sha=plan_sha)


def test_real_complete_inventory_source_descriptor_and_numerical_path(tmp_path,record):
 attempt,pins=complete_attempt(tmp_path,record)
 report=A.audit_attempt(attempt,**pins,dimension=2)
 assert report['numerical_audit_passed']
 assert report['provenance']['captured_source_count']==11
 assert not report['provenance']['git_commit_blob_verification_performed']
 assert report['provenance']['artifact_count']>30


@pytest.mark.parametrize('fault',['failed','timeout','unreaped','plan','source','descriptor','vector','prefix','extra','receipt_pin'])
def test_resealed_provenance_corruption_refuses(tmp_path,record,fault):
 attempt,pins=complete_attempt(tmp_path,record)
 if fault in ('failed','timeout','unreaped'):
  path=attempt/'receipt.json';d=json.loads(path.read_text())
  d[{'failed':'exit_code','timeout':'timed_out','unreaped':'worker_termination_verified'}[fault]]={'failed':2,'timeout':True,'unreaped':False}[fault]
  write(path,d)
 if fault=='plan':
  path=attempt/'run/original-plan.json';d=json.loads(path.read_text());d['target']['alpha']=.1;write(path,d)
 if fault=='source':(attempt/'source/codex/crosscheck/qwen_map.py').write_text('foreign source')
 if fault=='descriptor':
  path=attempt/'run/fold.json';d=json.loads(path.read_text());d['matrix']['shape']=[1,1];write(path,d)
 if fault in ('vector','prefix'):
  path=attempt/'run/inputs.json';d=json.loads(path.read_text())
  if fault=='vector':d['steering_artifact']['sha256']='a'*64
  if fault=='prefix':d['prefix_ids']=d['prefix_ids'][:-1]
  write(path,d)
 pins['receipt_sha']=reseal(attempt)
 if fault=='extra':(attempt/'extra.txt').write_text('not inventoried')
 if fault=='receipt_pin':pins['receipt_sha']='c'*64
 with pytest.raises(ValueError):A.audit_attempt(attempt,**pins,dimension=2)


@pytest.mark.parametrize('soft,hard,expected',[
 (-1,-1,4*1024**3),(2*1024**3,8*1024**3,2*1024**3),
 (8*1024**3,8*1024**3,4*1024**3)])
def test_audit_resource_cap_preserves_stricter_soft(monkeypatch,soft,hard,expected):
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
 monkeypatch.setattr(A.os,'getpriority',lambda *_:10)
 monkeypatch.setattr(A.resource,'getrlimit',lambda _:(soft,hard))
 calls=[];monkeypatch.setattr(A.resource,'setrlimit',lambda kind,limits:calls.append((kind,limits)))
 admitted=A.admit_resources()
 assert calls==[(A.resource.RLIMIT_AS,(expected,4*1024**3))]
 assert admitted['actual_soft_address_space_bytes']==expected
 assert admitted['timeout_seconds']==180


@pytest.mark.parametrize('fault',['nice','thread','hard'])
def test_audit_resource_refusals_precede_math(monkeypatch,fault):
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
 monkeypatch.setattr(A.os,'getpriority',lambda *_:0 if fault=='nice' else 10)
 if fault=='thread':monkeypatch.setenv('OPENBLAS_NUM_THREADS','2')
 monkeypatch.setattr(A.resource,'getrlimit',lambda _:(1024**3,1024**3)if fault=='hard'else(-1,-1))
 monkeypatch.setattr(A.resource,'setrlimit',lambda *_:pytest.fail('must refuse before changing caps'))
 with pytest.raises(ValueError):A.admit_resources()


def test_audit_deadline_fails_loudly():
 with pytest.raises(TimeoutError,match='180-second'):A.deadline_expired()
