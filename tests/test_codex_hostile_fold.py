"""Task7 synthetic controls only: no pretrained checkpoint or target run."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'codex/hostile-fold-crosscheck'

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
A=module('task7_assets_test',TASK/'assets.py')
P=module('task7_pilot_test',TASK/'resource_pilot.py')
Q=module('task7_qwen_synthetic_helpers',ROOT/'tests/test_codex_qwen_map.py')

@pytest.fixture
def tiny():return Q.tiny.__wrapped__()

def plan():return dict(schema='codex-hostile-fold-profile-v1',envelope=dict(P.ENVELOPE),source_sha256={s:'a'*64 for s in P.SOURCES},
 seed_sha256=A.SEED_SHA256,vector_sha256=A.VECTOR_SHA256,batch_directions=16,purpose='synthetic resource profile')

def test_fixed_provisional_resource_contract():
 assert P.validate_plan(plan())['envelope']['minimum_start_available_gib']==28
 assert P.ENVELOPE['minimum_runtime_available_gib']==16

@pytest.mark.parametrize('alter',[lambda p:p['envelope'].update(cpu_threads=True),lambda p:p['envelope'].update(memory_gib=13),
 lambda p:p['envelope'].update(minimum_runtime_available_gib=12),lambda p:p.update(batch_directions=8),
 lambda p:p['source_sha256'].pop('codex/crosscheck/qwen_map.py'),lambda p:p.update(seed_sha256='bad')])
def test_refuse_scope_or_resource_changes(alter):
 p=plan();alter(p)
 with pytest.raises(ValueError):P.validate_plan(p)


def asset_fixture(tmp_path,monkeypatch):
 seed=tmp_path/'states.npy';vector=tmp_path/'vectors.npz';np.save(seed,np.arange(3*896,dtype=np.float64).reshape(3,896))
 np.savez(vector,**{A.VECTOR_KEY:np.arange(896,dtype=np.float64),'philosophy':np.ones(896)})
 monkeypatch.setattr(A,'SEED',seed);monkeypatch.setattr(A,'VECTOR',vector)
 monkeypatch.setattr(A,'SEED_SHA256',A.sha(seed));monkeypatch.setattr(A,'VECTOR_SHA256',A.sha(vector))
 return seed,vector

def test_explicit_hostility_key_and_first_seed_row(tmp_path,monkeypatch):
 s,v=asset_fixture(tmp_path,monkeypatch);x,w=A.seed_and_vector()
 np.testing.assert_array_equal(x,np.arange(896,dtype=np.float64));np.testing.assert_array_equal(w,x)
 x[0]=-5;np.testing.assert_array_equal(np.load(s)[0],np.arange(896))
 v.write_bytes(b'changed')
 with pytest.raises(ValueError,match='hash'):A.seed_and_vector()

@pytest.mark.parametrize('bad',['dtype','shape','nonfinite'])
def test_resealed_invalid_vector_is_refused(tmp_path,monkeypatch,bad):
 _,v=asset_fixture(tmp_path,monkeypatch);a=np.zeros(896,dtype=np.float64)
 if bad=='dtype':a=a.astype(np.float32)
 if bad=='shape':a=a[:-1]
 if bad=='nonfinite':a[10]=np.nan
 np.savez(v,**{A.VECTOR_KEY:a});monkeypatch.setattr(A,'VECTOR_SHA256',A.sha(v))
 with pytest.raises(ValueError,match='full finite'):A.seed_and_vector()


def test_reused_operations_differentiate_log10_temperature_and_full_composed_state(tiny):
 s=P.setup_module();target=s.Target('hostile',2,'log10T','alpha',A.ALPHA,A.PARAMETER,(-.05,-.043));ops=s.Operations(tiny,target,chunk=2)
 x=tiny.initial_state().numpy();p=A.PARAMETER;r=np.arange(1,tiny.d+1,dtype=np.float64);r/=np.linalg.norm(r)
 jac=ops.dense(x,p);j= tiny.jvp(s.tensor(x),s.tensor(r),A.ALPHA,p,2,ops.fixed(p)[2]).numpy()
 np.testing.assert_allclose(jac@r,j,rtol=2e-12,atol=2e-13)
 gp=ops.parameter_derivative(x,p);h=2e-5
 np.testing.assert_allclose(gp,(ops.period_map(x,p+h)-ops.period_map(x,p-h))/(2*h),rtol=3e-6,atol=2e-10)
 hrr=ops.second(x,r,p)
 fun=lambda y:tiny.jvp(s.tensor(y),s.tensor(r),A.ALPHA,p,2,ops.fixed(p)[2]).numpy()
 np.testing.assert_allclose(hrr,(fun(x+h*r)-fun(x-h*r))/(2*h),rtol=5e-6,atol=2e-9)
 gx,gp2=ops.critical_grad(x,p,r,r)
 def scalar(y,t):return r@tiny.jvp(s.tensor(y),s.tensor(r),A.ALPHA,t,2).numpy()
 np.testing.assert_allclose(gx@r,(scalar(x+h*r,p)-scalar(x-h*r,p))/(2*h),rtol=5e-6,atol=2e-9)
 np.testing.assert_allclose(gp2,(scalar(x,p+h)-scalar(x,p-h))/(2*h),rtol=5e-6,atol=2e-9)


def test_profile_retains_outputs_and_never_calls_them_a_fold(tiny,monkeypatch):
 monkeypatch.setitem(sys.modules,'assets',A);events=[];checks=[]
 result=P.profile(tiny,tiny.initial_state().numpy(),lambda n,v:events.append((n,v)),lambda:checks.append(True),batch=2)
 assert len(result['rows'])==6 and not result['fold_qualified'] and not result['certificate_passed']
 assert not result['normal_form_coefficients_computed'] and checks
 assert {r['operation']for r in result['rows']}=={'period-map','composed-jvp','batched-composed-jvp','parameter-ad','nested-state-ad','critical-gradient-ad'}
 assert next(v for n,v in events if n=='batched-composed-jvp')['output'].shape==(2,tiny.d)


def test_profile_stops_on_resource_guard_without_model_call():
 class Never:
  d=8
  def __getattr__(self,k):raise AssertionError('model touched: '+k)
 def stop():raise RuntimeError('headroom floor')
 with pytest.raises(RuntimeError,match='headroom'):P.profile(Never(),np.zeros(8),lambda *a:None,stop,batch=2)
