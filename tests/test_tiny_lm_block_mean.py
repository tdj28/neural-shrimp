import numpy as np
import pytest
from neural_shrimp import tiny_lm_block_mean as H

def test_mean_of_nonsymmetric_products_dense_oracle():
 rng=np.random.default_rng(42);hs=rng.normal(size=(16,3,3));hs=(hs+hs.transpose(0,2,1))/2;eta=.12;states=np.zeros((17,3));states[:,0]=np.arange(17);u=np.array([1.,-.2,.4]);seen=[];saved={}
 def hvp(state,v,j,k,phase):seen.append((j,k,phase,v.copy()));return hs[int(state[0])]@v
 trace=H.power(states,u,eta,hvp,lambda n,v:saved.__setitem__(n,v),lambda:None)
 js=np.eye(3)-eta*hs;mean=sum(js[2*k+1]@js[2*k]for k in range(8))/8
 assert np.linalg.norm(mean-mean.T)>1e-4
 assert np.allclose(trace[-1],np.linalg.matrix_power(mean,8)@u,rtol=2e-14,atol=2e-14)
 assert len(seen)==128 and len(saved)==9
 for j in range(8):
  assert all(np.array_equal(seen[j*16+2*k][3],trace[j])for k in range(8))
  for k in range(8):assert np.array_equal(seen[j*16+2*k+1][3],trace[j]-eta*(hs[2*k]@trace[j]))

def test_commuting_variation_is_not_ordering_evidence():
 states=np.arange(17.)[:,None];u=np.ones(1);hs=np.linspace(.1,1.,16);eta=.5
 trace=H.power(states,u,eta,lambda s,v,*_:hs[int(s[0])]*v,lambda *a:None,lambda:None)
 j=1-eta*hs;assert not np.isclose(trace[-1][0],np.prod(j),rtol=.1)
 assert np.isclose(trace[-1][0],np.mean(j[::2]*j[1::2])**8)

def test_incomplete_and_nonfinite_actions_keep_completed_path():
 saves={};count=[0]
 def hvp(*a):
  count[0]+=1
  if count[0]==4:raise RuntimeError('failed native')
  return np.zeros(2)
 with pytest.raises(RuntimeError):H.power(np.zeros((17,2)),np.ones(2),.1,hvp,lambda n,v:saves.__setitem__(n,v),lambda:None)
 assert saves['block-mean-powers'].shape==(1,2)and saves['power-0-block-ends'].shape==(1,2)

def test_forecast_cold_floor_and_full_vector_storage():
 r=H.forecast([.1,.2],[126336]*2,.63,10000);assert r['seconds']==2*512*.63+30 and r['bytes']==2*1280*126336+64*1024**2+10000

def test_candidate_error_does_not_gate_response_qualification():
 old=dict(ordered=np.ones(2),mean=np.ones(2)*3,frozen=np.ones(2)*100,direction=np.ones(2),ce_gradient=np.ones(2))
 out=H.compare(np.ones(2)*1000,old,[np.ones(2)]*2,[2.,2.]);assert out['qualified']and out['comparisons']['block_mean']['vector_error']>900
 old['ordered']=np.ones(2)*2;assert not H.compare(np.ones(2),old,[np.ones(2)]*2,[2.,2.])['qualified']
