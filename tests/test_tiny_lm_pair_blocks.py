import numpy as np
import pytest
from neural_shrimp import tiny_lm_pair_blocks as B
from neural_shrimp import tiny_lm_parity_order as H

def dense(ms,u,order):
 rows=[u.copy()]
 for i in order:rows.append(ms[i]@rows[-1])
 return np.asarray(rows)

def test_same_noncommuting_pair_has_zero_block_contrast_but_nonzero_parity():
 a=np.diag([.7,1.2]);b=np.array([[.95,.15],[.15,.85]]);ms=np.array([a,b]*8);u=np.array([.6,.8]);v={k:dense(ms,u,o)[-1]for k,o in dict(H.ORDERS,**B.ORDERS).items()};parity=(v['Q']+v['T']-v['P']-v['R'])/2
 assert np.linalg.norm(a@b-b@a)>1e-2 and np.linalg.norm(parity)>1e-3
 report=B.compare(u,v['P'],v['R'],v['U'],v['V'],u,u,parity)
 assert report['qualified']and report['even_norm']<1e-13 and set(report['products'])=={'P','R','U','V'}

def test_actual_orders_and_full_retention():
 ms=np.array([np.array([[.6+.01*k,.02],[.02,1.1-.01*k]])for k in range(16)]);states=np.arange(34,dtype=float).reshape(17,2);u=np.array([.6,.8]);saved=[]
 for name in B.ORDERS:
  calls=[]
  def hvp(s,v):k=int(s[0]/2);calls.append(k);return (v-ms[k]@v)/.2
  result=B.propagate(states,u,.2,name,hvp,lambda t,ix:saved.append((t,ix)),lambda:None)
  assert np.allclose(result,dense(ms,u,B.ORDERS[name]),rtol=1e-14,atol=1e-15)and calls==list(B.ORDERS[name])and saved[-1][1]==calls
 assert B.U==(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1)and B.V==B.U[::-1]
 assert 4*sum(len(v)for v in B.ORDERS.values())==128

def test_all_four_qualification_and_separate_descriptive_outcomes():
 cases=[dict(seed=s,rate_role=r,qualified=True,comparison=dict(even_relative_ordered=3,even_relative_parity_even=.05))for s in(28161,28162)for r in('original','quarter_step')]
 d=B.descriptive(cases);assert d['material_in_both_originals']and d['tenfold_attenuation_in_both_originals']
 cases[1]['qualified']=False;assert not any(B.descriptive(cases).values())
 with pytest.raises(ValueError):B.descriptive(cases[:-1])

def test_failed_action_preserves_partial_trace():
 saved=[];n=[0]
 def hvp(*_):
  n[0]+=1;return np.array([np.nan,0])if n[0]==3 else np.ones(2)
 with pytest.raises(ValueError):B.propagate(np.ones((17,2)),np.ones(2),.1,'U',hvp,lambda t,i:saved.append((t,i)),lambda:None)
 assert saved[0][0].shape==(3,2)and saved[0][1]==[14,15]

def test_cold_cost_and_all_vector_copies():
 n=15776*8+128;f=B.forecast([.1,.2],[n,n],.570807749)
 assert f['seconds']==2*128*.570807749+30 and f['vectors']==356 and f['bytes']==2*356*n+64*1024**2
 assert f['seconds']<240 and f['bytes']<192*1024**2
 with pytest.raises(ValueError):B.forecast([.1,.2],[n,n-1],.5)
