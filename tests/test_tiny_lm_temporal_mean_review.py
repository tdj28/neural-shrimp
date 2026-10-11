import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import review_tiny_lm_temporal_mean as R

def test_independent_mean_cancelling_and_wrong_update():
 v=np.array([.3,-.7]);a=np.array([[(-1)**i*(i+1)*.01,.02*i]for i in range(16)]);eta=.13
 correct=v-(eta/16)*np.sum(a,axis=0)
 assert R.mean_step(v,a,eta,correct)<=1
 with pytest.raises(ValueError,match='rounding'):R.mean_step(v,a,eta,correct+1e-6)

def test_noncommuting_mean_is_not_ordered():
 A=np.array([[.8,.1],[.1,.6]]);B=np.array([[.7,-.2],[-.2,.9]]);u=np.array([1.,0.]);eta=1.
 actions=np.array([(np.eye(2)-m)@u for m in [A,B]*8]);mean=(A+B)@u/2
 assert R.mean_step(u,actions,eta,mean)<=1
 with pytest.raises(ValueError):R.mean_step(u,actions,eta,B@A@u)

def test_metrics_analytic_response_and_ce():
 u=np.array([1.,0.]);trace=np.repeat(u[None,:],17,axis=0);branches=[]
 for e in (.001,.0005):branches.append(dict(plus=e*trace,minus=-e*trace,ce_plus={h:2*e for h in R.H},ce_minus={h:-2*e for h in R.H}))
 rows=R.metrics(dict(branches=branches,eps=[.001,.0005],cegrad=np.repeat(np.array([[2.,0.]]),3,axis=0)),dict(mean=trace,ordered=trace,frozen=trace))
 assert all(r['mechanism_qualified']and r['training_ce_slope_resolved']and r['comparisons']['mean']['vector_error']==0 for r in rows)

def test_input_return_identity_rejects_wrong_state(monkeypatch,tmp_path):
 x=np.zeros(15776);other=x.copy();other[0]=1
 a=dict(sequence=1,index=1,operation='gradient',case='x',state={'id':'wrong'})
 b=dict(a,returned={'id':'ret'},seconds=.01)
 monkeypatch.setattr(R,'read',lambda p:b if str(p).endswith('-return.json')else a)
 ev=R.Events(tmp_path,lambda:None);monkeypatch.setattr(ev,'obj',lambda p:other if p['id']=='wrong'else x)
 with pytest.raises(ValueError,match='actual state'):ev.call('x','gradient',x)

def test_real_metadata_never_opens_arrays(monkeypatch):
 monkeypatch.setattr(R.np,'load',lambda *a,**k:(_ for _ in()).throw(AssertionError('arrays opened')))
 inv,out=R.metadata(lambda:None)
 assert len(inv)==9 and out['event_pairs']==1308
