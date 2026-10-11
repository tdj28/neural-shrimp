import sys
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural_shrimp.signed_flip_output_scout import evaluate,flat,qualify
class Nonlinear:
 n=1
 def cusp_step(self,x,h,o):return[-.3*x[0]+.02*x[0]*x[0]*x[0]+.01*h*x[0]+.02*o]
def test_all_columns_and_actual_output_partial():
 z=np.array([.3,.4,1.]);o=.6;v,J,do,d=evaluate(Nonlinear(),z,o,18);eps=1e-5
 for k in range(3):
  e=np.eye(3)[k]*eps;a=evaluate(Nonlinear(),z+e,o,18)[0];b=evaluate(Nonlinear(),z-e,o,18)[0];assert max(abs((a-b)/(2*eps)-J[:,k]))<1e-9
 a=evaluate(Nonlinear(),z,o+eps,18)[0];b=evaluate(Nonlinear(),z,o-eps,18)[0];assert max(abs((a-b)/(2*eps)-do))<1e-9
 q=flat(z,o,d);assert q[-2]==z[1]and q[-1]==o

def test_output_survey_retains_fixed_output_and_full470_reverse(monkeypatch):
 import neural_shrimp.signed_flip_output_scout as S
 def synthetic(model,z,o,period,tick,observe):
  ideal=np.r_[o,np.zeros(12),80.,1.,np.zeros(12)];v=np.asarray(z)-ideal;J=np.eye(27);do=np.r_[-1.,np.zeros(26)];x=np.asarray(z[:13]);orbit=np.repeat(x[None,:],19,axis=0);orbit[1:18,1]=np.arange(1,18)*.001;mode=np.repeat(np.asarray(z[14:])[None,:],19,axis=0);M=np.diag([-1.]+[.5]*12);tick();return v,J,do,dict(orbit=orbit,modes=mode,monodromy=M,native_returns=[])
 monkeypatch.setattr(S,'evaluate',synthetic);events=[];model=type('M',(),{'n':13})();seed=np.r_[.8,np.zeros(12),80.,1.,np.zeros(12)]
 r=S.survey(model,seed,.8,max_points=3,observe=lambda k,v:events.append((k,v)))
 assert len(r['points'])==3 and r['evaluations']==5 and all(p['hidden']==80. and p['native_flat'][-1]==p['output']for p in r['points'])
 joins=[v for k,v in events if k=='continuation_attempt'];assert len(joins)==2 and all(v['passed']and v['reverse_coordinates']==470 for v in joins)
 with pytest.raises(RuntimeError,match='return cap'):S.survey(model,seed,.8,max_evaluations=1)
