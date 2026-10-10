import numpy as np
import pytest
from neural_shrimp.unit_pair_section import section_evaluation,native_association

class Normal:
 n=3
 def cusp_step(self,z,h,o):
  x,y,w=z;return[x+y,y+h-1+x*x*.5+(o-1)*y+2*x*y,w*.2]


def test_all_section_columns_and_actual_hidden_derivative():
 u=np.array([.03,-.01,.02,.1,.97,.8,.2,.1,.1,.7,-.2]);hidden=1.001
 v,J,fh,data,z,t=section_evaluation(Normal(),u,hidden,2)
 assert t==u[3]and z[3]==hidden
 eps=1e-6
 for k in range(len(u)):
  du=np.eye(len(u))[k]*eps
  fd=(section_evaluation(Normal(),u+du,hidden,2)[0]-section_evaluation(Normal(),u-du,hidden,2)[0])/(2*eps)
  np.testing.assert_allclose(J[:,k],fd,atol=1e-8,rtol=1e-6)
 fd=(section_evaluation(Normal(),u,hidden+eps,2)[0]-section_evaluation(Normal(),u,hidden-eps,2)[0])/(2*eps)
 np.testing.assert_allclose(fh,fd,atol=1e-8,rtol=1e-6)

def test_reverse_native_association_checks_all704_without_phase_transform():
 z=np.arange(704,dtype=float)/100
 assert native_association(z,z)['passed']
 other=z.copy();other[455]+=1e-4;assert not native_association(other,z)['passed']
 with pytest.raises(ValueError):native_association(z[:-1],z)


def test_complete_reverse_prefix_and_every_section_join(monkeypatch):
 import neural_shrimp.unit_pair_section as U
 class Fake:n=13
 def evaluation(model,z,period,t,tick):
  target=np.zeros(41);target[13]=t;target[14]=1;target[15]=1;target[29]=1
  ht=np.zeros(41);ht[13]=-1
  data={'orbit':np.ones((19,13)),'q_phases':np.ones((19,13)),'r_phases':np.ones((19,13)),'monodromy':np.eye(13),'z':np.array(z)}
  return z-target,np.eye(41),ht,data
 monkeypatch.setattr(U,'evaluate',evaluation);monkeypatch.setattr(U,'qualify',lambda *a,**k:{'passed':True,'complement_unit_gap':.4})
 monkeypatch.setattr(U,'native_flat',lambda z,data:np.r_[z,np.zeros(663)])
 points=[]
 for i,t in enumerate(np.linspace(.1,.3,20)):
  z=np.zeros(41);z[13]=t;z[14]=1;z[15]=1;z[29]=1;points.append({'index':i,'t':float(t),'z':z.tolist(),'native_flat':np.r_[z,np.zeros(663)].tolist()})
 events=[];result=U.survey(Fake(),points,max_points=3,hidden_step=.01,observe=lambda k,v:events.append((k,v)))
 assert result['status']=='point_cap'and len(result['reverse_prefix'])==19 and len(result['points'])==3
 assert all(v['association']['passed']for v in result['reverse_prefix'])
 attempts=[v for k,v in events if k=='attempt'];assert len(attempts)==2 and all(v['passed']and v['reverse']['passed']for v in attempts)
 assert result['points'][-1]['t']==pytest.approx(.32)
