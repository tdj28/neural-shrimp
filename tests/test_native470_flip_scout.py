import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp.signed_flip_scout import evaluate
class Linear:
 n=1
 def cusp_step(self,x,h,o):return[(.2+.1*h+.2*o)*x[0]]
def test_full_section_derivative_and_direction_columns():
 z=np.array([.4,.7,1.]);h=.5;p=3;a=.2+.1*h+.2*z[1];v,J,dh,d=evaluate(Linear(),z,h,p)
 E=np.array([[a**p-1,p*a**(p-1)*.2*z[0],0],[0,p*a**(p-1)*.2*z[2],a**p+1],[0,0,z[2]]]);assert np.max(abs(J-E))<1e-13
 assert np.max(abs(dh-[p*a**(p-1)*.1*z[0],p*a**(p-1)*.1*z[2],0]))<1e-13
 assert len(d['orbit'])==p+1 and len(d['native_returns'])==p

def test_native_input_retained_before_failure():
 class Broken:
  n=1
  def cusp_step(self,*a):raise ValueError('failed phase')
 import pytest
 events=[]
 with pytest.raises(ValueError,match='failed phase'):evaluate(Broken(),[0,.5,1],.2,18,observe=lambda k,v:events.append((k,v)))
 assert [x[0]for x in events]==['native_input']and events[0][1]['phase']==0

def test_nonlinear_native_jacobian_and_actual_hidden_column():
 class Nonlinear:
  n=1
  def cusp_step(self,x,h,o):return[-.3*x[0]+.02*x[0]*x[0]*x[0]+.01*h*x[0]+.02*o]
 z=np.array([.3,.6,1.]);h=.4;v,J,dh,_=evaluate(Nonlinear(),z,h,18);eps=1e-5
 for k in range(3):
  e=np.eye(3)[k]*eps;a=evaluate(Nonlinear(),z+e,h,18)[0];b=evaluate(Nonlinear(),z-e,h,18)[0];assert np.max(abs((a-b)/(2*eps)-J[:,k]))<1e-9
 a=evaluate(Nonlinear(),z,h+eps,18)[0];b=evaluate(Nonlinear(),z,h-eps,18)[0];assert np.max(abs((a-b)/(2*eps)-dh))<1e-9

def test_full470_curved_association_strict_all_coordinates_and_gauge():
 import copy,pytest
 import scout_native470_flip as D
 def b(x):return {'mid':[x,0],'rad':[0,0]}
 c=dict(center=[b(0)]*470,slope=[b(0)]*470,curvature=[b(0)]*470,normal=['0']*468+['1','0'],origin=['0']*470,parameter_center='0',settings={'halfwidth_exponent':0},coordinate_radii=[b(1)]*470)
 c['slope'][468]=b(1);c['curvature'][7]=b(2);z=np.zeros(470);z[468]=.5;z[7]=.25
 assert D.association(z,c)['passed'];z[7]=1.25;assert not D.association(z,c)['passed'];z[7]=1.;z[468]=1.;assert not D.association(z,c)['parameter_inside']
 c=copy.deepcopy(c);c['curvature'][0]['rad']=[1,-80]
 with pytest.raises(ValueError,match='exact saved'):D.association(z,c)

def synthetic_section(z,h,period,tick,observe):
 n=13;ideal=np.r_[h,np.zeros(12),.8,1.,np.zeros(12)];v=np.asarray(z)-ideal;J=np.eye(27);dh=np.r_[-1.,np.zeros(26)];x=np.asarray(z[:13]);orbit=np.repeat(x[None,:],19,axis=0);orbit[1:18,1]=np.arange(1,18)*.001;mode=np.repeat(np.asarray(z[14:])[None,:],19,axis=0);M=np.diag([-1.]+[.5]*12)
 tick();return v,J,dh,dict(orbit=orbit,modes=mode,monodromy=M,native_returns=[])

def test_survey_retains_full_returns_and470_reverse_without_extra_advance(monkeypatch):
 import neural_shrimp.signed_flip_scout as S
 monkeypatch.setattr(S,'evaluate',lambda model,*a:synthetic_section(*a));events=[];model=type('M',(),{'n':13})();seed=np.r_[80.,np.zeros(12),.8,1.,np.zeros(12)]
 ans=S.survey(model,seed,80.,max_points=3,observe=lambda k,v:events.append((k,v)))
 assert ans['status']=='point_cap'and len(ans['points'])==3 and ans['evaluations']==5
 attempts=[r for k,r in events if k=='continuation_attempt'];assert len(attempts)==2 and all(r['passed']and r['reverse_coordinates']==470 and r['reverse_error']==0 for r in attempts)
 assert len([r for k,r in events if k=='evaluation_return'])==5 and len([r for k,r in events if k=='tangent_proposal'])==2

def test_survey_caps_and_resource_failures_never_retry(monkeypatch):
 import pytest
 import neural_shrimp.signed_flip_scout as S
 monkeypatch.setattr(S,'evaluate',lambda model,*a:synthetic_section(*a));events=[];model=type('M',(),{'n':13})();seed=np.r_[80.,np.zeros(12),.8,1.,np.zeros(12)]
 with pytest.raises(RuntimeError,match='fixed full return cap'):S.survey(model,seed,80.,max_evaluations=2,observe=lambda k,v:events.append((k,v)))
 assert len([r for k,r in events if k=='evaluation_input'])==2
 events=[]
 with pytest.raises(RuntimeError,match='resource'):S.survey(model,seed,80.,tick=lambda:(_ for _ in()).throw(RuntimeError('resource')),observe=lambda k,v:events.append((k,v)))
 assert [k for k,v in events]==['evaluation_input','evaluation_failure']

def test_nonfinite_retention_and_inclusive_worker_phase_cap(tmp_path,monkeypatch):
 import pytest,gzip,json
 import scout_native470_flip as D
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.observe('native_return',{'values':[float('nan'),float('inf')]});v=json.loads(gzip.decompress(next((tmp_path/'events').iterdir()).read_bytes()));assert v['values']==[{'nonfinite':'nan'},{'nonfinite':'inf'}]
 w.calls=5976
 with pytest.raises(RuntimeError,match='phase cap'):w.tick()
 assert w.calls==5976

def test_complete_isolated_actual_source_archive(tmp_path):
 import subprocess,shutil
 import scout_native470_flip as D
 for name in D.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import scout_native470_flip as d;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_full_unrelated_native18_profile(tmp_path,monkeypatch):
 import scout_native470_flip as D
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();D.profile(w);r=D.read(tmp_path/'profile.json')
 assert r['passed']and r['target_arrays_unopened']and w.calls==216 and len(r['seconds'])==12 and len(list((tmp_path/'events').glob('*native_return*')))==216

def test_actual_metadata_preflight_never_decodes_proof_or_seed(tmp_path,monkeypatch):
 import scout_native470_flip as D
 class W:
  def check(self):pass
  def save(self,*a):pass
 old=D.read
 def guarded(p):
  assert p.name not in ('target-inputs.json','target-certificate.json.gz','independent512.json','independent470-anchor.json')
  return old(p)
 monkeypatch.setattr(D,'read',guarded);card=dict(old(ROOT/D.CARD),status='reviewed');assert D.binding(W(),card,metadata_only=True).is_dir()
