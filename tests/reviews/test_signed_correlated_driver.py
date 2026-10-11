import os,subprocess,sys,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_signed_correlated as D
import launch_signed_correlated_review as L
HISTORICAL=ROOT

def card():return dict(D.read(ROOT/D.CARD),status='reviewed')
class W:
 def tick(self):pass
 def save(self,*args,**kwargs):pass

def test_actual_metadata_before_target_arrays(monkeypatch):
 # Scratch-source archive checks use the actual repository object database only.
 monkeypatch.setattr(D.A.C,'ROOT',HISTORICAL)
 old=D.read
 def read(p):
  assert p.name not in ('target-certificate.json.gz','independent512.json','connected-left-endpoint.json')
  assert '/events/' not in str(p)
  return old(p)
 monkeypatch.setattr(D,'read',read)
 p,prior=D.bind(card(),W(),True)
 assert p.name=='attempt003' and prior.name=='attempt001'

def test_unreviewed_and_changed_limits_reject():
 c=card();c['status']='prepared'
 with pytest.raises(ValueError):D.bind(c,W(),True)
 c=card();c['limits']['precision']=256
 with pytest.raises(ValueError):D.bind(c,W(),True)

def test_exact_negative_scalar_and_last_coordinate_anchor():
 from flint import arb,ctx
 from neural_shrimp.rigorous_orbits import vector_json,ball_json
 with ctx.workprec(512):
  c=[arb(0)]*470;s=[arb(0)]*470;a=[arb(0)]*470;n=[arb(0)]*468+[arb(1),arb(0)];r=[arb(1)]*470
  ok=D.D.anchor(vector_json(c),c,s,a,n,c,D.R.exact('-49/256'),arb(1)/8,r,exact_parameter='-17/256');assert ok['passed'] and ok['coordinates']==470
  b=vector_json(c);b[-1]=ball_json(arb(1));assert not D.D.anchor(b,c,s,a,n,c,D.R.exact('-49/256'),arb(1)/8,r,exact_parameter='-17/256')['passed']

def test_archive_closure_and_launcher_limits(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools','src'];import review_signed_correlated as d;import launch_signed_correlated_review as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 source=(ROOT/'tools/launch_signed_correlated_review.py').read_text();assert '10*1024**3' in source and '770*1024**2' in source and '6*1024**2' in source


def test_actual_complete_native_schema_without_interval_reduction(monkeypatch):
 p=Path(card()['producer']['root']);domain,proposal,g,paths=D.acquisitions(p)
 assert len(g)==len(paths)==36 and len(domain['error'])==470
 chart=D.read(p/'run/target-certificate.json.gz')
 assert chart['inverse_polynomial_encoding']=='exact-dyadic-pairs'
 assert chart['inverse_polynomial_exact'][0]==proposal['preconditioner_exact']
 assert chart['settings']==dict(halfwidth_exponent=-3,radius_exponent=-12)
 assert chart['parameter_endpoints']==['-81/256','-17/256']
 for k in ('center','slope','curvature'):assert chart[k]==domain[k]==proposal[k]
 assert all(len(row)==470 for row in chart['inverse_polynomial_exact'][0])
