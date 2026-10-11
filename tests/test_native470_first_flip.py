import sys,copy,subprocess
from pathlib import Path
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import pytest
from neural_shrimp.native470_first_components import SyntheticFlip,synthetic_fixture,RetainedMap,encode
from neural_shrimp.rigorous_eigenpair_locus import assemble,proposal
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube
from neural_shrimp.rigorous_orbits import exact
import probe_native470_first_flip as D

def test_full470_signed_fixture_all_residuals_and_condensed_regular_gauge():
 with ctx.workprec(256):
  z,k=synthetic_fixture();g,j,local=assemble(SyntheticFlip(),list(map(exact,z)),18,-1,list(map(exact,k['normal'])),list(map(exact,z)),arb(0));c,s=proposal(j,local,-1)
  assert g.nrows()==470 and all(g[i,0]==0 for i in range(470))and c.nrows()==470 and s.nrows()==28
  g,_,_=assemble(SyntheticFlip(),list(map(exact,z)),18,1,list(map(exact,k['normal'])),list(map(exact,z)),arb(0));assert g[455,0]!=0

def test_small_signed_tube_strict_root_anchor_and_primitive2_with_complete_acquisition():
 with ctx.workprec(256):
  z,k=synthetic_fixture(2,2);events=[];ticks=[];m=RetainedMap(SyntheticFlip(2),lambda kind,row:events.append((kind,row)))
  r,_=quadratic_tube(m,z,**k,tick=lambda:ticks.append(1));assert r['passed']and all(r['anchor_check']['coordinate_inclusions'])and m.calls==14 and len(ticks)==15
  assert [k for k,v in events]==['native_input','native_return']*14

def test_preinput_survives_native_exception():
 class Bad:
  n=2
  def cusp_step(self,*a):raise RuntimeError('native failed')
 with ctx.workprec(256):
  z,k=synthetic_fixture(2,2);events=[]
  with pytest.raises(RuntimeError,match='native failed'):quadratic_tube(RetainedMap(Bad(),lambda a,b:events.append((a,b))),z,**k)
  assert len(events)==1 and events[0][0]=='native_input'and len(events[0][1]['inputs'])==4

def test_source_archive_and_settings(tmp_path):
 assert D.SETTINGS['sigma']==-1 and D.settings()['profile_ticks']+D.settings()['target_ticks']==278 and D.settings()['native_calls']==276
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 subprocess.run([sys.executable,'-I','-B','-c',"import sys,pathlib;sys.path.insert(0,'tools');import probe_native470_first_flip as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES)"],cwd=tmp_path,check=True,capture_output=True)

def test_actual_metadata_binding_and_exact_lower_precision_seed(tmp_path,monkeypatch):
 class Check:
  def check(self):pass
  def save(self,*a):pass
 with ctx.workprec(512):fine=arb(1)+arb((1,-410));encoded=__import__('neural_shrimp.rigorous_orbits',fromlist=['ball_json']).ball_json(fine)
 with ctx.workprec(256):
  assert exact(D.exact_mid(fine+arb(0))).is_exact()
  card=D.read(ROOT/D.CARD);card['status']='reviewed';known,inp=D.binding(Check(),card)
  assert len(known['box'])==470 and inp['network']['activation']=='tanh'
