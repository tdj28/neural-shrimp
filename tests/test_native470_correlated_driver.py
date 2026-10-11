import sys,subprocess,shutil,json,time
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native470_correlated as D

def test_actual_metadata_only_and_literal_independent_authority(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a):pass
 old=D.read
 def guarded(p):
  assert p.name not in ('target-inputs.json','target-certificate.json.gz','independent512.json','connected-left-endpoint.json')
  return old(p)
 monkeypatch.setattr(D,'read',guarded);card=dict(old(ROOT/D.CARD),status='reviewed');p,a=D.binding(W(),card,metadata_only=True)
 assert p.name==a.name=='attempt001'and D.SETTINGS['parameter']=='-49/256'
 bad=json.loads(json.dumps(card));bad['independent']['outcome']['root_coordinates']=469
 with pytest.raises(ValueError):D.binding(W(),bad,metadata_only=True)

def test_full470_unrelated_refresh_correlated_and_counts():
 class W:
  ticks=0;native=0
  def tick(self):self.ticks+=1
  def observe(self,k,v):self.native+=k=='native_input'
 with ctx.workprec(256):
  ini,kw=D.synthetic_fixture();w=W();r=D.chart(D.Retained(D.SyntheticFlip(),w.observe),ini,kw['normal'],ini,'0',['1']*470,kw['anchor'],kw,w)
  assert w.ticks==150 and w.native==144
  assert r['bound']['dimension']==470 and all(r['bound']['coordinate_inclusions'])and r['anchor_check']['passed']and not r['gates']['primitive_period']

def test_gradient_input_and_completed_bad_return_retention():
 from neural_shrimp.rigorous_eigenpair_path import PathJet
 class M:
  n=1
  def cusp_step(self,x,h,o):return[x[0]*arb('nan')]
 events=[];m=D.Retained(M(),lambda k,v:events.append((k,v)));v=D.GradientPathJet(PathJet([arb(1),arb(0),arb(0),arb(0)]),[1,0,0]);m.cusp_step([v],v,v)
 assert [k for k,v in events]==['native_input','native_return']and events[0][1]['inputs'][0]['kind']=='gradient_path'

def test_caps_and_archive_import_closure(tmp_path):
 assert D.LIMITS['profile_ticks']+D.LIMITS['target_ticks']==336 and D.LIMITS['native_calls']==324
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;from pathlib import Path;sys.path.insert(0,'tools');import probe_native470_correlated as d;import launch_native470_correlated as l;r=Path.cwd();assert d.OUT.parent==l.OUT;loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)


def test_real_full_profile_exact_timing_inputs_and_exclusive003(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path/'run');D.OUT.mkdir();monkeypatch.setattr(D,'retained_failure',lambda w:None);monkeypatch.setattr(D,'retained_second_failure',lambda w:None)
 with ctx.workprec(256):
  w=D.Work();D.profile(w)
 assert w.native==180 and w.ticks==186
 profile=D.read(D.OUT/'profile.json');assert profile['passed']and profile['target_arrays_unopened']
 old=D.read(D.OUT/'profile-native-fixture.json')
 assert all(v['rad']==[0,0]for k in ('state','mode','rates')for v in old[k])
 assert "attempt003"in (ROOT/'tools/launch_native470_correlated.py').read_text()


def test_actual_archived_compressed_schema_and_both_retained_failures():
 class W:
  def check(self):pass
  def save(self,*a):pass
 D.retained_failure(W());D.retained_second_failure(W())
 c=D.read(ROOT/D.CARD);p=Path(c['producer']['root']);v=D.read(p/'run/target-certificate.json.gz')
 assert v['shooting_dimension']==470 and v['sigma']==-1 and v['parameter_endpoints'][0]=='-17/256'and len(v['center'])==len(v['weights'])==470
 assert all(x['rad']==[0,0]for k in ('center','slope','curvature')for x in v[k])


def test_actual_full_saved_binding_before_any_new_native(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a):pass
 def forbidden(*a,**kw):raise AssertionError('new native map construction forbidden')
 monkeypatch.setattr(D.C,'TwoRateCuspMap',forbidden)
 card=dict(D.read(ROOT/D.CARD),status='reviewed')
 with ctx.workprec(256):inp,initial,known,anchor=D.binding(W(),card)
 assert len(initial)==len(known['box'])==len(anchor['box'])==470 and known['exact_parameter']=='-17/256'and anchor['exact_parameter']=='-17/256'
