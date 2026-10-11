import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_clark_localized_successor as W
import launch_clark_localized_successor as L

def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';return c

def test_fixed_metadata_and_all_native_counts():
 c=card();W.card_check(c);assert W.OUT.parent==L.OUT
 assert W.CAPS['native_phase_returns']==3*(78+120+1)+6+25*6==753
 assert W.CAPS['algebra']==24*4+2 and W.CAPS['subset']==25
 assert W.SETTINGS['initial_amplitude_squared']=='1/512'
 p=Path(c['known']['root'])/'run';o=W.read(p/'outcome.json');ib=W.read(p/'input-binding.json');assert o['passed']and o['new_independent_tubes']==1 and o['new_native_contexts']==4
 assert ib['parent']==W.NATIVE_PRIOR['root']and W.entry(p/'connected-endpoint.json')==W.KNOWN['endpoint']
 for key in('known','screen','prior','native_prior'):
  bad=copy.deepcopy(c);bad[key]['source']='0'*40
  with pytest.raises(ValueError,match='pins'):W.card_check(bad)

def test_actual_cost_floor_and_full51_synthetic(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work()
 with ctx.workprec(384):unit,raw=W.synthetic_profile(w)
 p=W.read(ROOT/W.C.CARD);profile=W.read(Path(p['profile']['root'])/'run/profile.json');prior=W.read(Path(W.PRIOR['root'])/'run/outcome.json');screen=W.read(Path(W.SCREEN['root'])/'run/outcome.json');size=lambda s:sum(v['bytes']for v in W.read(Path(s['root'])/'run/manifest.json')['files'].values())
 f=W.forecast(profile,prior,screen,size(W.PRIOR),size(W.SCREEN),unit,raw);assert f['passed']and f['seconds']>=2*prior['elapsed_seconds']+48*screen['seconds']/10+20
 assert w.calls['native_phase_returns']==0 and w.calls['subset']==1 and w.calls['algebra']==2
 assert not W.forecast(profile,dict(prior,elapsed_seconds=300),screen,1,1,unit,raw)['passed']
 assert not W.forecast(profile,prior,screen,2*1024**3,1,unit,raw)['passed']

def test_exploratory_schedule_does_not_relax_q_gate():
 h=arb((1,-14));h,s=W.next_width(h,True,arb('.8'),0);assert s==1
 h,s=W.next_width(h,True,arb('.8'),s);assert h==arb((1,-13))and s==0
 assert W.next_width(h,True,arb('.9'),1)==(h,0)
 assert W.next_width(h,False,arb('.1'),1)==(h/2,0)
 assert W.next_width(arb((1,-7)),True,arb('.8'),1)==(arb((1,-7)),0)
 left=arb(1)/512;h=arb((1,-14));streak=0
 for _ in range(24):
  if left==arb(1)/16:break
  h=min(h,(arb(1)/16-left)/2);left+=2*h;h,streak=W.next_width(h,True,arb('.8'),streak)
 assert left==arb(1)/16

def test_march_fallback_requires_native_and_original_is_retained(monkeypatch):
 class Stub:
  def __init__(self):self.rows={};self.subsets=0;self.contexts=0
  def family(self,*args):return arb_mat(51,1),W.CT.eye(51),arb_mat(51,1),None
  def save(self,k,v,*a):self.rows[k]=v
  def charge(self,k):self.subsets+=k=='subset'
  def emit(self,*a):pass
  def tick(self):pass
  def context(self,*a,**k):
   self.contexts+=1;return dict(passed=self.contexts!=1,positive_amplitude=True,partner_orientation=[W.ball_json(arb(1))]*2)
 known=[arb(0)]*51
 class Cache:used={0,1};acquire=None
 monkeypatch.setattr(W.C,'prepare',lambda *a:([arb(0)]*51,[arb(0)]*51,[arb(0)]*51,[arb(1)]*51,Cache()))
 proof=dict(passed=False,contraction=W.ball_json(arb(2)),inverse_coefficients=[],jacobian_coefficients=[],residual_coefficients=[])
 monkeypatch.setattr(W.CT,'tube',lambda *a,**k:(known,[known,known],{},proof))
 monkeypatch.setattr(W.S,'restrict',lambda *a,**k:(known,[known,known],{},dict(proof,passed=True,contraction=W.ball_json(arb('.8'))),[arb(1)]*51))
 monkeypatch.setattr(W.CT,'join',lambda *a:dict(passed=True,coordinates=51))
 monkeypatch.setattr(W.EL,'localize',lambda *a,**k:dict(passed=True,restricted=True,endpoint=known))
 monkeypatch.setattr(W,'serialize_localization',lambda v:v)
 w=Stub();rows=W.march(w,None,known)
 assert rows and not w.rows['charts/01/decision.json']['passed']and w.rows['charts/01/decision.json']['subset_fallback']
 assert not w.rows['charts/01/tube-original.json.gz']['passed']and w.rows['charts/01/tube.json.gz']['passed']
 assert w.rows['continuation.json']['passed']and w.subsets==w.contexts<=24
 monkeypatch.setattr(W.CT,'join',lambda *a:dict(passed=False,coordinates=50))
 with pytest.raises(ValueError,match='predecessor'):W.march(Stub(),None,known)

def test_failed_lp_never_promotes_and_finite_attempt_cap(monkeypatch):
 class Stub:
  def __init__(self):self.rows={}
  def family(self,*a):return arb_mat(51,1),W.CT.eye(51),arb_mat(51,1),None
  def save(self,k,v,*a):self.rows[k]=v
  def charge(self,*a):pass
  def emit(self,*a):pass
  def tick(self):pass
  def context(self,*a,**k):raise AssertionError('failed LP cannot qualify native')
 known=[arb(0)]*51
 class Cache:used={0,1};acquire=None
 monkeypatch.setattr(W.C,'prepare',lambda *a:(known,known,known,[arb(1)]*51,Cache()))
 p=dict(passed=False,contraction=W.ball_json(arb(2)),inverse_coefficients=[],jacobian_coefficients=[],residual_coefficients=[])
 monkeypatch.setattr(W.CT,'tube',lambda *a,**k:(known,[known,known],{},p));monkeypatch.setattr(W.S,'restrict',lambda *a,**k:None);w=Stub()
 with pytest.raises(ValueError,match='incomplete'):W.march(w,None,known)
 assert w.rows['continuation.json']['accepted']==0 and len(w.rows['continuation.json']['attempts'])==4

def test_isolated_archive_startup_and_closing_reserve(tmp_path,monkeypatch):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import certify_clark_localized_successor as w;import launch_clark_localized_successor as l;assert w.OUT.parent==l.OUT;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=2*1024**3-4*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('failed.json',{'passed':False},True)
 text=(ROOT/'tools/launch_clark_localized_successor.py').read_text();assert "'internal_seconds':600"in text and "'external_seconds':615"in text and 'certify_clark_localized_successor.py'in text


def test_full51_localization_profile_and_cost_is_charged(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work()
 with ctx.workprec(384):seconds,raw=W.localization_profile(w)
 assert w.calls['localization']==1 and w.calls['native_phase_returns']==0 and seconds<5 and raw<1024**2
 p=W.read(ROOT/W.C.CARD);profile=W.read(Path(p['profile']['root'])/'run/profile.json');prior=W.read(Path(W.PRIOR['root'])/'run/outcome.json');screen=W.read(Path(W.SCREEN['root'])/'run/outcome.json')
 base=W.forecast(profile,prior,screen,1,1,.1,1)
 f=W.forecast(profile,prior,screen,1,1,.1,1,localization_seconds=10,localization_bytes=1024)
 assert not f['passed']and f['seconds']>=base['seconds']+48*9.9 and f['bytes']==base['bytes']+48*1024
