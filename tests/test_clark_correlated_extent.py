import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_clark_correlated_extent as W
import launch_clark_correlated_extent as L

def card():
 c=W.read(ROOT/W.CARD);c['status']='reviewed';return c

def test_fixed_inclusive_caps_and_old_profile_actual_floor(tmp_path,monkeypatch):
 c=card();W.card_check(c);assert W.OUT.parent==L.OUT
 assert W.CAPS['native_phase_returns']==3*(102+160+1)+6+33*6==993
 assert W.CAPS['path']==5*32 and W.CAPS['cache_reuses']==2*32
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();old=W.admission(w,c);a=W.read(tmp_path/'extent-profile.json')['forecast'];assert a['passed']and a['seconds']<600 and a['bytes']<2*1024**3
 bad=copy.deepcopy(c);bad['settings']['attempts']=33
 with pytest.raises(ValueError):W.card_check(bad)
 profile=W.read(Path(old['profile']['root'])/'run/profile.json');assert not W.forecast(profile,1000,1000)['passed']
 assert not W.forecast(profile,1,2*1024**3)['passed']

def test_exact_independent_endpoint_no_native(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work()
 with ctx.workprec(384):known,m=W.bind_endpoint(w,card())
 assert len(known)==51 and set(m)=={'schema','j0'}and w.calls['native_phase_returns']==0
 bad=copy.deepcopy(card());bad['independent_extension']['endpoint']['sha256']='0'*64
 with pytest.raises(ValueError,match='literal independently'):W.bind_endpoint(w,bad)

def test_actual_numeric_index_metadata_without_arrays():
 c=card();r=W.read(Path(c['endpoint_numeric_parent']['root'])/'run/result.json');d=W.read(Path(c['geometry']['root'])/'run/points.json');row=next(v for v in d if v['parent']==0 and v['tag']==c['endpoint_numeric_tag'])
 assert c['endpoint_numeric_tag']=='fold47'and r['points'][47]['gain']==row['gain']==4.109326262981831 and 1/2048<row['b']<1/16
 assert r['points'][1]['gain']!=row['gain']

def test_step_ladder_streak_and_failures():
 h=arb((1,-15));h1,s=W.next_width(h,True,arb('.2'),0);assert h1==h and s==1
 h2,s=W.next_width(h1,True,arb('.2'),s);assert h2==2*h and s==0
 h3,s=W.next_width(h2,False,arb('.1'),1);assert h3==h and s==0
 assert W.next_width(h,True,arb('.25'),1)==(h,0)
 assert W.next_width(arb((1,-9)),True,arb('.1'),1)==(arb((1,-9)),0)
 left=arb(1)/2048;h=arb((1,-15));streak=0
 for _ in range(32):
  if left==arb(1)/16:break
  h=min(h,(arb(1)/16-left)/2);left+=2*h;h,streak=W.next_width(h,True,arb('.1'),streak)
 assert left==arb(1)/16

def test_closing_reserve_and_archived_import(tmp_path,monkeypatch):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import certify_clark_correlated_extent as w;import launch_clark_correlated_extent as l;assert w.OUT.parent==l.OUT;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=2*1024**3-4*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',{})
 w.save('failed.json',{'passed':False},True)
 text=(ROOT/'tools/launch_clark_correlated_extent.py').read_text();assert "'internal_seconds':600"in text and "'external_seconds':615"in text and 'certify_clark_correlated_extent.py'in text


def test_full_march_retains_negative_and_stops_on_association(monkeypatch):
 from flint import arb_mat
 class Stub:
  def __init__(self):self.rows={}
  def family(self,*args):return arb_mat(51,1),W.CT.eye(51),arb_mat(51,1),None
  def save(self,k,v,*a):self.rows[k]=v
  def charge(self,*a):pass
  def emit(self,*a):pass
  def tick(self):pass
  def context(self,*a,**k):return dict(passed=True,positive_amplitude=True,partner_orientation=[W.ball_json(arb(1))]*2)
 known=[arb(0)]*51;counter=[0]
 class Cache:used={0,1};acquire=None
 monkeypatch.setattr(W.C,'prepare',lambda *a:([arb(0)]*51,[arb(0)]*51,[arb(0)]*51,[arb(1)]*51,Cache()))
 def tube(*a,**k):
  counter[0]+=1;return known,[known,known],None,dict(passed=counter[0]!=1,contraction=W.ball_json(arb('.1')))
 monkeypatch.setattr(W.CT,'tube',tube);monkeypatch.setattr(W.CT,'join',lambda *a:dict(passed=True,coordinates=51))
 w=Stub();rows=W.march(w,None,known);assert w.rows['continuation.json']['passed']and rows and not w.rows['charts/01/decision.json']['passed']
 assert w.rows['connected-endpoint.json']['parameter']==W.ball_json(arb(1)/16)
 counter[0]=1;monkeypatch.setattr(W.CT,'join',lambda *a:dict(passed=False,coordinates=51))
 w=Stub()
 with pytest.raises(ValueError,match='predecessor'):W.march(w,None,known)
 assert len([k for k in w.rows if k.endswith('/attempt.json')])==1
