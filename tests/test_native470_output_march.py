import os,sys,subprocess,time
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import march_native470_output as D
import launch_native470_output_march as L
from neural_shrimp.rigorous_orbits import vector_json as v,ball_json as b

def test_fixed_output_direction_and_width_policy():
 p,l,r=map(F,D.next_parameter('3/16',-18));assert l==F(3,16)and p-l==r-p==F(1,2**18)
 good=dict(uniform=True,joined=True,primitive=True,contraction=F(1,10),minimum=-24,maximum=-12)
 assert D.next_state(-18,1,**good)['width']==-17
 bad=dict(good,uniform=False);assert D.next_state(-18,1,**bad)['width']==-19
 assert D.next_state(-18,1,**dict(good,joined=False))['action']=='stop'
 with pytest.raises(ValueError):D.next_parameter('0',-11)

def test_pending_independent_authority_prevents_launch():
 c=D.read(ROOT/D.CARD);c['status']='reviewed';c['independent']=None
 with pytest.raises(ValueError,match='independent authority'):D.binding(None,c,True)

def test_one_forward_right_root_then_join_failure_stops(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);c=[arb(0)]*468+[arb(76),arb(4)/5];normal=['0']*469+['1'];origin=['0']*470
 known=dict(box=v(c),normal=normal,origin=origin,exact_parameter='4/5',dimension=13,period=18,sigma=-1)
 chart=dict(center=v(c),slope=v([arb(0)]*469+[arb(1)]),curvature=v([arb(0)]*470),parameter_center='4/5',weights=['1']*470,normal=normal,origin=origin)
 monkeypatch.setattr(D,'binding',lambda *a:(known,chart,{},['1']*470));monkeypatch.setattr(D.C.C,'TwoRateCuspMap',lambda _:type('M',(),{'n':13})());calls=[]
 def fake(*args):
  w=args[-1];anchor=args[6];settings=args[7];parameter=F(args[4]);h=F(2)**settings['halfwidth_exponent'];assert F(anchor['exact_parameter'])==parameter-h
  calls.append(parameter);w.native+=144;w.ticks+=150;ends=[]
  for x in(parameter-h,parameter+h):z=c.copy();z[-1]=D.exact(str(x));ends.append(v(z))
  good=len(calls)==1;r=dict(chart,passed=good,gates=dict(uniform_normalized_eigenpair_locus=True,primitive_period=True,anchor_identified=good),bound=dict(contraction=b(arb(1)/10),coordinate_inclusions=[True]*470),anchor_check=dict(coordinate_inclusions=[good]*470),endpoint_root_boxes=ends,parameter_center=str(parameter));w.save('target-certificate.json.gz',r);return r
 monkeypatch.setattr(D.C,'chart',fake);o=D.march(D.Work(),D.read(ROOT/D.CARD));assert len(calls)==2 and o['accepted_indices']==[1]and o['reason']=='successful_root_join_failed'and not o['passed']and o['connected'];assert F(o['last_known_parameter'])==F(4,5)+F(1,2**17)

def test_actual_predecessor_metadata_no_native(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a,**k):pass
 old=D.W.read
 def read(p):
  assert p.name not in('target-certificate.json.gz','known-root-identity.json')
  return old(p)
 monkeypatch.setattr(D.W,'read',read);p,m,a=D.W.bind(W(),D.read(ROOT/D.W.CARD),True)
 assert p.name=='attempt001'and m.name=='attempt001'and a.name=='attempt001'

def test_isolated_archive(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import march_native470_output as d;import launch_native470_output_march as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert D.LIMITS['native_calls']==180+24*144 and D.LIMITS['ticks']==186+24*150
