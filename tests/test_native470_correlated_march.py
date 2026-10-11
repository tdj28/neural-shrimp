import copy,os,subprocess,sys,time
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import march_native470_correlated as D
import launch_native470_correlated_march as L
from neural_shrimp.rigorous_orbits import vector_json,ball_json

def card():return dict(D.read(ROOT/D.CARD),status='reviewed')

def test_actual_metadata_without_target_arrays(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a,**kw):pass
 old=D.read
 def read(p):
  assert p.name not in('target-certificate.json.gz','independent512.json','connected-left-endpoint.json')
  assert '/events/'not in str(p)
  return old(p)
 monkeypatch.setattr(D,'read',read);p,a=D.binding(W(),card(),True)
 assert p.name=='attempt003'and a.name=='attempt001'
 with pytest.raises(ValueError):D.binding(W(),dict(card(),status='prepared'),True)

def fixture():
 c=[arb(0)]*470;c[-2]=arb(84);c[-1]=arb(1);v=vector_json(c);normal=['0']*468+['1','0'];origin=['0']*470
 old=dict(center=v,slope=vector_json([arb(0)]*468+[arb(1),arb(0)]),curvature=vector_json([arb(0)]*470),normal=normal,origin=origin,parameter_center='84',weights=['1']*470)
 known=dict(box=v,normal=normal,origin=origin,exact_parameter='84',dimension=13,period=18,sigma=-1);return known,old

def test_completed_join_failure_never_advances_or_retries(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);known,old=fixture();monkeypatch.setattr(D,'binding',lambda w,c:(known,old,{},['1']*470));calls=[]
 monkeypatch.setattr(D.C.C,'TwoRateCuspMap',lambda _:type('M',(),{'n':13})())
 def chart(*args):
  w=args[-1];calls.append(1);w.native+=144;w.ticks+=150
  r=dict(passed=False,gates=dict(uniform_normalized_eigenpair_locus=True,primitive_period=True,anchor_identified=False),bound=dict(contraction=ball_json(arb(1)/10),coordinate_inclusions=[True]*470),anchor_check=dict(coordinate_inclusions=[True]*469+[False]),endpoint_root_boxes=[known['box']]*2)
  w.save('target-certificate.json.gz',r);return r
 monkeypatch.setattr(D.C,'chart',chart);o=D.march(D.Work(),card());assert len(calls)==1 and not o['passed']and not o['connected']and o['accepted_indices']==[]and o['reason']=='successful_root_join_failed'

def test_real_exclusive_prior_binding_namespaces(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();D.Prefix(w,'preflight-').save('actual-cost-floor.json',{});D.Prefix(w,'ancestry-').save('actual-cost-floor.json',{});assert len(list(tmp_path.glob('*cost-floor.json')))==2

def test_isolated_archive_and_fixed_envelope(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import march_native470_correlated as d;import launch_native470_correlated_march as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert D.LIMITS['native_calls']==180+24*144 and D.LIMITS['ticks']==186+24*150
 source=(ROOT/'tools/launch_native470_correlated_march.py').read_text();assert 'external240s unit cap'in source and '6015'in source and '6*1024**3+8*1024**2'in source


def test_unit_start_is_complete_atomic_record(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.unit(1);p=tmp_path/'attempt001/unit-start.json';r=D.read(p);assert r['index']==1 and r['maximum_seconds']==240 and not list(tmp_path.rglob('*.part'))

def test_actual_completed_payload_schema_without_new_native(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a,**kw):pass
 monkeypatch.setattr(D.C.C.TwoRateCuspMap,'cusp_step',lambda *a:pytest.fail('new native evaluation forbidden'))
 k,c,m,w=D.binding(W(),card());assert len(k['box'])==470 and k['exact_parameter']=='-81/256'and len(w)==470 and c['passed']
