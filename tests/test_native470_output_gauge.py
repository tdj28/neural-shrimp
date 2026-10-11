import copy,os,subprocess,sys,time
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import probe_native470_output_gauge as D
import launch_native470_output_gauge as L
from neural_shrimp.rigorous_orbits import vector_json

def card():return dict(D.read(ROOT/D.CARD),status='reviewed')
class Memory:
 def __init__(self):self.start=time.monotonic();self.rows={};self.used=0;self.native=180;self.ticks=186
 def check(self):pass
 def tick(self):self.ticks+=1
 def save(self,n,v,closing=False):assert n not in self.rows;self.rows[n]=v;self.used+=len(str(v))
 def observe(self,k,v):
  if k=='native_input':self.native+=1

def test_actual_full_metadata_without_certificate_arrays(monkeypatch):
 w=Memory();old=D.read
 def read(p):
  assert p.name!='target-certificate.json.gz'and '/events/'not in str(p)
  return old(p)
 monkeypatch.setattr(D,'read',read);p=D.bind(w,card(),True);assert p.name=='attempt001'and w.native==180
 with pytest.raises(ValueError):D.bind(Memory(),dict(card(),status='prepared'),True)

def test_changed_gauge_full_projection_and_interior_overlap():
 with ctx.workprec(256):
  n=470;zero=vector_json([arb(0)]*n);x=[arb(1)]*n;x[-2]=arb(77);x[-1]=arb('0.8366')+arb(0,arb((1,-40)));origin=['0']*n
  known=dict(box=vector_json(x),normal=['0']*468+['1','0'],origin=origin,exact_parameter='-1',dimension=13,period=18,sigma=-1)
  slope=[arb(0)]*468+[arb(1),-arb(1)/8192];old=dict(parameter_center='0',slope=vector_json(slope),curvature=zero,origin=origin)
  initial,normal,o,p,anchor=D.output_proposal(known,old);assert normal==['0']*469+['1']and o==origin and 'exact_parameter'not in anchor and len(anchor['box'])==470
  projected=anchor['box'][-1];h=arb((1,-15));assert D.exact(p)-h<projected and projected<D.exact(p)+h and D.exact(initial[-2])<77
  bad=copy.deepcopy(old);bad['slope'][-1]=zero[-1]
  with pytest.raises(ValueError):D.output_proposal(known,bad)

def test_full54phase_unrelated_endpoint_profile(monkeypatch):
 with ctx.workprec(256):
  w=Memory();monkeypatch.setattr(D,'read',lambda _:{});monkeypatch.setattr(D,'old_polynomial',lambda *a:arb_mat([[int(i==j)for j in range(470)]for i in range(470)]));D.endpoint_profile(w)
  assert w.native==234 and w.ticks==243 and len(w.rows['endpoint-profile.json']['times'])==3
  assert all(len(w.rows[f'profile-endpoint-{k}.json.gz']['residual'])==470 for k in range(3))

def test_source_closure_and_real_limits(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import probe_native470_output_gauge as d;import launch_native470_output_gauge as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert D.LIMITS['native_calls']==180+54+54+144 and D.LIMITS['ticks']==186+57+57+150
 text=(ROOT/'tools/launch_native470_output_gauge.py').read_text();assert "'internal_seconds':300"in text and "'external_seconds':315"in text and '1026*1024**2'in text

def test_actual_saved_domain_and_projection_without_new_native(monkeypatch):
 with ctx.workprec(256):
  monkeypatch.setattr(D.C.C.TwoRateCuspMap,'cusp_step',lambda *a:pytest.fail('new native evaluation forbidden'))
  known,chart,domain,model=D.bind(Memory(),card());box=[D.original_ball(v)for v in known['box']];point=[v.mid()for v in box]
  assert D.fixed_parameter_domain(chart,domain,vector_json(box),vector_json(point),known['exact_parameter'])
  inverse=D.old_polynomial(chart,known['exact_parameter']);assert inverse.nrows()==inverse.ncols()==470
  initial,normal,origin,p,anchor=D.output_proposal(known,chart);assert len(initial)==470 and 'exact_parameter'not in anchor
