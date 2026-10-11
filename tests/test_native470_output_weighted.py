import copy,os,subprocess,sys,time
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import probe_native470_output_weighted as W
import launch_native470_output_weighted as L
class Memory:
 def __init__(self):self.rows={};self.start=time.monotonic();self.used=0
 def check(self):pass
 def tick(self):pass
 def save(self,n,v,closing=False):assert n not in self.rows;self.rows[n]=v

def card():return dict(W.read(ROOT/W.CARD),status='reviewed')
def test_actual_metadata_without_derivative_arrays(monkeypatch):
 old=W.read
 def rd(p):
  assert p.name not in ('target-certificate.json.gz','diagnostic.json')and '/events/'not in str(p)
  return old(p)
 monkeypatch.setattr(W,'read',rd);p,m,a=W.bind(Memory(),card(),True);assert p.name=='attempt001'and a.name=='attempt001'
 with pytest.raises(ValueError):W.bind(Memory(),dict(card(),status='prepared'),True)

def test_actual_original_domain_same_root_and_new_gauge_no_native(monkeypatch):
 monkeypatch.setattr(W.C.C.TwoRateCuspMap,'cusp_step',lambda *a:pytest.fail('target native forbidden'))
 with ctx.workprec(256):
  box,old,model,weights,known=W.bind(Memory(),card());ini,n,o,p,a=W.proposal(box,old)
  assert len(box)==len(ini)==len(weights)==470 and a['box']==box and 'exact_parameter'not in a
  assert n==['0']*469+['1']and W.F(weights[-2])==1 and W.F(weights[-1])==W.F(1,1048576)
  h=arb((1,-18));projected=box[-1]-W.exact(o[-1]);assert W.exact(p)-h<projected and projected<W.exact(p)+h
  bad=copy.deepcopy(known);bad['exact_parameter']='-2000/256'
  ap=Path(card()['independent']['root']);prior=W.read(ROOT/W.D.RECEIPT);ib=W.read(ap/'run/input-binding.json');out=W.read(ap/'run/outcome.json');oldib=W.read(Path(prior['root'])/'run/input-binding.json')
  with pytest.raises(ValueError):W.check_authority(out,ib,bad,prior,oldib)
  altered=copy.deepcopy(ib);altered['model']={'changed':True}
  with pytest.raises(ValueError):W.check_authority(out,altered,known,prior,oldib)

def test_actual_archive_closure_and_limits(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import probe_native470_output_weighted as w;import launch_native470_output_weighted as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(w.FILES),loaded-set(w.FILES);assert w.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert W.LIMITS['native_calls']==180+144 and W.LIMITS['ticks']==186+150
 text=(ROOT/'tools/launch_native470_output_weighted.py').read_text();assert "'internal_seconds':240"in text and "'external_seconds':255"in text and '770*1024**2'in text
