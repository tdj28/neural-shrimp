import sys,shutil,subprocess,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_signed470_advancing as D

def test_same_native_gauge_uses_actual_old_scalar_only():
 old=dict(normal=['0']*468+['1','0'],origin=['0']*470,parameter='0',weights=['1']*470);new=dict(old,parameter='-9/256');assert D.same_gauge(new,old)=='-1/256'
 new['origin']=list(new['origin']);new['origin'][3]='1'
 with pytest.raises(ValueError,match='identical'):D.same_gauge(new,old)

def test_real_binding_before_proof_arrays(monkeypatch):
 old=D.read
 def guarded(p):
  assert p.name not in ('target-certificate.json.gz','independent512.json');return old(p)
 monkeypatch.setattr(D,'read',guarded)
 class W:
  def tick(self):pass
  def save(self,*a):pass
 c=dict(old(ROOT/D.CARD),status='reviewed');p,a,inp,s=D.binding(c,W());assert s=='-1/256'and inp['settings']['halfwidth_exponent']==-5

def test_isolated_source_and_matching_launch_path(tmp_path):
 for name in D.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','tools'];import review_signed470_advancing as d;import launch_signed470_advancing_review as l;r=Path.cwd();assert d.OUT.parent==l.OUT;a={str(Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(d.FILES),a-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
