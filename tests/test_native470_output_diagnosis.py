import os,sys,subprocess,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import diagnose_native470_output_gauge as D
import launch_native470_output_diagnosis as L

class W:
 start=time.monotonic()
 def check(self):pass
 def save(self,*a,**kw):pass

def test_actual_metadata_with_failed_certificate_closed(monkeypatch):
 old=D.read
 def read(p):
  assert p.name!='target-certificate.json.gz'and '/events/'not in str(p)
  return old(p)
 monkeypatch.setattr(D,'read',read);c=dict(old(ROOT/D.CARD),status='reviewed');p=D.bind(W(),c,True);assert p.name=='attempt001'
 with pytest.raises(ValueError):D.bind(W(),dict(c,status='prepared'),True)

def test_isolated_archive_and_fixed_limits(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools','src'];import diagnose_native470_output_gauge as d;import launch_native470_output_diagnosis as l;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert d.OUT.parent==l.OUT"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert D.LIMITS['native_calls']==0 and '105'in(ROOT/'tools/launch_native470_output_diagnosis.py').read_text()
