import sys,subprocess
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import diagnose_cifar_nonlinearity as D

def test_actual_size_shape_control_without_research_arrays():
 r=D.synthetic(64);assert r['accepted_index']==6 and sum(x['nonlinear_observed']for x in r['rows'])==4

def test_failure_finalization_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=6*1024**2
 with pytest.raises(ValueError):w.save('blocked.json',dict(x=1))
 w.save('outcome.json',dict(passed=False),True);assert D.M.read(tmp_path/'outcome.json')['passed']is False

def test_complete_isolated_source_archive(tmp_path):
 packet=Path(__file__).resolve().parents[2]
 for n in D.FILES:
  src=D.C.ROOT/n
  if not src.is_file():src=packet/n
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes())
 code="import sys,pathlib;sys.path[:0]=['tools/reviews','tools','src'];import diagnose_cifar_nonlinearity as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in sys.modules.values()if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
