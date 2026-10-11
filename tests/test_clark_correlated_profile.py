import importlib.util,os,subprocess,sys
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
D=load('clark_path_profile','tools/profile_clark_correlated_fold.py');L=load('clark_path_profile_launcher','tools/launch_clark_correlated_fold_profile.py')

def test_exact_fixture_dimensions_and_caps():
 with ctx.workprec(384):
  j,c,s,a,e,b,h=D.fixture();assert len(j)==12 and len(c)==len(s)==len(a)==len(e)==51
  assert all(v.is_exact()for v in c+s+a+[b,h]+[v for row in j for v in row])
  assert all(x.contains(0)and x.rad()>0 for x in e)
 assert D.OUT.parent==L.OUT and D.CAPS['native_phases']==6*3+2*3+6==30
 src=(ROOT/'tools/launch_clark_correlated_fold_profile.py').read_text();assert "'internal_seconds':90"in src and "'external_seconds':105"in src and '258*1024**2'in src


def test_retained_failure_reserve_and_call_ceiling(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.calls['path']=D.CAPS['path']
 with pytest.raises(ValueError,match='ceiling'):w.charge('path')
 w.used=252*1024**2
 with pytest.raises(ValueError,match='closing reserve'):w.save('unadmitted.json',{})
 w.save('outcome.json',dict(passed=False),True);assert (tmp_path/'outcome.json').exists()


def test_isolated_archive_import_and_no_target_reader(tmp_path):
 for name in D.FILES:
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
 code="import runpy;d=runpy.run_path('tools/profile_clark_correlated_fold.py',run_name='profile_import');assert len(d['fixture']()[1])==51"
 result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15)
 assert result.returncode==0,result.stderr
 source=(ROOT/'tools/profile_clark_correlated_fold.py').read_text();assert 'np.load'not in source and 'target_arrays_unopened=True'in source
