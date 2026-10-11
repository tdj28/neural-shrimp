import sys,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_native705_inverse_correlation as D
import launch_native705_inverse_diagnosis as L

def test_source_closure_and_fixed_failed_scope(tmp_path):
    assert D.OUT.parent==L.OUT and D.OUT.parent.name=='later001-inverse-diagnosis'
    r=D.read(ROOT/'docs/reviews/native705-later001-bound-failures.json');assert r['outer']==D.FAILED and r['accepted_cells']==0 and r['total_native_calls']==1116
    for n in D.FILES:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    code="import sys,pathlib;sys.path.insert(0,'tools');import diagnose_native705_inverse_correlation as d;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(d.FILES),loaded-set(d.FILES)"
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_full_resource_envelope():
    text=(ROOT/'tools/launch_native705_inverse_diagnosis.py').read_text()
    assert "'internal_seconds':120"in text and "'external_seconds':135"in text
    assert "'worker_rss_kib':4*1024**2"in text and "'output_cap_bytes':26*1024**2"in text
    assert 'peak>4*1024**2'in text and '(512*1024**2,6*1024**3)'in text
