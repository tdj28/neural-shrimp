import importlib.util,os,subprocess,sys
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_correlated as D
spec=importlib.util.spec_from_file_location('clark_review_launcher',ROOT/'tools/launch_clark_correlated_review.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)


def test_fixed_actual_metadata_counts_geometry_and_exclusive_root():
 assert D.OUT.parent==L.OUT and sum(D.EXPECTED.values())==577
 out=D.read(D.PARENT/'run/outcome.json');cont=D.read(D.PARENT/'run/continuation.json')
 assert out['calls']['path']==40 and out['calls']['native_phase_returns']==234 and out['error']=='ValueError: bounded correlated critical range incomplete'
 assert cont['accepted']==6 and D.A.point(cont['last_parameter'])==D.F(3,8192)
 assert D.entry(D.PARENT/'manifest.json')['sha256']==D.PINS[0]and D.entry(D.PARENT/'run/manifest.json')['sha256']==D.PINS[1]
 assert D.read(D.PARENT/'launcher-result.json')['source_commit']==D.SOURCE
 text=(ROOT/'tools/launch_clark_correlated_review.py').read_text();assert 'tools/reviews/review_clark_correlated.py'in text and "'internal_seconds':180"in text and "'external_seconds':195"in text and '130*1024**2'in text


def test_full_size_synthetic_admission_zero_native(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work()
 with ctx.workprec(512):D.synthetic(w)
 out=D.read(tmp_path/'profile.json');assert out['passed']and out['native_calls']==0 and out['target_arrays_unopened']


def test_final_failure_retained_after_stop_and_near_limit(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=124*1024**2;w.stopped=True
 with pytest.raises(ValueError):w.save('unadmitted.json',{})
 w.save('outcome.json',dict(passed=False),True);w.save('manifest.json',dict(files={}),True);assert (tmp_path/'outcome.json').exists()


def test_isolated_archive_import_without_producer_theorems(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys,runpy;sys.path.insert(0,'tools/reviews');d=runpy.run_path('tools/reviews/review_clark_correlated.py',run_name='source_import');assert sum(d['EXPECTED'].values())==577;assert not any('rigorous_clark_' in x for x in sys.modules)"
 result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert result.returncode==0,result.stderr
