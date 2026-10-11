import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_bridge as W

def test_full26_and_native12_measured_profile(tmp_path):
 # Fresh worker, matching the real launcher: do not inherit pytest high-water RSS.
 code="import sys;sys.path.insert(0,"+repr(str(ROOT/'tools/reviews'))+");import review_clark_bridge as w;from flint import ctx;from pathlib import Path;w.OUT=Path("+repr(str(tmp_path))+");ctx.prec=512;w.synthetic(w.Work())"
 subprocess.run([sys.executable,'-I','-B','-c',code],check=True,capture_output=True,timeout=10)
 a=json.loads((tmp_path/'profile.json').read_text());assert a['passed']and a['target_arrays_unopened']and a['native_calls']==6

def test_failure_reserve(tmp_path,monkeypatch):
 monkeypatch.setattr(W,'OUT',tmp_path);w=W.Work();w.used=30*1024**2-1
 with pytest.raises(ValueError):w.save('large.json',dict(v='x'*50))
 w.save('outcome.json',dict(passed=False),True);assert(tmp_path/'outcome.json').is_file()

def test_isolated_archive_and_actual_launcher(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code="import sys;sys.path.insert(0,'tools/reviews');import review_clark_bridge as w;assert all((w.ROOT/n).is_file()for n in w.FILES)"
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
 s=(ROOT/'tools/launch_clark_bridge_review.py').read_text();assert str(W.OUT.parent)in s and "'internal_seconds':30,'external_seconds':45"in s and '(512*1024**2,3*1024**3)'in s


def test_exact_gain_set_coverage_accepts_outward_expression_radii():
 from flint import arb
 with ctx.workprec(512):
  s=arb(4);h=arb((1,-11))
  parameter=W.serial(arb(s,arb((536870913,-40))))
  predictor=W.serial(arb(s,arb((536870916,-40))))
  t=dict(parameter_center=W.serial(s),halfwidth=W.serial(h),predictor_box=[predictor])
  W.parameter_coverage(t,parameter,parameter)
  too_small=W.serial(arb(s,h/2))
  with pytest.raises(ValueError,match='closed gain range'):W.parameter_coverage(t,too_small,too_small)
  with pytest.raises(ValueError,match='same actual parameter'):W.parameter_coverage(t,parameter,predictor)
  t['predictor_box']=[too_small]
  with pytest.raises(ValueError,match='closed gain range'):W.parameter_coverage(t,parameter,parameter)
