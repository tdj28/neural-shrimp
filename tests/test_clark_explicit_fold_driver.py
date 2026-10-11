import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import pytest
from flint import arb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_clark_explicit_fold as D

def test_counts_bound_every_native_category_and_original_identity():
 card=D.read(ROOT/D.CARD);assert card['caps']==D.CAPS and card['settings']==D.SETTINGS and card['base_card']==D.entry(ROOT/D.B.CARD)
 assert D.CAPS['explicit']==3+1+12*7+5
 assert D.CAPS['native_phase_returns']==3*D.CAPS['explicit']+3*D.CAPS['anchor']+6*D.CAPS['implicit']+6*D.CAPS['context']
 for key in ['failed_implicit','diagnosis','binding_failure']:
  s=card[key];p=Path(s['root']);assert D.entry(p/'manifest.json')['sha256']==s['outer']and D.entry(p/'run/manifest.json')['sha256']==s['worker'];rt=D.read(p/'run/runtime.json');assert rt.get('source_commit',rt.get('commit'))==s['source']


def test_distinct_radius_units_and_strict_ceiling():
 assert D.radius(arb(0),0)==arb((1,-40))
 assert D.radius(arb(1)/128,26)==arb(1)/64
 with pytest.raises(ValueError):D.radius(arb(1)/128,25)
 with pytest.raises(ValueError):D.radius(arb(4),50)


def test_real_full51_synthetic_profile_only(tmp_path,monkeypatch):
 monkeypatch.setattr(D.B,'OUT',tmp_path)
 w=D.Work();D.synthetic(w);p=D.read(tmp_path/'profile.json');assert p['passed']and p['target_arrays_unopened']
 assert w.calls['explicit']==3 and w.calls['anchor']==w.calls['implicit']==w.calls['context']==1 and w.calls['native_phase_returns']==21
 assert p['remaining']==dict(explicit=90,implicit=1,anchor=1,context=13,algebra=48)


def test_archive_import_and_launcher_scope():
 with tempfile.TemporaryDirectory()as folder:
  root=Path(folder)
  for n in D.FILES:
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
  subprocess.run([sys.executable,'-B','-c',"import sys;sys.path[:0]=['tools','src'];import certify_clark_explicit_fold as d;assert callable(d.T.fixed_gain);assert callable(d.B.B.historical)"],cwd=root,check=True,capture_output=True)
 s=(ROOT/'tools/launch_clark_explicit_fold.py').read_text();assert 'certify_clark_explicit_fold.py'in s and 'explicit-fold-bridge/attempt002'in s
 assert 'T.conditional_endpoint_fold'in (ROOT/'tools/certify_clark_explicit_fold.py').read_text()


def test_actual_failed_and_passed_historical_exit_contract():
 card=D.read(ROOT/D.CARD)
 for key,expected in [('failed_implicit',2),('diagnosis',0),('binding_failure',2)]:
  s=card[key];r=Path(s['root']);D.historical(r,s['source'],expected)
  with pytest.raises(ValueError,match='completed exact expected execution'):D.historical(r,s['source'],2-expected)
