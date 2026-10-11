import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_clark_amplitude_fold as P
import launch_clark_amplitude_fold as L


def test_exact_fixed_caps_paths_and_complete_genealogy_metadata():
 card=P.read(ROOT/P.CARD);assert card['settings']==P.SETTINGS and card['caps']==P.CAPS and L.OUT==P.OUT.parent
 assert P.CAPS['family']==3+1+12*7+5 and P.CAPS['context']==1+12+1
 assert P.CAPS['native_phase_returns']==6*(P.CAPS['family']+P.CAPS['context'])
 assert card['base_card']==P.entry(ROOT/P.B.CARD)
 # Metadata contracts only, no derivative/array decoding.
 old=P.read(ROOT/P.B.CARD);ip=Path(old['independent']['root'])/'run';out=P.read(ip/'outcome.json')
 assert P.read(ip/'unfolding.json')['multiplier_leak_derivative']==out['multiplier_leak_derivative']
 assert P.read(ip/'quintic.json')['quintic']==out['quintic']
 n=Path(old['numerical']['root'])/'run'/card['numerical_fold']['point'];assert P.entry(n)==card['numerical_fold']['point_identity']
 md=P.read(n);assert md['array_sha256']==card['numerical_fold']['array_identity']['sha256']and md['gain']==4.0756


def test_radii_are_strict_finite_dyadic_proposals_with_fixed_bounds():
 with ctx.workprec(384):
  for e in(-50,-30,-9):
   v=arb((1,e));r=P.dyadic_radius(v);assert r>v and r.is_exact()and r<=arb((1,-8))
  with pytest.raises(ValueError):P.dyadic_radius(arb((1,-8)))
  assert P.reducible(ValueError('strict full paired inverse bound'))
  assert not P.reducible(ValueError('source identity'))
  assert not P.reducible(ValueError('all26 exact predecessor join'))


def test_full_native_profile_is_bounded_in_fresh_process_before_any_bind(tmp_path):
 code='import sys;sys.path.insert(0,'+repr(str(ROOT/'tools'))+');import certify_clark_amplitude_fold as p;from flint import ctx;p.OUT=__import__("pathlib").Path('+repr(str(tmp_path))+');ctx.prec=384;w=p.Work();p.synthetic(w);assert w.calls["family"]==3 and w.calls["context"]==1 and w.calls["native_phase_returns"]==24;assert p.read(p.OUT/"profile.json")["passed"]'
 env=dict(os.environ,**dict.fromkeys(P.THREADS,'1'),PYTHONDONTWRITEBYTECODE='1');result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,timeout=30,capture_output=True);assert result.returncode==0,result.stderr.decode()


def test_failed_acquisition_input_and_complete_success_returns_preserved(tmp_path,monkeypatch):
 monkeypatch.setattr(P,'OUT',tmp_path);w=P.Work();monkeypatch.setattr(P.A,'augmented',lambda *args:(_ for _ in ()).throw(ValueError('strict full paired inverse bound')))
 with pytest.raises(ValueError):w.family([[arb(1)]],[arb(0),arb('1/2'),arb(1),arb(1)],arb(0))
 assert w.calls['family']==1 and len(list((tmp_path/'events').glob('*-family_attempt.json.gz')))==1
 w.used=508*1024**2-5
 with pytest.raises(ValueError):w.save('oversize.json',dict(x='large'))
 w.save('failure.json',dict(passed=False),closing=True)


def test_isolated_archive_import_closure_and_launcher_actual_entrypoint(tmp_path):
 for name in P.FILES:
  out=tmp_path/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
 code='import sys,pathlib;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import certify_clark_amplitude_fold as p;assert p.ROOT==pathlib.Path('+repr(str(tmp_path))+');loaded={str(pathlib.Path(m.__file__).resolve().relative_to(p.ROOT))for m in list(sys.modules.values())if getattr(m,"__file__",None)and pathlib.Path(m.__file__).resolve().is_relative_to(p.ROOT)};assert loaded<=set(p.FILES),loaded-set(p.FILES)'
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
 source=(ROOT/'tools/launch_clark_amplitude_fold.py').read_text();assert 'tools/certify_clark_amplitude_fold.py'in source and "'external_seconds':195"in source and '>195'in source and '514*1024**2'in source
