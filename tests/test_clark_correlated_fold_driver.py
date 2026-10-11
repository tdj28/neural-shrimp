import copy,gzip,importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import certify_clark_correlated_fold as D
import launch_clark_correlated_fold as L
from test_rigorous_clark_correlated_path import setup


def test_actual_profile_caps_settings_and_metadata_admission(tmp_path,monkeypatch):
 card=D.read(ROOT/D.CARD);assert card['settings']==D.SETTINGS and card['caps']==D.CAPS and D.OUT.parent==L.OUT
 assert D.CAPS['native_phase_returns']==3*(32+40+1+1)+8*6==270
 assert D.CAPS['path']==5*D.SETTINGS['attempts']and D.CAPS['cache_reuses']==2*D.SETTINGS['attempts']
 p=D.read(Path(card['profile']['root'])/'run/profile.json');got=D.forecast(p);assert got['passed']and got['seconds']<180 and got['bytes']<508*1024**2
 bad=copy.deepcopy(p);bad['max_seconds']['path']/=2
 with pytest.raises(ValueError):D.forecast(bad)
 bad=copy.deepcopy(p);bad['times']['path']=[100]*6;bad['max_seconds']['path']=100;assert not D.forecast(bad)['passed']
 for k,exitcode in [('profile',0),('failed_correlated',2),('failed_explicit',2),('diagnosis',0)]:
  spec=card[k];rt=D.read(Path(spec['root'])/'run/runtime.json');la=D.read(Path(spec['root'])/'launcher-result.json');assert rt['source_commit']==spec['source']==la['source_commit']and la['exit_code']==exitcode and not la['owned_live_pids']
 monkeypatch.setattr(D.B,'OUT',tmp_path);monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();D.bind_profile(w,card);assert (tmp_path/'profile-binding.json').exists()and w.calls['native_phase_returns']==0


def test_cached_full_input_and_array_identity_and_single_use(tmp_path,monkeypatch):
 monkeypatch.setattr(D.B,'OUT',tmp_path);monkeypatch.setattr(D,'OUT',tmp_path)
 with ctx.workprec(384):
  j,c,s,a,b=setup();zero=[arb(0)]*11;w=D.Work();answer=w.path(j,c,s,a,zero,b,arb(0));cache=D.ResidualCache(w);cache.add(j,c,s,a,zero,b,arb(0),answer)
  wrong=c.copy();wrong[0]+=arb(1)/64
  with pytest.raises(ValueError,match='literal unique'):cache.acquire(j,wrong,s,a,zero,b,arb(0))
  assert cache.acquire(j,c,s,a,zero,b,arb(0))is answer and w.calls['path']==1 and w.calls['cache_reuses']==1
  with pytest.raises(ValueError,match='literal unique'):cache.acquire(j,c,s,a,zero,b,arb(0))
  cache2=D.ResidualCache(w);answer2=w.path(j,c,s,a,zero,b,arb(0));cache2.add(j,c,s,a,zero,b,arb(0),answer2);answer2[0][0,0]+=1
  with pytest.raises(ValueError,match='immutable same'):cache2.acquire(j,c,s,a,zero,b,arb(0))


def test_full_error_is_never_replaced_by_zero_cache(tmp_path,monkeypatch):
 monkeypatch.setattr(D.B,'OUT',tmp_path);monkeypatch.setattr(D,'OUT',tmp_path)
 with ctx.workprec(384):
  j,c,s,a,b=setup();zero=[arb(0)]*11;error=[arb(0,arb((1,-30)))]*11;w=D.Work();cache=D.ResidualCache(w);answer=w.path(j,c,s,a,zero,b,arb(0));cache.add(j,c,s,a,zero,b,arb(0),answer);out=cache.acquire(j,c,s,a,error,b,arb(0));assert w.calls['path']==2 and w.calls['cache_reuses']==0
  row=json.loads(gzip.decompress((tmp_path/w.last_path).read_bytes()));assert row['error']==D.vector_json(error)
  with pytest.raises(ValueError,match='only zero'):cache.add(j,c,s,a,error,b,arb(0),out)


def test_archive_import_and_expected_endpoint_curved_join(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 result=subprocess.run([sys.executable,'-c',"import runpy;d=runpy.run_path('tools/certify_clark_correlated_fold.py',run_name='source_import');assert d['CAPS']['path']==40"],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert result.returncode==0,result.stderr
 source=(ROOT/'tools/certify_clark_correlated_fold.py').read_text();assert "CT.join(data['unknown'],b,c['center'],c['slope'],c['curvature']"in source
 # Same endpoint implementation except the required quadratic projection.
 old=(ROOT/'tools/certify_clark_explicit_fold.py').read_text();old=old[old.index('def endpoint('):old.index('\n\ndef main():')]
 new=source[source.index('def endpoint('):source.index('\n\ndef main():')]
 assert new.strip()==old.replace("T.join(data['unknown'],b,c['center'],c['slope'],c['parameter'],c['halfwidth'],c['radii'])","CT.join(data['unknown'],b,c['center'],c['slope'],c['curvature'],c['parameter'],c['halfwidth'],c['radii'])").strip()


def test_actual_retained_failure_metadata_before_old_target_binder(tmp_path,monkeypatch):
 monkeypatch.setattr(D.B,'OUT',tmp_path);monkeypatch.setattr(D,'OUT',tmp_path);monkeypatch.setattr(D.OLD,'bind',lambda w:('unopened',))
 w=D.Work();assert D.bind(w,D.read(ROOT/D.CARD))==('unopened',)and w.calls['native_phase_returns']==0
 assert (tmp_path/'failed_correlated-binding.json').exists()and (tmp_path/'failed_explicit-binding.json').exists()and (tmp_path/'diagnosis-binding.json').exists()
