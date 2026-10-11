import copy,os,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
import certify_clark_correlated_extension as D
import launch_clark_correlated_extension as L

def test_actual_counts_profile_and_exclusive_source_bound_card(tmp_path,monkeypatch):
 card=D.read(ROOT/D.CARD);assert card['settings']==D.SETTINGS and card['caps']==D.CAPS and D.OUT.parent==L.OUT
 assert D.CAPS['native_phase_returns']==3*(18+20+1+1)+3*6==138
 assert D.CAPS['path']==5*D.SETTINGS['attempts']and D.CAPS['cache_reuses']==2*D.SETTINGS['attempts']
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();old=D.admission(w,card);assert w.calls['native_phase_returns']==0
 out=D.read(tmp_path/'extension-profile.json');assert out['forecast']['passed']and out['forecast']['seconds']<90 and out['forecast']['bytes']<252*1024**2
 profile=D.read(Path(old['profile']['root'])/'run/profile.json');bad=copy.deepcopy(profile);bad['times']['path']=[100]*6;bad['max_seconds']['path']=100;assert not D.forecast(bad)['passed']

def test_actual_full_independent_prefix_metadata_and_changed_endpoint(tmp_path,monkeypatch):
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();card=D.read(ROOT/D.CARD)
 with ctx.workprec(384):known,charts,binding=D.prefix(w,card)
 assert len(known)==51 and len(charts)==6 and charts[-1]['parameter']+charts[-1]['halfwidth']==arb(3)/8192 and w.calls['native_phase_returns']==0
 bad=copy.deepcopy(card);bad['independent_prefix']['endpoint']['sha256']='0'*64
 with pytest.raises(ValueError,match='literal full51'):D.prefix(w,bad)

def test_inherited_march_exact_small_differential_and_endpoint_unchanged():
 old=(ROOT/'tools/certify_clark_correlated_fold.py').read_text();new=(ROOT/'tools/certify_clark_correlated_extension.py').read_text()
 old=old[old.index('def march('):old.index('\ndef endpoint(')].strip();new=new[new.index('def march('):new.index('\ndef main(')].strip()
 expected=old.replace('left=arb(0);','left=arb(3)/8192;').replace('h=arb((1,-13))','h=arb((1,-15))').replace('range(1,9)','range(1,5)').replace('len(accepted)>=6','len(accepted)>=2').replace('initial_parameter=ball_json(arb(0))','initial_parameter=ball_json(arb(3)/8192)')
 assert new==expected
 source=(ROOT/'tools/certify_clark_correlated_extension.py').read_text();assert 'C.endpoint(w,j,gain,arrays,organizer,prior+charts)'in source

def test_owned_caps_archive_import_and_closing_failure(tmp_path,monkeypatch):
 text=(ROOT/'tools/launch_clark_correlated_extension.py').read_text();assert "'internal_seconds':90"in text and "'external_seconds':105"in text and '258*1024**2'in text and 'certify_clark_correlated_extension.py'in text
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 result=subprocess.run([sys.executable,'-c',"import runpy;d=runpy.run_path('tools/certify_clark_correlated_extension.py',run_name='source_import');assert d['CAPS']['native_phase_returns']==138"],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert result.returncode==0,result.stderr
 monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();w.used=252*1024**2;w.stopped=True
 with pytest.raises(ValueError):w.save('no.json',{})
 w.save('outcome.json',dict(passed=False),True);assert (tmp_path/'outcome.json').exists()
