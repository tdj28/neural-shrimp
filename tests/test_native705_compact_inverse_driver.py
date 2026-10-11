import importlib.util,subprocess,sys,copy
from pathlib import Path
from fractions import Fraction
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_compact_inverse as N
import launch_native705_compact_inverse as L

def test_fixed_fresh_domains_and_full_known_anchor_not_old_coarse_gauge():
 assert N.SETTINGS['halfwidth_exponent']==-4 and N.SETTINGS['radius_exponent']==-7 and N.SETTINGS['newton_steps']==0
 assert Fraction(N.SETTINGS['parameter'])-Fraction(1,16)<Fraction(-1,2048)<Fraction(N.SETTINGS['parameter'])+Fraction(1,16)
 assert N.TARGET==L.BASE/'attempt001-target/run' and N.BASE.name=='correlated002'
 assert N.C.APINS['endpoint']=='d56ed2992697924b767e16b76e7cd1271384899b039e52e8d416f790318ac9be'

def test_all_archive_dependencies_present_and_actual_entrypoint():
 assert set(N.P.SOURCES)<=set(N.SOURCES) and all((ROOT/n).is_file()for n in N.SOURCES)
 r=subprocess.run([sys.executable,str(ROOT/'tools/launch_native705_compact_inverse.py'),'--help'],capture_output=True,text=True)
 assert r.returncode==0 and '--profile-worker'in r.stdout

def test_actual_native_baseline_is_not_erased_by_synthetic_speedup():
 m=dict(one_fresh_chart_seconds=50,rss_kib=1000000,compressed_storage_bytes=10000)
 p=dict(passed=True,connected=True,ticks=148,strict_inclusions=705,anchor_inclusions=705,seconds=132.82,peak_rss_kib=3766432,stored_bytes_before_final=385620844)
 out=N.actual_forecast(m,p);assert out['seconds']==pytest.approx(176.025)and out['rss_kib']>4700000 and out['raw_bytes']>480000000
 for k in('ticks','strict_inclusions','anchor_inclusions'):
  bad=copy.deepcopy(p);bad[k]-=1
  with pytest.raises(ValueError):N.actual_forecast(m,bad)
 p['seconds']=200
 with pytest.raises(ValueError):N.actual_forecast(m,p)

def test_historical_driver_defaults_and_compact_canonical_observer_are_explicit():
 assert N.C.COMPACT is False
 source=(ROOT/'tools/probe_native705_correlated_inverse.py').read_text()
 assert "if COMPACT and k=='correlated_chart_return'"in source and "'canonical_certificate':'certificate.json.gz'"in source
 assert "halfwidth_exponent=SETTINGS['halfwidth_exponent']"in source
