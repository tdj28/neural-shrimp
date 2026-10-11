import importlib.util,sys,subprocess
from pathlib import Path
from fractions import Fraction
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json,exact
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native_unit_pair_pal_repair as R
import launch_native_unit_pair_pal_repair as L

def diagnostic():return {'scaling':{'iterations':512,'collatz_max':.9016},'resolvent_proposal':{'nonnegative':True,'radii':[2.**-20]*705}}

def test_exact_fourfold_radii_and_scope_guards():
 w=R.repair_weights(diagnostic());assert len(w)==705 and all(Fraction(v)*Fraction(2)**-10==Fraction(2)**-18 for v in w)
 for which in ['nonnegative','size','nan','large','zero']:
  d=diagnostic();p=d['resolvent_proposal']
  if which=='nonnegative':p['nonnegative']=False
  elif which=='size':p['radii'].pop()
  else:p['radii'][0]={'nan':float('nan'),'large':1.,'zero':0.}[which]
  with pytest.raises(ValueError):R.repair_weights(d)

def test_original_anchor_unchanged_new_exact_gauge_interval():
 with ctx.workprec(256):
  a=[arb(0)]*702+[arb(1),arb(1),exact('1/32768')];p={'slope':[[ball_json(arb(1))]for _ in range(704)],'weights':['1/2']*704};w=R.repair_weights(diagnostic())
  z,n,o,got,prop=R.chart_seed(a,p,w);assert got==w and prop['known_endpoint']==[ball_json(v)for v in a]
  assert all(exact(v).is_exact()for v in z+n+o)
  c=Fraction(R.SETTINGS['parameter']);h=Fraction(2)**R.SETTINGS['halfwidth_exponent'];assert(c-h,c+h)==(Fraction(-1,4096),Fraction(63,4096))

def test_actual_entrypoint_and_shared_exclusive_path():
 assert R.TARGET==L.C.TARGET==R.BASE/'attempt003-target/run'
 q=subprocess.run([sys.executable,str(ROOT/'tools/launch_native_unit_pair_pal_repair.py'),'--help'],text=True,capture_output=True,timeout=15);assert q.returncode==0,q.stderr
 assert 'probe_native_unit_pair_pal_repair.py'in (ROOT/'tools/launch_native_unit_pair_pal_repair.py').read_text()

def test_original_profile_kernel_sources_retained():
 assert set(R.C.P.SOURCES)<=set(R.SOURCES) and R.SETTINGS['precision']==256 and R.SETTINGS['radius_exponent']==-10
