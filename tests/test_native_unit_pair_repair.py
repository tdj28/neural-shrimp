import importlib.util,math
from fractions import Fraction
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
R=load('probe_native_unit_pair_repair');L=load('launch_native_unit_pair_repair')
def diagnostic():return {'scaling':{'iterations':512,'collatz_max':.59},'resolvent_proposal':{'nonnegative':True,'radii':[.001]*704}}

def test_exact_dyadic_physical_radius_scaling_and_width():
 d=diagnostic();w=R.repair_weights(d)
 assert len(w)==704 and all(Fraction(2)**-11*Fraction(v)==Fraction.from_float(.001)/8 for v in w)
 assert R.SETTINGS==dict(R.P.SETTINGS,parameter='1/65536',halfwidth_exponent=-16,radius_exponent=-11)
 assert R.P.SETTINGS['halfwidth_exponent']==-14
 assert R.BASE==L.BASE
@pytest.mark.parametrize('bad',[0.,-1.,math.nan,math.inf,1.,True])
def test_invalid_proposed_radius_cannot_reach_target(bad):
 d=diagnostic();d['resolvent_proposal']['radii'][13]=bad
 with pytest.raises(ValueError):R.repair_weights(d)

def test_incomplete_or_unstable_numerical_proposal_rejected():
 for key in ['count','nonnegative','iterations','collatz']:
  d=diagnostic()
  if key=='count':d['resolvent_proposal']['radii'].pop()
  elif key=='nonnegative':d['resolvent_proposal']['nonnegative']=False
  elif key=='iterations':d['scaling']['iterations']=511
  else:d['scaling']['collatz_max']=.96
  with pytest.raises(ValueError):R.repair_weights(d)

def test_fresh_kernel_and_original_704_anchor_not_replaced_by_old_diagnostic():
 source=(ROOT/'tools/probe_native_unit_pair_repair.py').read_text()
 assert 'unit_pair_tube(model,initial,**SETTINGS,weights=weights,anchor=anchor,tick=tick,observe=observe)'in source
 assert "'settings':P.SETTINGS,'python'"in source
 assert "args.out==BASE/'attempt002-target/run'"in source
