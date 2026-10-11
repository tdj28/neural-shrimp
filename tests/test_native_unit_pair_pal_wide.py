import sys,subprocess
from pathlib import Path
from fractions import Fraction
from flint import arb,ctx
import pytest
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))

def module():
 # Adapter runs in its own process in production; restore inherited module globals for test isolation.
 import probe_native_unit_pair_pal_batch as b
 saved={k:getattr(b,k)for k in ['BASE','SOURCES','settings','seed','bind_parent']}
 import probe_native_unit_pair_pal_wide as w
 for k,v in saved.items():setattr(b,k,v)
 return w

def test_exact_four_wider_domains_and_no_fifth():
 w=module();right=Fraction(319,4096)
 for i in range(1,5):
  c=w.settings(i);h=Fraction(2)**c['halfwidth_exponent'];assert Fraction(c['parameter'])-h==right;right=Fraction(c['parameter'])+h
 assert right==Fraction(831,4096)
 with pytest.raises(ValueError):w.settings(5)

def test_three_quarter_radius_proposal_retains_all_native_coordinates():
 w=module();z=ball_json(arb(0))
 p={'parameter_center':'287/4096','center':[z]*705,'slope':[z]*705,'curvature':[z]*705,'weights':['1/8']*705,'normal':['1']+['0']*704,'origin':['0']*705}
 _,n,o,v=w.seed(p,1);assert len(v)==705 and v==['3/32']*705 and n==p['normal']and o==p['origin']

def test_wider_join_uses_correct_halfwidth_and_all705():
 w=module();z=ball_json(arb(0));one=ball_json(arb(1));previous={'period':18,'dimension':13,'parameter_endpoints':['255/4096','319/4096'],'normal':['1']+['0']*704,'origin':['0']*705,'endpoint_root_boxes':[[z]*705,[z]*705]}
 current={'period':18,'dimension':13,'parameter_center':'383/4096','parameter_endpoints':['319/4096','447/4096'],'normal':previous['normal'],'origin':previous['origin'],'center':[z]*705,'slope':[z]*705,'curvature':[z]*705,'coordinate_radii':[one]*705,'settings':{'precision':256,'newton_steps':2,'halfwidth_exponent':-6,'radius_exponent':-10}}
 assert w.ordered_join(previous,current)['passed']
 current['settings']['halfwidth_exponent']=-7
 with pytest.raises(ValueError):w.ordered_join(previous,current)

def test_fresh_adapter_entrypoint_and_source_closure():
 w=module();p=subprocess.run([sys.executable,str(ROOT/'tools/probe_native_unit_pair_pal_wide.py'),'--help'],capture_output=True,text=True,timeout=15);assert p.returncode==0,p.stderr
 assert all((ROOT/n).exists()for n in w.SOURCES)
 assert "'settings'"in (ROOT/'tools/launch_native_unit_pair_pal_wide.py').read_text()
