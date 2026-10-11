from pathlib import Path
from fractions import Fraction
import sys,copy,subprocess
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native_unit_pair_pal_batch as B

def test_fixed_four_adjacent_intervals_and_out_of_range_stop():
 prior=Fraction(63,4096)
 for k in range(1,5):
  s=B.settings(k);c=Fraction(s['parameter']);h=Fraction(2)**s['halfwidth_exponent'];assert c-h==prior;prior=c+h
 assert prior==Fraction(319,4096)
 for k in [0,5,1.0,True]:
  with pytest.raises(ValueError):B.settings(k)

def test_all705_seed_and_exact_twice_weights_no_assumed_anchor():
 with ctx.workprec(256):
  p={'parameter_center':'31/4096','center':[ball_json(arb(1))]*705,'slope':[ball_json(arb(2))]*705,'curvature':[ball_json(arb(4))]*705,'normal':['1']+['0']*704,'origin':['1']*705,'weights':['1/16']*705}
  z,n,o,w=B.seed(p,4);assert all(Fraction(x)==Fraction(1)+Fraction(2,16)+Fraction(2,256)for x in z)
  assert w==['1/8']*705 and n==p['normal']and o==p['origin']

def test_uniform_proof_never_claims_a_missing_anchor():
 p={'shooting_dimension':705,'dimension':13,'period':18,'gates':{'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'anchor_identified':False},'coordinate_inclusions':[True]*705}
 assert B.uniform_admission(p,True)and not B.uniform_admission(p,False)
 p['coordinate_inclusions'][704]=False;assert not B.uniform_admission(p,True)

def fixture():
 z=ball_json(arb(0));one=ball_json(arb(1));p={'period':18,'dimension':13,'parameter_endpoints':['-1/4096','63/4096'],'normal':['1']+['0']*704,'origin':['0']*705,'endpoint_root_boxes':[[z]*705,[z]*705]};c={'period':18,'dimension':13,'parameter_center':'95/4096','parameter_endpoints':['63/4096','127/4096'],'normal':p['normal'][:],'origin':p['origin'][:],'center':[z]*705,'slope':[z]*705,'curvature':[z]*705,'coordinate_radii':[one]*705};return p,c

def test_complete_exact_gauge_join_and_changed_gauge_rejected():
 p,c=fixture();r=B.ordered_join(p,c);assert r['passed']and len(r['coordinate_inclusions'])==705
 c['normal'][1]='1'
 with pytest.raises(ValueError,match='gauge'):B.ordered_join(p,c)

def test_gap_or_one_failed_native_coordinate_not_joined():
 p,c=fixture();c['parameter_endpoints'][0]='62/4096'
 with pytest.raises(ValueError,match='adjoining'):B.ordered_join(p,c)
 p,c=fixture();p['endpoint_root_boxes'][1][350]=ball_json(arb(2));assert not B.ordered_join(p,c)['passed']

def test_worker_entrypoint_and_fixed_parent_audit_paths():
 q=subprocess.run([sys.executable,str(ROOT/'tools/probe_native_unit_pair_pal_batch.py'),'--help'],text=True,capture_output=True,timeout=15);assert q.returncode==0,q.stderr
 assert '--unit'in q.stdout and B.AUDIT.name==B.JAUDIT.name=='attempt001'
 assert 'recomputed_endpoints'in (ROOT/'tools/probe_native_unit_pair_pal_batch.py').read_text()
