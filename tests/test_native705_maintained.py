import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_maintained as W
import launch_native705_maintained as L

def fixture():
 z=[ball_json(arb(0))]*705;s=[ball_json(arb(0))]*704+[ball_json(arb(1))]
 return dict(center=z,slope=s,curvature=z,cubic=z,normal=['0']*704+['1'],origin=['0']*705,weights=['1']*705,parameter_center='31/2048',parameter_endpoints=['-1/2048','63/2048'],endpoint_root_boxes=[z,z],settings=dict(halfwidth_exponent=-6,radius_exponent=-10))

def test_same_gauge_exact_scalar_and_changed_gauge_projection(tmp_path,monkeypatch):
 with ctx.workprec(256):
  proposal=fixture();anchor=L.anchor_from(proposal);p={'sha256':'p','bytes':1};monkeypatch.setattr(W,'entry',lambda _:p)
  c=L.make_card('source',1,-6,'same_gauge',{},proposal,anchor,{}, {},p)
  assert c['settings']['parameter']=='47/1024'and c['anchor']['exact_parameter']=='63/2048'and c['normal']==proposal['normal']
  assert W.validate_card(c,'source',1)
  c=L.make_card('source',1,-6,'rotated_gauge',{},proposal,anchor,{}, {},p)
  assert c['settings']['parameter']=='31/2048'and'exact_parameter'not in c['anchor'];assert W.validate_card(c,'source',1)
  c['anchor']['exact_parameter']='0'
  with pytest.raises(ValueError):W.validate_card(c,'source',1)

def test_wrong_exact_gauge_rejected_before_join():
 p=fixture();a=L.anchor_from(p);a['normal']=['1']+['0']*704
 with pytest.raises(ValueError):W.join(a,p)

def test_actual_entrypoints_paths_and_source_closure():
 assert L.BASE==W.BASE and L.LIMITS['attempts']==32 and L.LIMITS['maximum_halfwidth']==-3
 for n in W.SOURCES:assert(ROOT/n).is_file()
 for name in ['probe_native705_maintained.py','launch_native705_maintained.py']:
  p=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True);assert p.returncode==0 and'usage:'in p.stdout

def test_source_parent_matches_committed_receipt():
 r=json.loads((ROOT/'docs/reviews/native705-cubic-endpoint-point001.json').read_text());assert L.PARENT['root']==r['root']and L.PARENT['source']==r['source']and L.PARENT['outer']==r['outer']['sha256']and L.PARENT['worker']==r['worker']['sha256']and L.PARENT['endpoint']==r['connected_endpoint']['sha256']
 assert(Path(L.PARENT['root'])/'run/connected-endpoint.json').is_file()


def test_policy_uses_literal_upper_bound_not_inexact_midpoint():
 from fractions import Fraction as F
 b={'mid':[1,-3],'rad':[0,0]};d={'contraction':b};c={'bound':{'contraction':b}}
 assert L.contraction_upper(d,c)==F(1,8)
 with pytest.raises(ValueError):L.contraction_upper({'contraction':{'mid':[1,-3],'rad':[1,-20]}},c)
 with pytest.raises(ValueError):L.contraction_upper({'contraction':{'mid':[1,-4],'rad':[0,0]}},c)
