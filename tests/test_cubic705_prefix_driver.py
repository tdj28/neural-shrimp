import sys,copy,json
from pathlib import Path
from flint import arb,ctx
import pytest
sys.path[:0]=[str(Path(__file__).parents[1]/'tools/reviews'),str(Path(__file__).parents[1]/'tools')]
import review_cubic705_prefix as W
import launch_cubic705_prefix_review as L
from neural_shrimp.rigorous_orbits import vector_json

def test_exact_card_source_closure_and_owned_bounds():
 card=json.loads((W.ROOT/W.CARD).read_text());assert len(card['producer']['workers'])==3 and W.OUT.parent==L.OUT
 assert all((W.ROOT/n).is_file()for n in W.FILES)
 source=Path(L.__file__).read_text();assert "'internal_seconds':900"in source and "'external_seconds':920"in source and '2050*1024**2'in source
 assert set(W.EXPECTED)=={'direction_return','cubic_proposal_input','cubic_proposal_return','cubic_affine_proposal_input','cubic_quadratic_proposal_input','pal_path_phase_return','pal_path_assembly_return','cubic_predictor_return','cubic_correlated_domain','gradient_path_input','gradient_path_return','gradient_path_assembly_return','cubic_path_input','cubic_path_return','cubic_path_assembly_return','cubic_correlated_chart_return'}

def test_full_actual_error_domain_not_decoder_inflation():
 with ctx.workprec(256):
  c=[arb(0)]*702+[arb(1),arb(1),arb(1)/64];z=[arb(0)]*705;rho=arb(1)/1024;error=[arb(0,rho)]*705
  p=dict(center=vector_json(c),slope=vector_json(z),curvature=vector_json(z),cubic=vector_json(z));d=dict(p,parameter='71/1024',halfwidth_exponent=-5,radius_exponent=-10,error=vector_json(error),whole_box=vector_json([c[i]+error[i]for i in range(705)]));cfg=dict(parameter='71/1024',halfwidth_exponent=-5,radius_exponent=-10,period=18,precision=256,newton_steps=2,predictor_degree=3,residual_degree=4);card=dict(normal=['0']*704+['1'],origin=['0']*705,weights=['1']*705)
  W.geometry(d,p,cfg,card)
  bad=copy.deepcopy(d);bad['error'][11]=vector_json([arb(0,rho/2)])[0]
  with pytest.raises(ValueError):W.geometry(bad,p,cfg,card)
  bad=copy.deepcopy(d);bad['whole_box'][4]=vector_json([arb(0)])[0]
  with pytest.raises(ValueError):W.geometry(bad,p,cfg,card)
