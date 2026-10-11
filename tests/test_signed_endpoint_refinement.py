import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_signed_endpoint_refinement import refine,fixed_parameter_domain
from neural_shrimp.rigorous_orbits import vector_json

def test_nonlinear_conditional_root_and_finite_newton_proposals():
 with ctx.workprec(128):
  C=arb_mat([[arb(1)/4]]);q=arb(1)/2;point=[arb(3)/2];known=[arb(3)/2+arb(0,arb(1)/2)];root=arb(2).sqrt()
  widths=[]
  for _ in range(3):
   residual=[point[0]*point[0]-2];box,r=refine(C,q,[arb(1)],point,residual,known);assert box[0].contains(root);widths.append(box[0].rad());known=box;point=[(point[0]-(C*arb_mat([[residual[0]]]))[0,0]).mid()]
  assert widths[-1]<widths[0]/8

def test_weighted_nonnormal_exact_root():
 with ctx.workprec(128):
  A=arb_mat([[1,4],[0,1]]);C=arb_mat([[1,-3],[0,1]]);point=[arb(1),arb(2)];root=arb_mat([[arb(9)/8],[arb(33)/16]]);b=A*root;res=A*arb_mat([[v]for v in point])-b
  box,r=refine(C,arb(1)/4,[arb(8),arb(1)],point,[res[i,0]for i in range(2)],[arb(1,1),arb(2,1)]);assert all(box[i].contains(root[i,0])for i in range(2))
  with pytest.raises(ValueError):refine(C,arb(1),[arb(8),arb(1)],point,[res[i,0]for i in range(2)],[arb(1,1),arb(2,1)])

def test_original_correction_domain_not_broad_whole_box():
 z=vector_json([arb(0)]);one=vector_json([arb(1)]);chart=dict(parameter_center='0',parameter_endpoints=['-1','1'],center=z,slope=one,curvature=z);domain=dict(error=vector_json([arb(0,arb(1)/16)]))
 assert fixed_parameter_domain(chart,domain,vector_json([arb(1)]),vector_json([arb(1)]),'1')
 with pytest.raises(ValueError):fixed_parameter_domain(chart,domain,vector_json([arb(1)]),vector_json([arb(0)]),'1')
