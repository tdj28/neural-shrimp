import sys,copy
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
import review_rotated705_math as A
from neural_shrimp.rigorous_orbits import vector_json,matrix_json

def test_nonnormal_known_root_and_intersections():
 with ctx.workprec(256):
  D=arb_mat([[arb(1)/8,arb(1)/4],[0,arb(1)/8]]);rhs=arb_mat([[arb(1)/1024],[-arb(1)/2048]])
  exact=(arb_mat([[1,0],[0,1]])-D).solve(rhs);e,q,r=A.localize(D,rhs,[arb(1),arb(1)],[arb(0,1)]*2)
  assert q<1 and all(e[i].contains(exact[i,0])for i in range(2)) and all(v.rad()<arb(1)/100 for v in e)

def test_noncontractive_rejected():
 with pytest.raises(ValueError):A.localize(arb_mat([[1]]),arb_mat([[1]]),[arb(1)],[arb(0,1)])

def test_literal_segment_excludes_decoder_inflation():
 p=[arb(1)];e=vector_json([arb(0,arb(1)/1024)])
 assert A.segment_in_domain(p,e,[arb(1)],[arb(1,arb(1)/2048)])
 with pytest.raises(ValueError):A.segment_in_domain(p,e,[arb(1)+arb(1)/512],[arb(1)])

def fixture():
 n=2;point=[arb(1)/8,arb(1)/4,arb(1),arb(0),arb(0),arb(1),arb(2),arb(3),arb(1)/64];normal=[arb(0)]*8+[arb(1)];origin=[arb(0)]*9
 t=point[-1];c=t.sqrt().cos();e=t.sqrt().sin()/t.sqrt();d=t*e
 outputs=[point[:2]+[c,-d],point[:2]+[e,c]]
 events=[dict(phase=0,direction=name,input=vector_json(point[:2]+point[mode:mode+2]+point[-3:-1]),value=matrix_json(arb_mat([[x]for x in out])))for name,mode,out in zip(('q','r'),(2,4),outputs)]
 return point,normal,origin,events

def test_full_residual_terminal_signs_and_input_binding():
 with ctx.workprec(256):
  p,n,o,e=fixture();v=A.residual(p,1,n,o,p[-1],e);assert all(v[i,0].contains(0)for i in range(9))
  bad=copy.deepcopy(e);bad[0]['input'][0]=vector_json([arb(0)])[0]
  with pytest.raises(ValueError):A.residual(p,1,n,o,p[-1],bad)
  bad=copy.deepcopy(e);bad[1]['value'][2][0]=vector_json([arb(0)])[0]
  assert not A.residual(p,1,n,o,p[-1],bad)[4,0].contains(0)


def test_new_width_requires_explicit_permission_and_exact_domain():
 import review_cubic705_prefix as W
 with ctx.workprec(256):
  c=[arb(0)]*702+[arb(1),arb(1),arb(1)/64];z=[arb(0)]*705;err=[arb(0,arb(1)/1024)]*705
  p=dict(center=vector_json(c),slope=vector_json(z),curvature=vector_json(z),cubic=vector_json(z))
  d=dict(p,parameter='31/2048',halfwidth_exponent=-6,radius_exponent=-10,error=vector_json(err),whole_box=vector_json([x+y for x,y in zip(c,err)]))
  cfg=dict(parameter='31/2048',halfwidth_exponent=-6,radius_exponent=-10,period=18,precision=256,newton_steps=2,predictor_degree=3,residual_degree=4)
  card=dict(normal=['0']*704+['1'],origin=['0']*705,weights=['1']*705)
  with pytest.raises(ValueError):W.geometry(d,p,cfg,card)
  assert W.geometry(d,p,cfg,card,allowed_widths=(-6,))[7]==arb(1)/64
  d['halfwidth_exponent']=-5
  with pytest.raises(ValueError):W.geometry(d,p,cfg,card,allowed_widths=(-6,))
