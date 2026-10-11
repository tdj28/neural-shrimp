import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_junction_lift import sharpen_known_root,lift_known_junction,contained
from neural_shrimp.rigorous_orbits import vector_json,eye
from neural_shrimp.rigorous_unit_pair import entire_pair
from neural_shrimp.rigorous_orbits import exact

def fixture():
 c,e,d=entire_pair(exact('1/10'))
 class Rotation:
  n=4
  def cusp_step(self,z,h,o):
   x,y,v,w=z;return[o*(c*x+e*y),o*(-d*x+c*y),-h*v+v*v*v,w*arb('0.2')]
 y=list(map(exact,['0']*4+['1','1']+['1','0','0','0']+['0','1','0','0']+['1/10']+['0','0','1','0']))
 return Rotation(),y


def test_known_root_intersections_preserve_nonlinear_exact_root():
 with ctx.workprec(384):
  c=[arb(1)];box=vector_json([arb(1,arb(1)/8)]);J=arb_mat([[arb(2,arb(1)/4)]]);C=arb_mat([[arb(1)/2]])
  root,proof=sharpen_known_root(c,box,C,arb_mat([[0]]),J)
  assert proof['passed']and root[0].contains(1)and root[0].rad()<arb(1)/1000 and contained(root[0],box[0])


def test_refinement_rejects_original_domain_edge_not_outward_decoded_domain():
 with ctx.workprec(384):
  box=vector_json([arb(1,arb(1)/8)]);b=box[0];rad=arb(tuple(b['rad']));inflated=arb(arb(1),rad)
  assert inflated.rad()>rad
  c=arb(1)+rad+(inflated.rad()-rad)/2
  assert not contained(c,b)
  with pytest.raises(ValueError,match='original literal'):sharpen_known_root([c],box,eye(1),arb_mat([[0]]),eye(1))


def test_literal_native_unit_pair_packing_and_input_before_each_call():
 with ctx.workprec(384):
  model,box=fixture();events=[]
  flat,proof=lift_known_junction(model,box,1,observe=lambda k,v:events.append((k,v)))
  assert len(flat)==15 and proof['closure_contains_zero']and proof['normalization_contains_zero']
  assert flat[:4]==box[:4]and flat[4:8]==box[6:10]and flat[8:12]==box[10:14]and flat[-3:]==box[4:6]+[box[14]]
  assert [k for k,v in events]==['lift_native_input','direction_return','lift_native_input','direction_return']
  assert [v['direction']for k,v in events if k=='lift_native_input']==['q','r']


def test_native_failure_keeps_prior_input_and_bad_angle_rejects():
 with ctx.workprec(192):
  model,box=fixture();events=[]
  class Bad:
   n=4
   def cusp_step(self,*args):raise RuntimeError('retained native failure')
  with pytest.raises(RuntimeError):lift_known_junction(Bad(),box,1,observe=lambda k,v:events.append(k))
  assert events==['lift_native_input']
  box[14]=arb(0)
  with pytest.raises(ValueError,match='nonzero angle'):lift_known_junction(model,box,1)


def test_full_dimension_refinement_retains_all55_coordinates():
 with ctx.workprec(384):
  c=[arb(i)/64 for i in range(55)];box=vector_json([arb(v,arb(1)/1024)for v in c]);C=eye(55);J=eye(55);J[0,54]=arb(1)/4
  root,p=sharpen_known_root(c,box,C,arb_mat(55,1),J)
  assert len(root)==55 and all(x.contains(y)for x,y in zip(root,c))and p['dimension']==55
