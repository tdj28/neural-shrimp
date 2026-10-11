import importlib.util
from pathlib import Path
import pytest
from flint import ctx,arb
from neural_shrimp.rigorous_generalized_chain import augmented_chain
p=Path(__file__).parents[2]/'tools/reviews/review_chain_unfolding_independent.py';s=importlib.util.spec_from_file_location('independent_coefficients',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Analytic normal-form fixture supplies a nonlinear shear and a physical-rate
# dependent state shift; its known coefficients are not acquired from producer.
p=Path(__file__).parents[1]/'test_rigorous_chain_unfolding.py';s=importlib.util.spec_from_file_location('normal_form_fixture',p);f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
@pytest.mark.parametrize('shear,mix',[(False,False),(True,False),(True,True)])
def test_known_coupled_and_state_shifted_map(shear,mix):
 with ctx.workprec(512):
  z=f.box();_,j,_=augmented_chain(f.NormalForm(shear=shear,mix=mix),z,1);r=m.reduce_coefficients(z,j)
  for k,v in [('a',4),('b',7),('s',12),('unfolding_determinant',5 if mix else 1)]:assert m.dec(r[k]).contains(v)
  assert r['quadratic_nondegenerate']and r['resonant_nondegenerate']and r['unfolding_transverse']
@pytest.mark.parametrize('kwargs,gate',[({'A':0},'quadratic_nondegenerate'),({'B':4},'resonant_nondegenerate'),({'rank':False},'unfolding_transverse')])
def test_degenerate_maps_do_not_pass(kwargs,gate):
 z=f.box();_,j,_=augmented_chain(f.NormalForm(**kwargs),z,1);assert not m.reduce_coefficients(z,j)[gate]
def test_wrong_chain_and_negative_radius_rejected():
 z=f.box();z[-2]=arb(2);_,j,_=augmented_chain(f.NormalForm(),z,1)
 with pytest.raises(ValueError):m.reduce_coefficients(z,j)
 with pytest.raises(ValueError):m.dec({'mid':[0,0],'rad':[-1,-10]})
