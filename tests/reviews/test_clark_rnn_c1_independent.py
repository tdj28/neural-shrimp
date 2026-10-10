import ast,sys
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
import pytest
from flint import arb,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
import review_clark_rnn_c1_math as R
from neural_shrimp.rigorous_clark_rnn import BallClarkRNN,check_rnn_orbit
from neural_shrimp.rigorous_orbits import ball_json

def data(j=(('2',),),x=('19/10',),p=1,g='1',d='1'):
 return {'schema':'clark-rnn-c1-input-v1','j0':[list(r)for r in j],'gain':g,'delta':d,'period':p,'approximation':list(x)}

def test_bound_python_entry_points_parse():
 import review_clark_rnn_c1 as driver
 root=Path(__file__).resolve().parents[2]
 for name in driver.FILES:
  if name.endswith('.py'):ast.parse((root/name).read_text(),filename=name)

@pytest.mark.parametrize('bits',[128,256,384])
def test_independent_leak_and_column_derivative_at_origin(bits):
 with ctx.workprec(bits):
  x=data(j=(('1/4','-3/4'),('1/2','1/8')),x=('0','0'),g='3/2',d='3/4');m=R.ExpRNN(x);y,a=m.step([arb(0),arb(0)])
  assert all(v==0 for v in y)
  for i in range(2):
   for k in range(2):assert a[i,k]==R.exact(Fraction(1,4)*int(i==k)+Fraction(9,8)*Fraction(x['j0'][i][k]))

@pytest.mark.parametrize('point',[['1/4','-1/8'],['-3','2'],['0','0']])
def test_independent_exp_map_overlaps_native_point_and_box(point):
 with ctx.workprec(384):
  d=data(j=(('1/4','-3/4'),('1/2','1/8')),x=point,g='3/2',d='3/4');m=R.ExpRNN(d);p=BallClarkRNN(d)
  for radius in (arb(0),arb((1,-20))):
   x=[arb(R.exact(v),radius)for v in point];a,j=m.step(x);b,k=p.step(x)
   assert all(a[i].overlaps(b[i])for i in range(2))
   assert all(j[i,l].overlaps(k[i,l])for i in range(2)for l in range(2))

def test_known_nonzero_stable_root_independent_and_nested():
 d=data();old=check_rnn_orbit(d);inner=check_rnn_orbit(d,precision=384,radius_exponent=-100)
 assert R.strict_nested(old['box'],inner['box']);assert R.independent_proof(d,inner)['passed']

@pytest.mark.parametrize('mode',['bad_inverse','uncentered','antipode_self','unstable','nonprimitive'])
def test_failed_proof_gates_not_rescued(mode):
 d=data()
 if mode=='unstable':d=data(x=('0',))
 if mode in('antipode_self','nonprimitive'):d=data(j=(('0','-2'),('2','0')),x=('19/10','19/10'),p=4 if mode=='antipode_self'else 8)
 proof=check_rnn_orbit(d,precision=384,radius_exponent=-100)
 if mode=='bad_inverse':proof['preconditioner']=[[ball_json(arb(0))]]
 if mode=='uncentered':
  proof['center'][0]=ball_json(arb(0))
  with pytest.raises(ValueError,match='center'):R.independent_proof(d,proof)
 else:assert not R.independent_proof(d,proof)['passed']

def test_strict_nesting_and_decoder_fail_closed():
 x={'mid':[1,0],'rad':[1,-20]};assert not R.strict_nested([x],[x])
 with pytest.raises(ValueError):R.restore({'mid':[1,0],'rad':[-1,-20]})
 with pytest.raises(ValueError):R.restore({'mid':[1.,0],'rad':[0,0]})
