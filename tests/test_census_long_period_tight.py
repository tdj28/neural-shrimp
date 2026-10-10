import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path[:0]=[str(Path(__file__).parents[1]/'tools'),str(Path(__file__).parents[1]/'src')]
import certify_census_long_period_tight as T

def test_fixed640_radius_settings_and_exact_three_return_forecast():
 assert T.SETTINGS==dict(precision=640,radius_exponent=-288,newton_steps=0,max_power=4096)
 assert T.forecast([.1,.2,.3])==2*3*58*.3+30
 with pytest.raises(ValueError):T.forecast([.1,0,.3])

def test_explicit_newton_proposal_solves_coupled_affine_fixed_point():
 with ctx.workprec(640):
  M=arb_mat([[arb(1)/2,arb(1)/4],[0,arb(1)/4]]);center=[arb(3),arb(-2)];target=[arb(1),arb(2)];end=[target[i]+sum(M[i,j]*(center[j]-target[j])for j in range(2))for i in range(2)]
  residual,correction,proposal=T.propose(center,end,M)
  assert all(proposal[i].contains(target[i])for i in range(2))
  assert max(abs(v)for row in [[residual[i,0]for i in range(2)]]for v in row)>0
  with pytest.raises(ZeroDivisionError):T.propose(center,center,T.K.eye(2))

def test_exact_saved_center_and_wholebox_distance_reject_changed_seed():
 rows=[dict(mid=[1,-2],rad=[0,0])for _ in range(72)];assert T.exact_center(rows)==['1/4']*72
 assert T.proximity(rows,['1/4']*72)['maximum_normalized_distance']=='0'
 bad=[dict(mid=[1,-2],rad=[1,-10])]+rows[1:]
 with pytest.raises(ValueError):T.exact_center(bad)
 with pytest.raises(ValueError):T.proximity(bad,['1/4']*72)
