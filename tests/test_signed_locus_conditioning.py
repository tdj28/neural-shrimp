import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flint import arb
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.signed_locus_conditioning import budget

def encoded(rows):return [[ball_json(arb(x))for x in row]for row in rows]
def fixture():return dict(shooting_dimension=2,preconditioner=encoded([[1,0],[0,1]]),defect=encoded([[0,.25],[.125,0]]),center_jacobian=encoded([[1,0],[0,1]]),condensed_border=encoded([[1,0],[0,2]]),quadratic_center_coefficients=encoded([[.01,.1,.2,99],[0,0,0,99]]),quadratic_whole_coefficients=encoded([[0,0,0,.4],[0,0,0,0]]),coordinate_radii=[ball_json(arb(1))]*2,settings=dict(halfwidth_exponent=-1))
def test_order_convention_and_nonnormal_scaling():
 r=budget(fixture());assert all(abs(a-b)<1e-12 for a,b in zip(r['forcing_order_max_ratio'],[.01,.05,.05,.05]));assert r['absolute_row_max']==.25;assert r['scaling']['collatz_max']<.3
 assert r['resolvent']['nonnegative']and r['condensed_singular_values']==[2.,1.]
def test_unstable_nonnegative_matrix_has_no_resolvent():
 c=fixture();c['defect']=encoded([[1.1,0],[0,1.2]]);r=budget(c);assert 'resolvent'not in r and r['scaling']['collatz_max']>1
