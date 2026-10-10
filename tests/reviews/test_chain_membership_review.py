import sys,copy
from pathlib import Path
from fractions import Fraction as F
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import review_chain_membership_math as R

def b(m=0,r=0):return{'mid':[m,0],'rad':[r,0]}
def chart():
 c={'period':1,'dimension':1,'sigma':1,'shooting_dimension':4,'normal':['0','0','1','0'],'origin':['0']*4,'center':[b(0),b(1),b(2),b(3)],'slope':[b(1),b(),b(1),b()],'curvature':[b(2),b(),b(),b()],'parameter_center':'2','parameter_endpoints':['1','3'],'settings':{'halfwidth_exponent':0,'radius_exponent':-3},'weights':['1']*4}
 l={k:c[k]for k in ('period','dimension','sigma','shooting_dimension','normal','origin')};l['box']=copy.deepcopy(c['center']);return l,c

def test_exact_square_includes_zero_and_positive_domains():
 assert R.square((F(-2),F(3)))==(0,9)
 assert R.square((F(-3),F(-2)))==(4,9)
 assert R.square((F(2),F(3)))==(4,9)

def test_curved_predictor_exact_membership_and_strict_failure():
 l,c=chart();assert R.check_membership(l,c)['passed']
 l['box'][0]={'mid':[1,-3],'rad':[0,0]};assert not R.check_membership(l,c)['passed']

def test_unknown_gauge_interval_cannot_be_replaced_by_midpoint():
 l,c=chart();l['box'][2]['rad']=[1,-2];r=R.check_membership(l,c)
 assert r['parameter_inside']and not r['passed']

@pytest.mark.parametrize('key,value',[('normal',['1','0','0','0']),('sigma',-1),('period',2)])
def test_wrong_native_gauge_or_phase_convention(key,value):
 l,c=chart();l[key]=value
 with pytest.raises(ValueError):R.check_membership(l,c)

def test_native_domain_chain_and_radius_inflation_guard():
 root=[b(0),b(2),b(3),b(1),b(0)];l,c=chart();l['schema']='neural-shrimp-conditional-chain-lift-v1'
 phases=[{'index':0,'input':[b(0),b(1),b(2),b(3)],'value':[[b(0)],[b(1)]]}]
 assert R.check_phase_domains(root,l,phases,n=1,period=1)['passed']
 phases[0]['input'][0]['rad']=[1,-20]
 with pytest.raises(ValueError,match='covers actual'):R.check_phase_domains(root,l,phases,n=1,period=1)

def test_phase_input_must_cover_original_root_and_both_rates():
 root=[b(0),b(2),b(3),b(1),b(0)];l,c=chart();l['schema']='neural-shrimp-conditional-chain-lift-v1'
 for k in range(4):
  p={'index':0,'input':[b(0),b(1),b(2),b(3)],'value':[[b(0)],[b(1)]]};p['input'][k]['mid'][0]+=1
  with pytest.raises(ValueError,match='preceding'):R.check_phase_domains(root,l,[p],n=1,period=1)

def test_nonfinite_negative_radius_or_inexact_predictor_rejected():
 l,c=chart();c['slope'][0]['rad']=[1,-20]
 with pytest.raises(ValueError,match='exact tube'):R.check_membership(l,c)
 for v in [{'finite':False},b(0,-1)]:
  with pytest.raises(ValueError):R.ball(v)


def test_nonzero_curved_predictor_keeps_half_factor():
 l,c=chart();l['box'][2]={'mid':[9,-2],'rad':[0,0]};l['box'][0]={'mid':[5,-4],'rad':[0,0]}
 assert R.check_membership(l,c)['passed']
 c['curvature'][0]=b(10)
 assert not R.check_membership(l,c)['passed']

def test_full_second_phase_domain_is_bound_to_prior_output():
 root=[b(0),b(2),b(3),b(1),b(0)];l,_=chart();l.update(schema='neural-shrimp-conditional-chain-lift-v1',period=2,shooting_dimension=6,box=[b(0),b(1),b(5),b(6),b(2),b(3)])
 phases=[{'index':0,'input':[b(0),b(1),b(2),b(3)],'value':[[b(5)],[b(6)]]},{'index':1,'input':[b(5),b(6),b(2),b(3)],'value':[[b(0)],[b(1)]]}]
 assert R.check_phase_domains(root,l,phases,n=1,period=2)['passed']
 phases[1]['input'][0]=b(4)
 with pytest.raises(ValueError,match='preceding'):R.check_phase_domains(root,l,phases,n=1,period=2)

def test_full470_synthetic_lift_and_membership():
 n=13;p=18;root=[b(i)for i in range(41)];root[13]=b(2);root[14]=b(3)
 phase=root[:n]+root[n+2:2*n+2];flat=phase*p+[b(2),b(3)]
 normal=['0']*470;normal[-2]='1'
 l={'schema':'neural-shrimp-conditional-chain-lift-v1','dimension':n,'period':p,'sigma':1,'shooting_dimension':470,'normal':normal,'origin':['0']*470,'box':flat}
 c={**l,'center':copy.deepcopy(flat),'slope':[b()for _ in flat],'curvature':[b()for _ in flat],'parameter_center':'2','parameter_endpoints':['1','3'],'settings':{'halfwidth_exponent':0,'radius_exponent':-3},'weights':['1']*470};c['slope'][-2]=b(1)
 phases=[{'index':k,'input':phase+[b(2),b(3)],'value':[[v]for v in phase]}for k in range(p)]
 assert R.check_phase_domains(root,l,phases)['passed']
 assert R.check_membership(l,c)['coordinate_inclusions']==[True]*470
