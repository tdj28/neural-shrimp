from flint import arb,ctx
import pytest
from neural_shrimp.rigorous_cubic_proposal import refresh_cubic_proposal
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_correlated_tube
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.rigorous_eigenpair_march import decode
from test_rigorous_unit_pair import Polynomial,Primitive
from test_rigorous_unit_pair_pal import fixture


@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_refreshed_native_predictor_can_only_be_promoted_by_full_new_proof(model,period):
 with ctx.workprec(256):
  ini,kw=fixture(model,period);N=len(ini);normal=[exact(v)for v in kw['normal']];origin=[arb(0)]*N;events=[];calls=[]
  data=refresh_cubic_proposal(model,[exact(v)for v in ini],period=period,normal=normal,origin=origin,parameter=exact('3/1024'),newton_steps=2,tick=lambda:calls.append(1),observe=lambda k,v:events.append((k,v)))
  assert len(calls)==10*period+5 and data['record']['native_calls']==10*period and data['record']['algebra_ticks']==5
  assert 'passed'not in data['record'] and events[0][0]=='cubic_proposal_input'
  proof=cubic_correlated_tube(model,data['center'],data['slope'],data['curvature'],data['cubic'],data['C0'],period=period,normal=normal,origin=origin,parameter='3/1024',halfwidth_exponent=-11,radius_exponent=-12,weights=[arb(1)]*N)
  assert proof['gates']['uniform_full_native_unit_pair_locus']and proof['gates']['primitive_period']and not proof['passed']and not proof['gates']['anchor_identified']
  if period==1:
   for t,end in zip([exact('5/2048'),exact('7/2048')],proof['endpoint_root_boxes']):
    with ctx.workprec(512):x=2*(t.sqrt().cos()-1);h=1-x*x/2;o=1-x
    assert decode(end[0]).contains(x)and decode(end[-3]).contains(h)and decode(end[-2]).contains(o)


def test_failed_native_attempt_keeps_complete_input_and_does_not_claim_proof():
 class Broken:
  n=2
  def cusp_step(self,*args):raise RuntimeError('synthetic interruption')
 ini,kw=fixture(Polynomial(),1);events=[]
 with pytest.raises(RuntimeError):refresh_cubic_proposal(Broken(),[exact(v)for v in ini],period=1,normal=[exact(v)for v in kw['normal']],origin=[arb(0)]*9,parameter=arb(0),observe=lambda k,v:events.append((k,v)))
 assert events[0][0]=='cubic_proposal_input'and len(events[0][1]['center'])==9
