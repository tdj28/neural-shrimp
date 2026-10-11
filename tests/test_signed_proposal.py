import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flint import arb,ctx
import pytest
from neural_shrimp.rigorous_signed_proposal import refresh_signed_proposal
from neural_shrimp.rigorous_orbits import exact
from neural_shrimp.native470_first_components import synthetic_fixture,SyntheticFlip

def test_full_two_phase_refresh_call_budget_and_finite_proposals():
 with ctx.workprec(192):
  ini,kw=synthetic_fixture(n=2,period=2);kw.pop('anchor');kw.pop('halfwidth_exponent');kw.pop('radius_exponent');kw.pop('precision');kw['normal']=[exact(v)for v in kw['normal']];kw['origin']=[exact(v)for v in kw['origin']];kw['parameter']=arb(0);calls=[0];ev=[]
  def tick():calls[0]+=1
  r=refresh_signed_proposal(SyntheticFlip(n=2),[exact(v)for v in ini],**kw,tick=tick,observe=lambda k,v:ev.append((k,v)))
  assert calls[0]==8+4 and r['record']['native_calls']==8 and r['record']['algebra_ticks']==4
  assert len([1 for k,v in ev if k=='signed_proposal_input'])==3 and len(r['curvature'])==10
  assert all(v.is_exact()for k in('center','slope','curvature')for v in r[k])

def test_seed_retained_before_failure_and_no_proof_flag():
 class Bad:
  n=1
  def cusp_step(self,*_):raise RuntimeError('native failed')
 events=[]
 with pytest.raises(RuntimeError,match='native failed'):refresh_signed_proposal(Bad(),[arb(1)]*4,period=1,sigma=-1,normal=[arb(1)]*4,origin=[arb(1)]*4,parameter=arb(0),observe=lambda k,v:events.append((k,v)))
 assert events[0][0]=='signed_proposal_input'and events[0][1]['sigma']==-1
