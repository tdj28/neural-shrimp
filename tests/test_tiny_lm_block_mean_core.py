import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import numpy as np,pytest
import review_tiny_lm_block_mean_core as C

def fixture():
 rng=np.random.default_rng(145);H=rng.normal(size=(16,5,5));H=(H+H.transpose(0,2,1))*.1;eta=.03;states=np.tile(np.arange(17)[:,None],(1,5));u=rng.normal(size=5);trace=[u];ends=[]
 for j in range(8):
  row=[]
  for k in range(8):
   m=trace[-1]-eta*(H[2*k]@trace[-1]);row.append(m-eta*(H[2*k+1]@m))
  ends.append(np.asarray(row));trace.append(np.sum(row,axis=0)/8)
 return H,eta,states,u,np.asarray(trace),ends

def test_full_recurrence_nonnormal_pairs_and_corruption():
 H,e,s,u,t,ends=fixture();calls=[]
 def action(label,kind,state,direction):
  j=int(label.split('/power-')[1].split('/')[0]);k=int(label.split('/block-')[1].split('/')[0]);p=int(label[-1]);assert kind=='hvp';np.testing.assert_array_equal(state,s[2*k+p]);expected=t[j]if p==0 else t[j]-e*(H[2*k]@t[j]);np.testing.assert_array_equal(direction,expected);calls.append(label);return H[2*k+p]@direction
 assert C.replay_powers(s,u,e,t,lambda j:ends[j],action,'case',lambda:None)<1 and len(calls)==128
 altered=[a.copy()for a in ends];altered[2][3,1]+=.0001
 with pytest.raises(ValueError):C.replay_powers(s,u,e,t,lambda j:altered[j],action,'case',lambda:None)
 bad=t.copy();bad[3,0]+=.001
 with pytest.raises(ValueError):C.mean_of_eight(ends[2],bad[3])

def test_fsum_cancellation_and_complete_mean():
 ends=np.tile(np.asarray([1e12,1.,-1e12,3.,2.,-2.,5.,7.])[:,None],(1,3));saved=np.sum(ends,axis=0)/8;assert C.mean_of_eight(ends,saved)<=1
 with pytest.raises(ValueError):C.mean_of_eight(ends[:7],saved)
 with pytest.raises(ValueError):C.mean_of_eight(np.full((8,3),np.nan),saved)

def test_comparison_separate_improvement_failure_and_unqualified_case():
 v=np.asarray([1.,2.]);old=dict(ordered=v.copy(),mean=100*v,frozen=10*v,direction=v.copy(),ce_gradient=np.array([3.,4.]));slopes=[11.,11.];row=C.comparison(2*v,old,[v,v],slopes);assert row['qualified']and row['comparisons']['block_mean']['vector_error']==1
 cases=[dict(seed=s,rate_role=r,qualified=True,comparison=row)for s in(28161,28162)for r in('original','quarter_step')];g=C.gates(cases);assert g==dict(all_four_qualified=True,accurate_in_both_originals=False,tenfold_better_than_single_step_mean_in_both=True)
 cases[-1]['qualified']=False;assert not any(C.gates(cases).values())

def test_no_producer_model_or_mathematics_import():
 from pathlib import Path
 text=Path(C.__file__).read_text();assert 'import review_tiny_lm_pair_blocks as B'in text and 'import explore_'not in text and 'neural_shrimp'not in text
