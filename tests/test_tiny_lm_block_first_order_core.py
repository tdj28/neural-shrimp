import sys
sys.path[:0]=['/tmp/block_first_order_independent_core','/tmp/block_mean_independent_core','/home/d7082791602/.codex/worktrees/post-edge-next-steps/neural-shrimp/tools']
import numpy as np,pytest
import review_tiny_lm_block_first_order_core as C

def fixture():
 rng=np.random.default_rng(783);H=rng.normal(size=(16,5,5))*.3;e=.08;states=np.tile(np.arange(17)[:,None],(1,5));u=rng.normal(size=5);powers=[u];diagonal=[]
 for j in range(8):
  ends=[]
  for k in range(8):m=powers[-1]-e*(H[2*k]@powers[-1]);ends.append(m-e*(H[2*k+1]@m))
  diagonal.append(ends[j]);powers.append(np.sum(ends,axis=0)/8)
 powers=np.asarray(powers);diagonal=np.asarray(diagonal);f=diagonal-powers[1:];cor=[np.zeros(5),f[0]];ends={}
 for j in range(1,8):
  row=[]
  for k in range(8):m=cor[-1]-e*(H[2*k]@cor[-1]);row.append(m-e*(H[2*k+1]@m))
  ends[j]=np.asarray(row);cor.append(np.sum(row,axis=0)/8+f[j])
 return H,e,states,powers,diagonal,f,np.asarray(cor),ends

def test_all112_nonnormal_actions_and_wrong_current_reject():
 H,e,s,p,d,f,c,ends=fixture();calls=[]
 def action(label,kind,state,direction):
  j=int(label.split('/correction-')[1].split('/')[0]);k=int(label.split('/block-')[1].split('/')[0]);phase=int(label[-1]);assert kind=='hvp';np.testing.assert_array_equal(state,s[2*k+phase]);expected=c[j]if phase==0 else c[j]-e*(H[2*k]@c[j]);np.testing.assert_array_equal(direction,expected);calls.append(label);return H[2*k+phase]@direction
 assert C.replay_corrections(s,p,d,f,c,p[-1]+c[-1],e,lambda j:ends[j],action,'case',lambda:None)<1 and len(calls)==112
 broken=f.copy();broken[2,0]+=.001
 with pytest.raises(ValueError):C.replay_corrections(s,p,d,broken,c,p[-1]+c[-1],e,lambda j:ends[j],action,'case',lambda:None)
 bad={k:v.copy()for k,v in ends.items()};bad[1][0,1]+=.001
 with pytest.raises(ValueError):C.replay_corrections(s,p,d,f,c,p[-1]+c[-1],e,lambda j:bad[j],action,'case',lambda:None)

def test_independent_dense_duhamel_identity():
 H,e,s,p,d,f,c,ends=fixture();I=np.eye(5);B=[(I-e*H[2*k+1])@(I-e*H[2*k])for k in range(8)];A=sum(B)/8;expected=sum((np.linalg.matrix_power(A,7-j)@(B[j]-A)@np.linalg.matrix_power(A,j)@p[0]for j in range(8)))
 np.testing.assert_allclose(c[-1],expected,rtol=2e-13,atol=2e-14)

def test_fsum_cancellation_zero_action_and_mutation():
 ends=np.tile(np.array([1e12,1.,-1e12,3.,2.,-2.,5.,7.])[:,None],(1,3));f=np.array([1.,2.,3.]);saved=np.sum(ends,axis=0)/8+f;assert C.correction_sum(ends,f,saved)<=1
 with pytest.raises(ValueError):C.correction_sum(ends,f,saved+1.)
 H,e,s,p,d,f,c,ends=fixture();bad=c.copy();bad[0,0]=.01
 with pytest.raises(ValueError):C.replay_corrections(s,p,d,f,bad,p[-1]+c[-1],e,lambda j:ends[j],None,'case',lambda:None)

def test_accuracy_improvement_are_separate_and_allfour_required():
 rows=[dict(seed=s,rate_role=r,qualified=True,comparison=dict(comparisons=dict(first_order=dict(vector_error=1.))),block_mean_comparison=dict(comparisons=dict(block_mean=dict(vector_error=4.))))for s in(28161,28162)for r in('original','quarter_step')]
 assert C.gates(rows)==dict(all_four_qualified=True,accurate_in_both_originals=False,twice_better_than_block_mean_in_both=True)
 rows[-1]['qualified']=False;assert not any(C.gates(rows).values())
 from pathlib import Path
 source=Path(C.__file__).read_text();assert 'import explore_'not in source and 'neural_shrimp'not in source
