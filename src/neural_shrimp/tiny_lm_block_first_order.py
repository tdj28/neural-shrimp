"""First derivative of an ordered block product about its constant-block path.

B_i(epsilon)=Bbar+epsilon(B_i-Bbar). The returned correction is d/depsilon
[B_7(epsilon)...B_0(epsilon)u] at zero, not the exact ordered response.
"""
import math
import numpy as np
from .tiny_lm_temporal_mean import need,vector,relative
TARGET_COUNTS=dict(hvp=448)
PROFILE_COUNTS=dict(hvp=2)
VECTOR_BOUND=1600
GATES=dict(qualification=.01,vector_error=.20,improvement_factor=2.)

def correct(states,powers,diagonal,eta,hvp,save,check):
    states=np.asarray(states,dtype=np.float64);powers=np.asarray(powers,dtype=np.float64);diagonal=np.asarray(diagonal,dtype=np.float64)
    need(states.ndim==2 and states.shape[0]==17 and powers.shape==(9,states.shape[1])and diagonal.shape==(8,states.shape[1]),'complete block mean path and diagonal actions')
    need(all(np.isfinite(v).all()for v in(states,powers,diagonal))and math.isfinite(eta)and eta>0,'finite complete path and rate')
    correction=[np.zeros(states.shape[1],dtype=np.float64)];forcing=diagonal-powers[1:]
    save('first-order-forcing',forcing)
    try:
        for j in range(8):
            check();ends=[]
            if j==0:
                # The exact linear action on zero needs no native evaluation.
                propagated=np.zeros(states.shape[1],dtype=np.float64)
            else:
                try:
                    for k in range(8):
                        check();a=vector(hvp(states[2*k],correction[-1],j,k,0),correction[-1].shape)
                        mid=vector(correction[-1]-eta*a,correction[-1].shape)
                        b=vector(hvp(states[2*k+1],mid,j,k,1),correction[-1].shape)
                        ends.append(vector(mid-eta*b,correction[-1].shape))
                    propagated=vector(np.sum(np.asarray(ends),axis=0)/8,correction[-1].shape)
                finally:save(f'correction-{j}-block-ends',np.asarray(ends).reshape(-1,states.shape[1]))
            correction.append(vector(propagated+forcing[j],correction[-1].shape))
    finally:save('first-order-corrections',np.asarray(correction))
    return powers[-1]+correction[-1]

def descriptive(cases):
    need([(v['seed'],v['rate_role'])for v in cases]==[(s,r)for s in(28161,28162)for r in('original','quarter_step')],'all four fixed consumed cases')
    qualified=all(v['qualified']for v in cases);original=[v for v in cases if v['rate_role']=='original']
    return dict(all_four_qualified=qualified,accurate_in_both_originals=bool(qualified and all(v['comparison']['comparisons']['first_order']['vector_error']<.2 for v in original)),twice_better_than_block_mean_in_both=bool(qualified and all(2*v['comparison']['comparisons']['first_order']['vector_error']<v['block_mean_comparison']['comparisons']['block_mean']['vector_error']for v in original)))

def forecast(times,sizes,old_max,already_bytes=0):
    need(len(times)==len(sizes)==2 and all(math.isfinite(t)and t>0 for t in[old_max,*times])and all(type(v)is int and v>0 for v in sizes),'two complete native costs plus entire retained cold floor')
    maximum=max(old_max,*times)
    return dict(seconds=2*448*maximum+30,bytes=2*VECTOR_BOUND*max(sizes)+64*1024**2+already_bytes,effective_max_hvp_seconds=maximum,old_max_hvp_seconds=old_max,new_max_hvp_seconds=max(times),vectors=VECTOR_BOUND)
