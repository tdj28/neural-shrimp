"""Retrospective mean of adjacent-step Jacobian products, in full state.

B[k] = J[2k+1] J[2k]; predict h16 with (sum(B)/8)^8 u.
The blocks need not be symmetric or commute. This is not an online forecast.
"""
import math
import numpy as np
from .tiny_lm_temporal_mean import need,vector,relative

TARGET_COUNTS=dict(hvp=512)
PROFILE_COUNTS=dict(hvp=2)
GATES=dict(qualification=.01,vector_error=.20,improvement_factor=10.)
# Four cases:64 states,4 directions,32 powers,256 within-block vectors,
# 512 returned HVPs,256 block endpoints, and at most64 baseline/response vectors.
# Round upward, then double the complete vector footprint in the forecast.
VECTOR_BOUND=1280

def power(states,u,eta,hvp,save,check):
    states=np.asarray(states,dtype=np.float64);u=vector(u)
    need(states.shape==(17,len(u))and np.isfinite(states).all()and math.isfinite(eta)and eta>0,'complete fixed path and positive rate')
    trace=[u.copy()]
    try:
        for j in range(8):
            current=trace[-1];ends=[]
            try:
                for k in range(8):
                    check()
                    a=vector(hvp(states[2*k],current,j,k,0),u.shape)
                    mid=vector(current-eta*a,u.shape)
                    b=vector(hvp(states[2*k+1],mid,j,k,1),u.shape)
                    ends.append(vector(mid-eta*b,u.shape))
                # Every block acts on exactly the same current vector.
                trace.append(vector(np.sum(np.asarray(ends),axis=0)/8,u.shape))
            finally:save(f'power-{j}-block-ends',np.asarray(ends).reshape(-1,len(u)))
    finally:save('block-mean-powers',np.asarray(trace))
    return np.asarray(trace)

def compare(pred,old,responses,ce_slopes):
    fine=vector(responses[1]);coarse=vector(responses[0],fine.shape);pred=vector(pred,fine.shape)
    need(set(old)=={'ordered','mean','frozen','direction','ce_gradient'},'literal complete comparison set')
    old={k:vector(v,fine.shape)for k,v in old.items()}
    need(len(ce_slopes)==2 and all(math.isfinite(x)for x in ce_slopes),'two finite nonlinear CE slopes')
    scale=max(abs(ce_slopes[1]),1e-8);gain=float(np.linalg.norm(fine));need(gain>0,'nonzero resolved response')
    metrics={}
    for name,value in dict(block_mean=pred,**{k:old[k]for k in ('ordered','mean','frozen','direction')}).items():
        norm=float(np.linalg.norm(value));slope=float(np.dot(old['ce_gradient'],value))
        metrics[name]=dict(vector_error=relative(value,fine),gain=norm,signed_alignment=None if norm==0 else float(np.dot(value,fine))/(norm*gain),training_ce_slope=slope,training_ce_slope_error=abs(slope-ce_slopes[1])/scale)
    fd=relative(coarse,fine);ce=abs(ce_slopes[0]-ce_slopes[1])/scale
    qualified=fd<=.01 and metrics['ordered']['vector_error']<=.01 and ce<=.01 and metrics['ordered']['training_ce_slope_error']<=.01
    return dict(qualified=bool(qualified),response_gain=gain,finite_difference_disagreement=fd,ce_slope_disagreement=ce,ce_slopes=list(ce_slopes),comparisons=metrics,gates=dict(GATES))

def descriptive(cases):
    need([(r['seed'],r['rate_role'])for r in cases]==[(s,r)for s in(28161,28162)for r in('original','quarter_step')],'all four fixed consumed cases')
    qualified=all(r['qualified']for r in cases)
    values=[r['comparison']['comparisons']for r in cases if r['rate_role']=='original']
    return dict(all_four_qualified=qualified,accurate_in_both_originals=bool(qualified and all(v['block_mean']['vector_error']<.2 for v in values)),tenfold_better_than_single_step_mean_in_both=bool(qualified and all(10*v['block_mean']['vector_error']<v['mean']['vector_error']for v in values)))

def forecast(times,sizes,old_max,already_bytes=0):
    need(len(times)==len(sizes)==2 and all(math.isfinite(t)and t>0 for t in [old_max,*times])and all(type(v)is int and v>0 for v in sizes),'two complete native costs plus retained cold floor')
    maximum=max(old_max,*times)
    return dict(seconds=2*512*maximum+30,bytes=2*VECTOR_BOUND*max(sizes)+64*1024**2+already_bytes,effective_max_hvp_seconds=maximum,old_max_hvp_seconds=old_max,new_max_hvp_seconds=max(times),vectors=VECTOR_BOUND)
