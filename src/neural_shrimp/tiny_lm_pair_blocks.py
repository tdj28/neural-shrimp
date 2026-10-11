"""Adjacent-pair-preserving order intervention in a fixed full-state response."""
import math
import numpy as np
from . import tiny_lm_parity_order as H
need,vector=H.need,H.vector
U=tuple(k for start in range(14,-1,-2)for k in(start,start+1));V=U[::-1]
ORDERS=dict(U=U,V=V)
TARGET_COUNTS=dict(hvp=128);PROFILE_COUNTS=dict(hvp=2)
GATES=dict(transpose=1e-8,decomposition=1e-12,material=.20,attenuation=.10)
#64 native-state copies+4 initial directions+128 new trajectory rows+128
#returned HVPs+12 derived contrasts+20 explicitly retained old P/R/frozen/CE/E.
VECTOR_BOUND=356

def propagate(states,u,eta,key,hvp,save,check):
 states=np.asarray(states,dtype=np.float64);u=vector(u)
 need(states.shape==(17,len(u))and np.isfinite(states).all()and math.isfinite(eta)and eta>0 and key in ORDERS,'literal reference and pair-block order')
 trace=[u.copy()];indices=[]
 try:
  for index in ORDERS[key]:
   check();action=vector(hvp(states[index],trace[-1]),u.shape);trace.append(vector(trace[-1]-eta*action,u.shape));indices.append(index)
 finally:save(np.asarray(trace),indices)
 return np.asarray(trace)

def compare(u,p,r,x,y,frozen,ce_gradient,parity_even,save=lambda *_:None):
 parity_even=vector(parity_even);p=vector(p,parity_even.shape)
 baseline=float(np.linalg.norm(parity_even));need(baseline>0 and math.isfinite(baseline),'qualified nonzero prior parity contrast')
 names=dict(chronological_odd='chronological_odd',grouped_odd='pair_block_odd',parity_even='pair_block_even')
 out=H.compare(u,p,r,x,y,frozen,ce_gradient,lambda key,value:save(names[key],value));out['products']={dict(P='P',R='R',Q='U',T='V')[k]:v for k,v in out['products'].items()}
 out['pair_block_odd_norm']=out.pop('grouped_odd_norm');out['pair_block_minus_ordered_squared']=out.pop('grouped_minus_ordered_squared');out['even_relative_parity_even']=out['even_norm']/baseline;out['prior_parity_even_norm']=baseline;out['gates']=dict(GATES)
 need(math.isfinite(out['even_relative_parity_even']),'finite attenuation');return out

def descriptive(cases):
 need([(c['seed'],c['rate_role'])for c in cases]==[(s,r)for s in(28161,28162)for r in('original','quarter_step')],'all four literal consumed cases')
 qualified=all(c['qualified']for c in cases);primary=[c['comparison']for c in cases if c['rate_role']=='original']
 return dict(all_four_qualified=qualified,material_in_both_originals=bool(qualified and all(c['even_relative_ordered']>=GATES['material']for c in primary)),tenfold_attenuation_in_both_originals=bool(qualified and all(c['even_relative_parity_even']<=GATES['attenuation']for c in primary)))

def forecast(times,return_bytes,old_max,already_bytes=0):
 H.forecast(times,return_bytes,old_max,already_bytes)
 maximum=max(old_max,*times)
 return dict(seconds=2*128*maximum+30,bytes=2*VECTOR_BOUND*max(return_bytes)+64*1024**2+already_bytes,effective_max_hvp_seconds=maximum,old_max_hvp_seconds=old_max,new_max_hvp_seconds=max(times),vectors=VECTOR_BOUND)
