"""Full-state reversal-even parity-order contrast on fixed consumed LM states."""
import math
import numpy as np

P=tuple(range(16));R=P[::-1]
Q=tuple(range(0,16,2))+tuple(range(1,16,2));T=Q[::-1]
ORDERS=dict(P=P,Q=Q,T=T,R=R)
TARGET_COUNTS=dict(hvp=224);PROFILE_COUNTS=dict(hvp=2)
GATES=dict(replay=1e-10,transpose=1e-8,decomposition=1e-12,material=.20)
# Four*16 states, four distinct initial directions,14*16 subsequent path
# vectors,224 complete HVP returns, and3 derived vectors in each of4cases.
# Exact NPY object aliases avoid retaining a second dense trajectory copy.
VECTOR_BOUND=4*16+4+14*16+224+3*4


def need(ok, message):
    if not ok:raise ValueError(message)


def vector(a,shape=None):
    a=np.asarray(a,dtype=np.float64)
    need(a.ndim==1 and np.isfinite(a).all() and (shape is None or a.shape==shape),'finite full vector')
    return a


def relative(a,b):return float(np.linalg.norm(a-b))/max(float(np.linalg.norm(b)),1e-12)


def paths(role):
    need(role in ('original','quarter_step'),'fixed rate role')
    return ('P','Q','T') if role=='original' else ('P','Q','T','R')


def propagate(states,u,eta,key,hvp,save,check):
    states=np.asarray(states,dtype=np.float64);u=vector(u)
    need(states.shape==(17,len(u)) and np.isfinite(states).all() and math.isfinite(eta) and eta>0 and key in ORDERS,'literal full reference and permutation')
    trace=[u.copy()];indices=[]
    try:
        for index in ORDERS[key]:
            check();action=vector(hvp(states[index],trace[-1]),u.shape)
            trace.append(vector(trace[-1]-eta*action,u.shape));indices.append(index)
    finally:save(np.asarray(trace),indices)
    return np.asarray(trace)


def compare(u,p,r,q,t,frozen,ce_gradient,save=lambda *_:None):
    u,p,r,q,t,f,g=[vector(x)for x in(u,p,r,q,t,frozen,ce_gradient)]
    need(all(x.shape==u.shape for x in(p,r,q,t,f,g)),'same complete vector space')
    oddp=(p-r)/2;oddq=(q-t)/2;even=(q+t-p-r)/2
    # Retain exactly these three complete derived vectors; all remaining
    # vectors in the identity are literal retained product results.
    save('chronological_odd',oddp);save('grouped_odd',oddq);save('parity_even',even)
    odddiff=oddq-oddp;delta=q-p
    scale=max(*(float(np.linalg.norm(x))for x in(p,r,q,t)),1e-12)
    vector_error=float(np.linalg.norm(delta-even-odddiff))/scale
    ne,no,nd=[float(np.linalg.norm(x))for x in(even,odddiff,delta)]
    cross=float(np.dot(even,odddiff));squared=abs(nd*nd-ne*ne-no*no-2*cross)/max(scale*scale,nd*nd,ne*ne+no*no)
    transpose=[]
    for a,b in((p,r),(q,t)):
        s=max(float(np.linalg.norm(u))*max(float(np.linalg.norm(a)),float(np.linalg.norm(b))),1e-12)
        transpose.append(abs(float(np.dot(u,a-b)))/s)
    normp=max(float(np.linalg.norm(p)),1e-12)
    def measures(v):
        n=float(np.linalg.norm(v));return dict(gain=n,relative_to_ordered=relative(v,p),alignment=None if n==0 else float(np.dot(v,p))/(n*normp),training_ce_slope=float(np.dot(g,v)))
    report=dict(qualified=bool(max(transpose)<=GATES['transpose'] and vector_error<=GATES['decomposition'] and squared<=GATES['decomposition']),transpose_errors=transpose,vector_identity_error=vector_error,squared_identity_error=squared,even_norm=ne,even_relative_ordered=ne/normp,even_relative_frozen_error=ne/max(float(np.linalg.norm(p-f)),1e-12),odd_difference_norm=no,chronological_odd_norm=float(np.linalg.norm(oddp)),grouped_odd_norm=float(np.linalg.norm(oddq)),twice_cross_term=2*cross,grouped_minus_ordered_squared=nd*nd,even_training_ce_slope=float(np.dot(g,even)),products={k:measures(v)for k,v in [('P',p),('R',r),('Q',q),('T',t)]},gates=dict(GATES))
    need(all(math.isfinite(report[k])for k in('even_norm','even_relative_ordered','even_relative_frozen_error','twice_cross_term','squared_identity_error')),'finite full contrast')
    return report


def replay(fresh,saved):
    fresh,saved=np.asarray(fresh),np.asarray(saved)
    need(fresh.shape==saved.shape and fresh.ndim==2 and fresh.shape[0]==17 and np.array_equal(fresh[0],saved[0]),'literal chronological initial direction')
    rows=[dict(horizon=h,error=relative(vector(fresh[h]),vector(saved[h])))for h in(1,4,16)]
    return dict(passed=all(r['error']<=GATES['replay']for r in rows),rows=rows)


def descriptive_gate(cases):
    need([(x['seed'],x['rate_role'])for x in cases]==[(s,r)for s in(28161,28162)for r in('original','quarter_step')],'all four fixed cases')
    return bool(all(c['qualified']for c in cases) and all(c['comparison']['even_relative_ordered']>=GATES['material']for c in cases if c['rate_role']=='original'))


def forecast(times,return_bytes,old_max,already_bytes=0):
    need(len(times)==len(return_bytes)==2 and all(type(t)is float and math.isfinite(t)and t>0 for t in times)and all(type(n)is int and n==15776*8+128 for n in return_bytes),'two complete full-dimensional HVP costs')
    need(type(old_max)is float and math.isfinite(old_max)and old_max>0 and type(already_bytes)is int and already_bytes>=0,'literal complete historical maximum and retained bytes')
    maximum=max(old_max,*times)
    return dict(seconds=2*224*maximum+30,bytes=2*VECTOR_BOUND*max(return_bytes)+64*1024**2+already_bytes,effective_max_hvp_seconds=maximum,old_max_hvp_seconds=old_max,new_max_hvp_seconds=max(times),vectors=VECTOR_BOUND)
