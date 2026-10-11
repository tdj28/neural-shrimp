"""Numerical simultaneous native +1/-1 root proposals, no rigorous claim."""
import numpy as np
from .signed_flip_scout import evaluate as signed_evaluate
from .unit_pair_scout import need


def augmented(model,y,period=18,tick=lambda:None,observe=lambda k,r:None):
    n=model.n;y=np.asarray(y,dtype=float);base=n+2;dim=3*n+2
    need(y.shape==(dim,)and np.isfinite(y).all(),'full plus/minus native dimensions')
    x=y[:n];h,o=y[n:n+2];q=y[base:base+n];w=y[base+n:]
    acquired=[]
    for name,direction in (('plus',q),('minus',w)):
        def record(kind,row):observe(kind,dict(direction=name,**row))
        value,J,dh,data=signed_evaluate(model,np.r_[x,o,direction],h,period,tick,record)
        acquired.append((value,J,dh,data))
    pv,pJ,ph,pd=acquired[0];mv,mJ,mh,md=acquired[1]
    need(np.array_equal(pd['orbit'],md['orbit'])and np.array_equal(pd['monodromy'],md['monodromy']),'same native map/phase in both critical directions')
    value=np.r_[pv[:n],pv[n:2*n]-2*q,mv[n:2*n],pv[-1],mv[-1]]
    J=np.zeros((dim,dim));J[:n,:n]=pJ[:n,:n];J[:n,n]=ph[:n];J[:n,n+1]=pJ[:n,n]
    for row,source,hidden,col,sign in ((n,pJ,ph,base,1),(2*n,mJ,mh,base+n,-1)):
        J[row:row+n,:n]=source[n:2*n,:n];J[row:row+n,n]=hidden[n:2*n];J[row:row+n,n+1]=source[n:2*n,n]
        J[row:row+n,col:col+n]=pd['monodromy']-sign*np.eye(n)
    J[-2,base:base+n]=q;J[-1,base+n:]=w
    need(np.isfinite(value).all()and np.isfinite(J).all(),'finite full native plus/minus return')
    return value,J,dict(orbit=pd['orbit'],q_phases=pd['modes'],w_phases=md['modes'],monodromy=pd['monodromy'],native_plus=pd['native_returns'],native_minus=md['native_returns'])


def qualify(y,value,data):
    n=data['monodromy'].shape[0];p=len(data['orbit'])-1;M=data['monodromy'];y=np.asarray(y);values,vectors=np.linalg.eig(M)
    plus=int(np.argmin(abs(values-1)));remaining=[j for j in range(n)if j!=plus];minus=min(remaining,key=lambda j:abs(values[j]+1));others=[j for j in remaining if j!=minus]
    residuals=[float(np.linalg.norm(M@vectors[:,j]-values[j]*vectors[:,j])/max(1.,np.linalg.norm(M)*np.linalg.norm(vectors[:,j])))for j in range(n)]
    divisors={str(k):float(max(abs(data['orbit'][k]-data['orbit'][0])))for k in range(1,p)if p%k==0}
    plus_gap=float(min(abs(values[j]-1)for j in range(n)if j!=plus));minus_gap=float(min(abs(values[j]+1)for j in range(n)if j!=minus))
    gates=dict(residual=bool(max(abs(value))<1e-9),positive_rates=bool(y[n]>0 and y[n+1]>0),primitive=all(v>1e-5 for v in divisors.values()),plus_one=bool(abs(values[plus]-1)<1e-7),minus_one=bool(abs(values[minus]+1)<1e-7),simple=bool(min(plus_gap,minus_gap)>1e-5),eigensystem=bool(max(residuals)<1e-9),complement=bool(max((abs(values[j])for j in others),default=0)<1-1e-5))
    return dict(passed=all(gates.values()),gates=gates,divisors=divisors,eigenvalues=[[float(v.real),float(v.imag)]for v in values],eigen_residuals=residuals,plus_index=plus,minus_index=minus,plus_gap=plus_gap,minus_gap=minus_gap,complement_max_modulus=float(max((abs(values[j])for j in others),default=0)),scope='Floating simultaneous native plus/minus-one proposal only; no rigorous multiplicity, generic fold/flip, unfolding, or branch connection.')
