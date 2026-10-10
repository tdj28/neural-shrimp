"""Bounded floating native unit-pair continuation; every result is a proposal."""
from fractions import Fraction
import math
import numpy as np


def need(condition,message):
    if not condition:raise ValueError(message)


class Jet:
    """Binary64 value, base derivative, fixed direction and mixed derivative."""
    def __init__(self,f,g,d,m):self.f,self.g,self.d,self.m=float(f),np.asarray(g),float(d),np.asarray(m)
    @classmethod
    def variable(cls,f,n,i,d=0.):return cls(f,np.eye(n)[i],d,np.zeros(n))
    def __add__(self,b):
        if not isinstance(b,Jet):return Jet(self.f+b,self.g,self.d,self.m)
        return Jet(self.f+b.f,self.g+b.g,self.d+b.d,self.m+b.m)
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,b):return self+(-b)
    def __rsub__(self,b):return -self+b
    def __mul__(self,b):
        if not isinstance(b,Jet):return Jet(self.f*b,self.g*b,self.d*b,self.m*b)
        return Jet(self.f*b.f,self.g*b.f+self.f*b.g,self.d*b.f+self.f*b.d,
                   self.m*b.f+self.g*b.d+self.d*b.g+self.f*b.m)
    __rmul__=__mul__
    def tanh(self):
        t=np.tanh(self.f);s=1-t*t
        return Jet(t,s*self.g,s*self.d,s*self.m-2*t*s*self.g*self.d)


class TanhMap:
    """Same analytic exact-input formula, evaluated in binary64 for scouting."""
    def __init__(self,inp):
        need(inp['schema']=='neural-shrimp-c1-input-v1','map schema');net=inp['network']
        need(net['activation']=='tanh'and type(net['bias'])is bool,'tanh/bias')
        self.d,self.w,self.bias=net['d'],net['width'],net['bias'];self.a=self.d*self.w
        self.hidden=self.a+self.w*self.bias;self.n=self.hidden+self.w+self.bias
        f=lambda v:float(Fraction(v))
        self.ins,self.out,self.loss,self.ridge=[f(net[k])for k in ('in_scale','out_scale','loss_scale','ridge')]
        self.x=[[f(v)for v in row]for row in inp['x']];self.y=[f(v)for v in inp['y']];self.k=2*self.loss/len(self.x)
        need(all(math.isfinite(v)for row in self.x for v in row)and np.isfinite(self.y).all(),'finite exact data conversion')
    def cusp_step(self,theta,h,o):
        grad=[v*self.ridge for v in theta]
        for x,y in zip(self.x,self.y):
            acts=[];features=[]
            for unit in range(self.w):
                feature=[(d*self.w+unit,self.ins*x[d])for d in range(self.d)]
                if self.bias:feature.append((self.a+unit,1.))
                acts.append(sum((theta[i]*v for i,v in feature),0).tanh());features.append(feature)
            prediction=sum((theta[self.hidden+i]*acts[i]for i in range(self.w)),0)*self.out
            if self.bias:prediction+=theta[-1]
            residual=prediction-y
            for unit in range(self.w):
                oi=self.hidden+unit;grad[oi]+=residual*acts[unit]*(self.k*self.out)
                back=residual*theta[oi]*(1-acts[unit]*acts[unit])*(self.k*self.out)
                for i,v in features[unit]:grad[i]+=back*v
            if self.bias:grad[-1]+=residual*self.k
        return [v-grad[i]*(h if i<self.hidden else o)for i,v in enumerate(theta)]


def coefficients(t):
    need(math.isfinite(t)and 0<=t<=3,'bounded angle squared')
    # Entire small-t polynomial avoids subtractive loss in e\'.
    if t<.01:
        c=sum((-1.)**k*t**k/math.factorial(2*k)for k in range(9))
        e=sum((-1.)**k*t**k/math.factorial(2*k+1)for k in range(9))
        ep=sum((-1.)**k*k*t**(k-1)/math.factorial(2*k+1)for k in range(1,9))
    else:
        a=math.sqrt(t);c=math.cos(a);e=math.sin(a)/a;ep=(c-e)/(2*t)
    return c,e,t*e,-e/2,ep,e+t*ep


def evaluate(model,z,period,t,tick=lambda:None):
    """z=(x, actual rates, q, r); all native state and mode equations retained."""
    n=model.n;z=np.asarray(z,dtype=float);base=n+2;dim=3*n+2
    need(z.shape==(dim,)and np.isfinite(z).all()and 1<=period<=18,'full floating dimensions')
    x=z[:n];rates=z[n:n+2];q=z[base:base+n];r=z[base+n:];passes=[]
    for direction in (q,r):
        state=[Jet.variable(x[i],base,i,direction[i])for i in range(n)]
        h=Jet.variable(rates[0],base,n);o=Jet.variable(rates[1],base,n+1)
        orbit=[x.copy()];modes=[direction.copy()]
        for _ in range(period):
            tick();state=model.cusp_step(state,h,o)
            orbit.append(np.array([v.f for v in state]));modes.append(np.array([v.d for v in state]))
        passes.append((state,orbit,modes))
    first,orbit,qs=passes[0];second,_,rs=passes[1]
    M=np.array([v.g[:n]for v in first]);c,e,d,cp,ep,dp=coefficients(t)
    value=np.r_[[v.f for v in first]-x,[v.d for v in first]-c*q+d*r,
                [v.d for v in second]-e*q-c*r,(q@q-1)/2,q@r]
    jac=np.zeros((dim,dim));jac[:n,:base]=[v.g for v in first];jac[:n,:n]-=np.eye(n)
    jac[n:2*n,:base]=[v.m for v in first];jac[n:2*n,base:base+n]=M-c*np.eye(n);jac[n:2*n,base+n:]=d*np.eye(n)
    jac[2*n:3*n,:base]=[v.m for v in second];jac[2*n:3*n,base:base+n]=-e*np.eye(n);jac[2*n:3*n,base+n:]=M-c*np.eye(n)
    jac[-2,base:base+n]=q;jac[-1,base:base+n]=r;jac[-1,base+n:]=q
    ht=np.r_[np.zeros(n),-cp*q+dp*r,-ep*q-cp*r,0.,0.]
    need(np.isfinite(value).all()and np.isfinite(jac).all(),'finite floating return')
    return value,jac,ht,{'orbit':np.array(orbit),'q_phases':np.array(qs),'r_phases':np.array(rs),'monodromy':M}


def native_flat(z,data):
    n=data['orbit'].shape[1]
    return np.r_[np.concatenate([np.r_[x,q,r]for x,q,r in zip(data['orbit'][:-1],data['q_phases'][:-1],data['r_phases'][:-1])]),z[n:n+2]]


def qualify(z,t,value,data,previous=None):
    n=data['monodromy'].shape[0];p=len(data['orbit'])-1;q=z[n+2:2*n+2];r=z[2*n+2:]
    vals,vecs=np.linalg.eig(data['monodromy']);critical=np.exp(1j*math.sqrt(t));i=int(np.argmin(abs(vals-critical)))
    others=[j for j in range(n)if j!=i];j=min(others,key=lambda j:abs(vals[j]-critical.conjugate()))
    eigerr=[float(np.linalg.norm(data['monodromy']@vecs[:,k]-vals[k]*vecs[:,k])/max(1.,np.linalg.norm(data['monodromy'])*np.linalg.norm(vecs[:,k])))for k in range(n)]
    rest=[v for k,v in enumerate(vals)if k not in (i,j)]
    divisors={str(d):float(np.max(abs(data['orbit'][d]-data['orbit'][0])))for d in range(1,p)if p%d==0}
    step=0. if previous is None else float(np.max(abs(data['orbit']-previous['orbit'])/np.maximum(1.,abs(previous['orbit']))))
    pair_error=max(abs(vals[i]-critical),abs(vals[j]-critical.conjugate()))
    gates={'residual':bool(max(abs(value))<1e-9),'positive_rates':bool(np.all(z[n:n+2]>0)),
           'native_primitive':all(v>1e-5 for v in divisors.values()),'native_step':step<.1,
           'full_spectrum_residual':max(eigerr)<1e-9,'actual_pair':pair_error<1e-7,'realified_independent':float(np.linalg.norm(r))>1e-5}
    return {'passed':all(gates.values()),'gates':gates,'residual_inf':float(max(abs(value))),'native_step':step,
            'divisor_distances':divisors,'eigenvalues':[[float(v.real),float(v.imag)]for v in vals],
            'eigen_residuals':eigerr,'pair_indices':[i,j],'pair_error':float(pair_error),
            'complement_max_modulus':float(max(abs(v)for v in rest))if rest else 0.,
            'complement_unit_gap':float(min(abs(abs(v)-1)for v in rest))if rest else 1.,
            'angle':math.sqrt(t),'rates':z[n:n+2].tolist(),
            'scope':'Floating native full-state/mode continuation diagnostics only; not a proof or rigorous branch connection.'}


def continue_branch(model,z,t,*,max_points=64,max_evaluations=360,t_max=math.pi**2/4,tick=lambda:None,observe=lambda k,v:None):
    calls=0;points=[];last=None
    def call(z,t):
        nonlocal calls
        need(calls<max_evaluations,'full return evaluation cap');calls+=1
        return evaluate(model,z,18,t,tick)
    # Numerical seed correction remains separately checked against rigorous endpoint by caller.
    z=np.array(z,dtype=float);v,J,ht,data=call(z,t)
    for k in range(9):
        if max(abs(v))<1e-10:break
        need(k<8,'initial corrector iteration cap')
        delta=np.linalg.solve(J,v);need(max(abs(delta)/np.maximum(1.,abs(z)))<.1,'local Newton guard')
        z-=delta;v,J,ht,data=call(z,t)
    initial=qualify(z,t,v,data);need(initial['passed'],'initial numerical branch qualification')
    observe('seed',{'z':z.tolist(),'t':t,'diagnostics':initial,'native_flat':native_flat(z,data).tolist()})
    for index in range(max_points):
        info=qualify(z,t,v,data,last);need(info['passed'],'qualified numerical native point')
        row={'index':index,'z':z.tolist(),'t':t,'diagnostics':info,'native_flat':native_flat(z,data).tolist()};observe('point',row);points.append(row)
        if info['complement_unit_gap']<.03:return {'status':'competing_unit_mode_proposal','points':points,'evaluations':calls}
        if t>=t_max:return {'status':'angle_ceiling_reached','points':points,'evaluations':calls}
        tangent=np.linalg.solve(J,-ht);step=min(max(.5*t,1/32768),.125,t_max-t);accepted=False
        for retry in range(5):
            tt=t+step;zz=z+tangent*step;record={'index':index,'retry':retry,'from_t':t,'to_t':tt,'passed':False}
            try:
                for iteration in range(9):
                    vv,JJ,hh,dd=call(zz,tt)
                    if max(abs(vv))<1e-10:break
                    need(iteration<8,'corrector iteration cap')
                    delta=np.linalg.solve(JJ,vv);need(max(abs(delta)/np.maximum(1.,abs(zz)))<.1,'local Newton guard')
                    zz-=delta
                info=qualify(zz,tt,vv,dd,data);need(info['passed'],'numerical corrector qualification')
                need(max(abs(zz-z)/np.maximum(1.,abs(z)))<.1,'all-coordinate neighboring point guard')
                record.update(passed=True,iterations=iteration+1,diagnostics=info)
                last=data;z,t,v,J,ht,data=zz,tt,vv,JJ,hh,dd;accepted=True
            except (ValueError,np.linalg.LinAlgError,FloatingPointError)as exc:record['error']=type(exc).__name__+': '+str(exc)
            observe('attempt',record)
            if accepted:break
            step/=2
        if not accepted:return {'status':'corrector_limit','points':points,'evaluations':calls}
    return {'status':'point_cap','points':points,'evaluations':calls}
