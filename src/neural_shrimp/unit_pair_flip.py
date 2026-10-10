"""Numerical full-state unit-pair/flip junction; every result is a proposal."""
import numpy as np
from .unit_pair_scout import Jet,evaluate,qualify,native_flat,need
from .unit_pair_section import native_association


def native_direction(model,x,rates,w,period=18,tick=lambda:None):
    n=model.n;base=n+2
    state=[Jet.variable(x[i],base,i,w[i])for i in range(n)]
    h=Jet.variable(rates[0],base,n);o=Jet.variable(rates[1],base,n+1)
    orbit=[np.asarray(x).copy()];modes=[np.asarray(w).copy()]
    for _ in range(period):
        tick();state=model.cusp_step(state,h,o)
        orbit.append(np.array([v.f for v in state]));modes.append(np.array([v.d for v in state]))
    return np.array([v.f for v in state]),np.array([v.g for v in state]),np.array([v.d for v in state]),np.array([v.m for v in state]),{'orbit':np.array(orbit),'w_phases':np.array(modes)}


def augmented(model,y,period=18,tick=lambda:None):
    n=model.n;k=3*n+2;dim=4*n+3;y=np.asarray(y,dtype=float)
    need(y.shape==(dim,)and np.isfinite(y).all(),'full native augmented dimensions')
    z,t,w=y[:k],float(y[k]),y[k+1:]
    value,J,ht,data=evaluate(model,z,period,t,tick)
    f,g,mw,mixed,flip=native_direction(model,z[:n],z[n:n+2],w,period,tick)
    need(np.array_equal(f,data['orbit'][-1])and np.array_equal(g[:,:n],data['monodromy']),'same native map and monodromy in all three directions')
    out=np.r_[value,mw+w,(w@w-1)/2];jac=np.zeros((dim,dim))
    jac[:k,:k]=J;jac[:k,k]=ht;jac[k:k+n,:n+2]=mixed
    jac[k:k+n,k+1:]=data['monodromy']+np.eye(n);jac[-1,k+1:]=w
    data.update(flip)
    need(np.isfinite(out).all()and np.isfinite(jac).all(),'finite full native junction return')
    return out,jac,data


def spectrum_report(M,w,angle):
    values,vectors=np.linalg.eig(M);pair=np.exp(1j*angle)
    i=int(np.argmin(abs(values-pair)));js=[j for j in range(len(w))if j!=i];j=min(js,key=lambda j:abs(values[j]-pair.conjugate()))
    rest=[k for k in js if k!=j];f=min(rest,key=lambda k:abs(values[k]+1));other=[k for k in rest if k!=f]
    eigres=[float(np.linalg.norm(M@vectors[:,k]-values[k]*vectors[:,k])/max(1.,np.linalg.norm(M)*np.linalg.norm(vectors[:,k])))for k in range(len(w))]
    residual=float(np.linalg.norm(M@w+w)/max(1.,np.linalg.norm(M)*np.linalg.norm(w)))
    gates={'unit_pair':max(abs(values[i]-pair),abs(values[j]-pair.conjugate()))<1e-7,'simple_flip':abs(values[f]+1)<1e-7 and min(abs(values[k]+1)for k in range(len(w))if k!=f)>1e-4,'remaining_attracting':max(abs(values[k])for k in other)<1-1e-5,'eigensystem_residual':max(eigres)<1e-9,'native_flip_vector':residual<1e-9}
    return {'passed':all(gates.values()),'gates':{k:bool(v)for k,v in gates.items()},'pair_indices':[i,j],'flip_index':f,'eigenvalues':[[float(v.real),float(v.imag)]for v in values],'eigen_residuals':eigres,'flip_vector_residual':residual,'remaining_max_modulus':float(max(abs(values[k])for k in other))}


def crossing_derivative(J,w,M,n):
    k=3*n+2;section=J[:k,:k].copy();section[:,n]=J[:k,k]
    tangent=np.linalg.solve(section,-J[:k,n]);physical=tangent[:n+2].copy();physical[n]=1.
    vals,vec=np.linalg.eig(M.T);idx=int(np.argmin(abs(vals+1)));left=vec[:,idx]
    need(abs(vals[idx].imag)<1e-8 and max(abs(left.imag))<1e-8,'real critical left mode')
    left=left.real;need(abs(left@w)>1e-8,'nonzero critical dual pairing');left/=left@w
    need(np.linalg.norm(left@M+left)<1e-7*max(1.,np.linalg.norm(left)*np.linalg.norm(M)),'critical left residual')
    derivative=float(left@(J[k:k+n,:n+2]@physical))
    return {'passed':abs(derivative)>1e-5,'multiplier_hidden_rate_derivative':derivative,'native_section_tangent':tangent.tolist(),'physical_state_rate_tangent':physical.tolist(),'left':left.tolist(),'scope':'Floating transverse crossing of the flip mode along the native unit-pair section; no interval or normal-form theorem.'}
