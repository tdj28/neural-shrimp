"""Input-correlated positive right surrogate; not the exact two-phase Jacobian."""
from fractions import Fraction
import numpy as np

def require(ok,message):
 if not ok:raise ValueError(message)

def input_basis(x,emit=lambda k,v:None,call=lambda k,f,*a,**kw:f(*a,**kw)):
 x=np.asarray(x,dtype=np.float64);require(x.ndim==2 and 0<len(x)<=x.shape[1]+1 and np.isfinite(x).all(),'finite wide input')
 z=np.column_stack((x,np.ones(len(x))));q,r=call('qr',np.linalg.qr,z.T,mode='reduced');emit('qr',dict(q=q,r=r))
 require(np.isfinite(q).all()and np.isfinite(r).all(),'finite QR');qr_res=np.linalg.norm(q@r-z.T)/max(np.linalg.norm(z),1e-300);orth=np.linalg.norm(q.T@q-np.eye(len(x)));emit('qr_checks',dict(relative_residual=float(qr_res),orthogonality=float(orth)));require(max(qr_res,orth)<=1e-10,'QR checks')
 small=r@r.T/len(x);values,vectors=call('eigh',np.linalg.eigh,small);emit('eigh',dict(matrix=small,values=values,vectors=vectors))
 require(np.isfinite(values).all()and np.isfinite(vectors).all()and np.all(np.diff(values)>=0),'finite ordered eigenpairs');scale=max(1.,float(np.max(np.abs(values))));res=np.linalg.norm(small@vectors-vectors*values)/max(np.linalg.norm(small),1e-300);orth=np.linalg.norm(vectors.T@vectors-np.eye(len(x)));checks=dict(relative_residual=float(res),orthogonality=float(orth),minimum_eigenvalue=float(values[0]),negative_tolerance=1e-12*scale);emit('eigh_checks',checks);require(max(res,orth)<=1e-10 and values[0]>=-checks['negative_tolerance'],'Gram eigenpair checks')
 basis=q@vectors;clipped=np.maximum(values,0.);emit('basis',dict(vectors=basis,eigenvalues=clipped));require(np.isfinite(basis).all(),'finite basis')
 return basis,clipped

class RightScaling:
 def __init__(self,basis,values,c0,c1,d0,d1,eta,lam,d,width,k,emit=lambda k,v:None):
  self.d,self.width,self.k=d,width,k;self.u=np.asarray(basis);values=np.asarray(values);c0,c1=np.asarray(c0),np.asarray(c1);d0,d1=np.asarray(d0),np.asarray(d1)
  require(self.u.shape==(d+1,len(values))and c0.shape==c1.shape==(width,)and d0.shape==d1.shape==(d*width+width+width*k+k,),'packing')
  require(all(np.isfinite(v).all()for v in(self.u,values,c0,c1,d0,d1))and np.all(values>=0)and np.isfinite(eta)and eta>0 and np.isfinite(lam)and lam>0,'finite scaling inputs')
  q=1-Fraction.from_float(float(eta))*Fraction.from_float(float(lam));self.floor=float(abs(q*q-1));require(self.floor>0 and np.isfinite(self.floor),'ridge floor');self.complement=1/self.floor
  h0=lam+values[:,None]*c0;h1=lam+values[:,None]*c1;proxy=(1-eta*h1)*(1-eta*h0)-1;self.modes=1/np.maximum(np.abs(proxy),self.floor)
  cut=(d+1)*width;op=(1-eta*d1[cut:])*(1-eta*d0[cut:])-1;self.output=1/np.maximum(np.abs(op),self.floor);require(np.isfinite(self.modes).all()and np.isfinite(self.output).all(),'finite inverse scales')
  emit('scaling',dict(hidden_mode_proxy=proxy,hidden_mode_inverse=self.modes,output_proxy=op,output_inverse=self.output,phase0_mean_curvature=c0,phase1_mean_curvature=c1))
  self.summary=dict(ridge_floor=self.floor,ridge_floor_exact=str(abs(q*q-1)),complement_inverse=self.complement,mode_count=len(values),hidden_min_inverse=float(self.modes.min()),hidden_max_inverse=float(self.modes.max()),hidden_floor_modes=int(np.sum(np.abs(proxy)<=self.floor)),scope='shared augmented-input Gram basis; replaces each sample-dependent hidden curvature by its mean, omits cross-hidden/output blocks; positive right scaling only')
 def apply(self,v):
  v=np.asarray(v,dtype=np.float64);cut=(self.d+1)*self.width;require(v.shape==(cut+self.width*self.k+self.k,)and np.isfinite(v).all(),'finite physical packing');hidden=v[:cut].reshape(self.d+1,self.width);coords=self.u.T@hidden;out=self.complement*hidden+self.u@((self.modes-self.complement)*coords);result=np.r_[out.ravel(),self.output*v[cut:]];require(np.isfinite(result).all(),'finite right return');return result
