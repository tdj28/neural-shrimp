"""Exact mathematical MSE Hessian diagonal; approximate two-step right scaling."""
from fractions import Fraction
import numpy as np

def loss_hessian_diagonal(theta,x,y,d,width,k,lam):
 theta=np.asarray(theta,dtype=np.float64);x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);sizes=(d*width,width,width*k,k)
 if theta.shape!=(sum(sizes),)or x.ndim!=2 or x.shape[1]!=d or y.shape!=(len(x),k)or len(x)==0 or not np.isfinite(lam)or lam<0:raise ValueError('model/input dimensions')
 if not all(np.isfinite(a).all()for a in(theta,x,y)):raise ValueError('finite model/data')
 i=np.cumsum(sizes)[:-1];w1,b1,w2,b2=np.split(theta,i);w1=w1.reshape(d,width);w2=w2.reshape(width,k);z=np.tanh(x@w1+b1);error=z@w2+b2-y;sech=1-z*z
 c=sech**2*np.sum(w2*w2,axis=1)[None,:]-2*z*sech*(error@w2.T)
 d1=(x*x).T@c/len(x)+lam;db1=c.mean(axis=0)+lam;d2=np.repeat((z*z).mean(axis=0)[:,None],k,axis=1)+lam;db2=np.full(k,1.+lam)
 diagonal=np.r_[d1.ravel(),db1,d2.ravel(),db2]
 return diagonal,dict(activation=z,error=error,hidden_curvature=c)

def scaling(d0,d1,eta,lam):
 d0,d1=np.asarray(d0),np.asarray(d1)
 if d0.shape!=d1.shape or not all(np.isfinite(x).all()for x in(d0,d1))or not(np.isfinite(eta)and eta>0 and np.isfinite(lam)and lam>0):raise ValueError('finite diagonal/rates')
 q=1-Fraction.from_float(float(eta))*Fraction.from_float(float(lam));ridge=abs(q*q-1);floor=float(ridge)
 if not np.isfinite(floor)or floor<=0:raise ValueError('positive ridge floor')
 proxy=(1-eta*d1)*(1-eta*d0)-1;inverse=1/np.maximum(np.abs(proxy),floor)
 if not np.isfinite(proxy).all()or not np.isfinite(inverse).all()or np.any(inverse<=0):raise ValueError('finite positive scaling')
 return inverse,proxy,dict(ridge_floor=floor,ridge_floor_exact=str(ridge),floor_coordinates=int(np.sum(np.abs(proxy)<=floor)),minimum_inverse=float(inverse.min()),maximum_inverse=float(inverse.max()),scope='positive right scaling from diagonal-phase product; omits off-diagonal Hessian-product contributions')
