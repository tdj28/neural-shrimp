"""Conditional exact subset Krawczyk inequalities; floating LP is only a proposal."""
from fractions import Fraction as F
import warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning

def need(v,m):
 if not v:raise ValueError(m)

def verify(A,beta,anchor,z):
 n=len(beta);need(1<=n<=51 and len(A)==len(anchor)==len(z)==n and all(len(row)==n for row in A),'complete bounded coefficient arrays')
 need(all(type(v)is F and v>=0 for row in A for v in row)and all(type(v)is F and v>=0 for v in beta+anchor)and all(type(v)is F for v in z),'exact nonnegative rational envelopes')
 geometry=all(0<v<=1 for v in z)
 if not geometry:return dict(passed=False,subset_positive=False,proposal=z,reason='radius proposal leaves actual fullE or has nonpositive coordinate')
 dz=[sum((A[i][j]*z[j]for j in range(n)),F(0))for i in range(n)];image=[beta[i]+dz[i]for i in range(n)];root=[image[i]<z[i]for i in range(n)];join=[anchor[i]<z[i]for i in range(n)];q=max(dz[i]/z[i]for i in range(n));margins=[z[i]-image[i]for i in range(n)];join_margins=[z[i]-anchor[i]for i in range(n)]
 return dict(passed=bool(all(root)and all(join)and q<1),subset_positive=True,proposal=z,coordinate_inclusions=root,known_coordinate_inclusions=join,contraction=q,image=image,image_margins=margins,known_margins=join_margins,minimum_image_margin=min(margins),minimum_known_margin=min(join_margins),scope='Exact rational subset inequalities conditional on source-authenticated fullE derivative/forcing and known-root envelopes. No native complement or independent acquisition theorem.')

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v)for v in x]
 return x

def propose(A,beta,anchor,observe=lambda record:None):
 n=len(beta);a=np.array([[float(v)for v in row]for row in A]);b=np.array([float(v)for v in beta]);e=np.array([float(v)for v in anchor]);need(a.shape==(n,n)and e.shape==b.shape==(n,)and np.isfinite(a).all()and np.isfinite(b).all()and np.isfinite(e).all(),'finite full floating proposal inputs')
 # Maximize common slack against both root inclusion and known-root join;
 # 0<r_new/r_old<=1 prevents any new derivative domain assumption.
 lhs=np.vstack([np.c_[a-np.eye(n),np.ones(n)],np.c_[-np.eye(n),np.ones(n)]]);rhs=np.r_[-b,-e];objective=np.r_[np.zeros(n),-1.]
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',message='Unrecognized options detected.*threads.*',category=OptimizeWarning)
  raw=linprog(objective,A_ub=lhs,b_ub=rhs,bounds=[(0.,1.)]*n+[(0.,1.)],method='highs-ds',options=dict(threads=1,time_limit=1.))
 record=plain(dict(raw));observe(record)
 if not raw.success or raw.x is None or not np.isfinite(raw.x).all():return dict(proposal=record,verification=None,scope='Floating solver failure is not mathematical infeasibility.')
 z=[F.from_float(float(x))for x in raw.x[:n]];return dict(proposal=record,verification=verify(A,beta,anchor,z))
