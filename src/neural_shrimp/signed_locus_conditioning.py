"""Saved floating error budgets for a signed eigenpair tube; proposals only."""
import numpy as np
from .unit_pair_conditioning import middle_radius,positive_scaling

def parse(rows):
 n=len(rows);m=len(rows[0]);a=np.empty((n,m));r=np.empty_like(a)
 for i,row in enumerate(rows):
  if len(row)!=m:raise ValueError('rectangular')
  for j,v in enumerate(row):a[i,j],r[i,j]=middle_radius(v)
 return a,r

def budget(chart):
 n=chart['shooting_dimension'];C,Cr=parse(chart['preconditioner']);D,Dr=parse(chart['defect']);cc,cr=parse(chart['quadratic_center_coefficients']);wc,wr=parse(chart['quadratic_whole_coefficients']);J,Jr=parse(chart['center_jacobian']);S,Sr=parse(chart['condensed_border'])
 if C.shape!=(n,n)or D.shape!=(n,n)or cc.shape!=(n,4)or wc.shape!=(n,4):raise ValueError('complete dimensions')
 radius=np.array([middle_radius(v)[0]for v in chart['coordinate_radii']]);h=2.**chart['settings']['halfwidth_exponent']
 if np.any(radius<=0):raise ValueError('positive radii')
 force=[]
 for k in range(4):
  m,r=(wc[:,k],wr[:,k])if k==3 else(cc[:,k],cr[:,k]);v=(np.abs(C@m)+np.abs(C)@r+Cr@(np.abs(m)+r))*h**k;force.append(v)
 B=np.abs(D)+Dr;scaling=positive_scaling(B);weights=np.array(scaling['weights']);norms=np.linalg.norm(J,axis=1);singular=np.linalg.svd(S,compute_uv=False)
 ans=dict(scope='Binary64 saved-domain diagnostics/proposals only; no verified norm, changed-domain proof or stability conclusion.',dimension=n,halfwidth=h,radii=radius.tolist(),forcing_order_max_ratio=[float(np.max(v/radius))for v in force],forcing_orders=[v.tolist()for v in force],defect_mid_max_row=float(np.max(np.abs(D).sum(axis=1))),defect_rad_max_row=float(np.max(Dr.sum(axis=1))),absolute_row_max=float(np.max(B.sum(axis=1))),scaling=scaling,center_J_row_norm_range=[float(norms.min()),float(norms.max())],condensed_singular_values=singular.tolist(),image_max_ratio=float(np.max((sum(force)+B@radius)/radius)),scaled_image_max_ratio=float(np.max((sum(force)+B@(radius.max()*weights))/(radius.max()*weights))))
 if scaling['collatz_max']<.95:
  r=np.linalg.solve(np.eye(n)-B,sum(force));ans['resolvent']=dict(nonnegative=bool(np.isfinite(r).all()and np.all(r>=0)),radii=r.tolist(),residual=float(np.max(np.abs((np.eye(n)-B)@r-sum(force)))))
 return ans
