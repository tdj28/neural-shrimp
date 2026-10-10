"""Fixed three-column residual minimization; no model or Krylov solver here."""
import numpy as np

def finite(a):
 a=np.asarray(a,dtype=np.float64)
 if not np.isfinite(a).all():raise ValueError('finite input')
 return a

def fit(directions,actions,residual,emit=lambda k,v:None):
 p,a,r=map(finite,(directions,actions,residual))
 if p.shape!=a.shape or p.ndim!=2 or p.shape[1]!=3 or r.shape!=(p.shape[0],):raise ValueError('exact three-column shape')
 scales=np.sqrt(np.sum(p*p,axis=0));emit('scales',dict(column_norms=scales))
 if not np.isfinite(scales).all()or np.any(scales<=0):raise ValueError('nonzero finite directions')
 pn=p/scales;an=a/scales;emit('normalized',dict(directions=pn,actions=an,residual=r))
 u,s,vh=np.linalg.svd(an,full_matrices=False);emit('svd',dict(u=u,s=s,vh=vh))
 if not all(np.isfinite(x).all()for x in(u,s,vh))or s.shape!=(3,)or np.any(s<0)or np.any(np.diff(s)>0)or s[0]<=0:raise ValueError('finite ordered SVD')
 reconstruction=float(np.linalg.norm(an-(u*s)@vh))/max(float(np.linalg.norm(an)),np.finfo(float).tiny);left_orth=float(np.linalg.norm(u.T@u-np.eye(3)));right_orth=float(np.linalg.norm(vh@vh.T-np.eye(3)));checks=dict(relative_reconstruction=reconstruction,left_orthogonality=left_orth,right_orthogonality=right_orth);emit('svd_checks',checks)
 if not all(np.isfinite(x)and x<=1e-10 for x in checks.values()):raise ValueError('SVD residual gates')
 rank=int(np.sum(s>1e-10*s[0]));condition=float(s[0]/s[-1])if s[-1]>0 else None
 if condition is not None and not np.isfinite(condition):condition=None
 decision=dict(rank=rank,condition=condition,rank_relative_threshold=1e-10,condition_limit=1e8,qualified=bool(rank==3 and condition is not None and condition<=1e8));emit('rank',decision)
 if not decision['qualified']:return None,decision
 coeff=vh.T@((u.T@(-r))/s);direction=pn@coeff;predicted=an@coeff;original=coeff/scales
 emit('fit',dict(normalized_coefficients=coeff,original_coefficients=original,direction=direction,predicted_action=predicted,predicted_error=predicted+r))
 if not all(np.isfinite(x).all()for x in(coeff,direction,predicted,original)):raise ValueError('finite fit')
 return dict(direction=direction,predicted_action=predicted,original_coefficients=original,normalized_coefficients=coeff),decision

def actual_gate(actual,predicted,residual):
 actual,predicted,r=map(finite,(actual,predicted,residual));den=float(np.linalg.norm(r))
 if den<=0:raise ValueError('nonzero residual')
 mismatch=float(np.linalg.norm(actual-predicted))/max(den,float(np.linalg.norm(actual)),float(np.linalg.norm(predicted)));forcing=float(np.linalg.norm(actual+r))/den;slope=float(r@actual)
 if not all(np.isfinite(x)for x in(mismatch,forcing,slope)):raise ValueError('finite actual diagnostics')
 return dict(relative_action_mismatch=mismatch,action_mismatch_limit=1e-8,action_linearity_pass=mismatch<=1e-8,true_forcing_ratio=forcing,forcing_limit=.5,forcing_test_pass=forcing<=.5,merit_directional_derivative=slope,descent_direction=slope<0,qualified=bool(mismatch<=1e-8 and forcing<=.5 and slope<0))
