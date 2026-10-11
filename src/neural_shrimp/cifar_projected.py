"""One predetermined same-state physical subspace; no iterative enlargement."""
import numpy as np


def finite(v):
    v=np.asarray(v,dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():
        raise ValueError('finite complete vector')
    return v


def basis(candidates, *, tolerance=1e-12):
    if len(candidates)!=4 or tolerance!=1e-12:
        raise ValueError('exact four fixed candidates and rank threshold')
    vectors=[finite(v) for v in candidates]
    if len({v.shape for v in vectors})!=1:
        raise ValueError('same state-space dimension')
    accepted=[]; records=[]
    for index,v in enumerate(vectors):
        original=float(np.linalg.norm(v)); w=v.copy(); projections=[]
        # Two complete MGS passes; rank decisions use geometry, never actions.
        for _ in range(2):
            row=[]
            for u in accepted:
                coefficient=float(np.dot(u,w)); row.append(coefficient)
                w-=coefficient*u
            projections.append(row)
        remainder=float(np.linalg.norm(w))
        keep=bool(original>0 and remainder>tolerance*original)
        records.append(dict(index=index,original_norm=original,
                            remainder_norm=remainder,accepted=keep,
                            projections=projections))
        if keep:accepted.append(w/remainder)
    if not accepted:raise ValueError('empty fixed physical span')
    U=np.column_stack(accepted)
    if np.linalg.norm(U.T@U-np.eye(U.shape[1]),ord=np.inf)>1e-10:
        raise ValueError('orthogonal fixed span')
    return U,records


def proposal(U, actions, residual):
    U=np.asarray(U,dtype=np.float64); actions=np.asarray(actions,dtype=np.float64)
    r=finite(residual)
    if (U.ndim!=2 or U.shape!=actions.shape or U.shape[0]!=len(r)
        or not 1<=U.shape[1]<=4 or not np.isfinite(U).all()
        or not np.isfinite(actions).all() or np.linalg.norm(r)==0):
        raise ValueError('complete fixed span and physical actions')
    coefficients,_,rank,singular=np.linalg.lstsq(actions,-r,rcond=1e-12)
    p=U@coefficients; ap=actions@coefficients
    return p,ap,dict(coefficients=coefficients.tolist(),rank=int(rank),
                    singular_values=singular.tolist(),rcond=1e-12,
                    dimension=U.shape[1],
                    predicted_forcing=float(np.linalg.norm(r+ap)/np.linalg.norm(r)))


def qualify(residual, prediction, actual):
    r,p,a=[finite(v) for v in (residual,prediction,actual)]
    if r.shape!=p.shape or r.shape!=a.shape or np.linalg.norm(r)==0:
        raise ValueError('same physical residual space')
    error=float(np.linalg.norm(a-p)/max(np.linalg.norm(a),np.linalg.norm(p),1e-300))
    forcing=float(np.linalg.norm(r+a)/np.linalg.norm(r)); slope=float(np.dot(r,a))
    return dict(action_linearity_error=error,action_linearity_pass=bool(error<=5e-8),
                true_forcing=forcing,true_slope=slope,
                forcing_pass=bool(forcing<=.5),usable_descent=bool(slope<0))
