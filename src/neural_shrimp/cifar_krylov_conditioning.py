"""Saved numerical action geometry; no Hessian calls or full-spectrum claim."""
import numpy as np
SETTINGS=dict(max_columns=256,orthogonality_limit=1e-6,recurrence_limit=1e-8,least_squares_rcond=1e-12,span_representation_limit=1e-6,fixed_prefixes=[16,32,64,128,256])

def norm(v):return float(np.linalg.norm(v))
def diagnose(V,W,g,p,hp,tick=lambda:None):
    V,W,g,p,hp=[np.asarray(x,dtype=np.float64)for x in (V,W,g,p,hp)]
    if V.ndim!=2 or W.shape!=V.shape or not 1<=V.shape[1]<=256 or V.shape[0]<V.shape[1]or any(x.shape!=(V.shape[0],)for x in (g,p,hp))or not all(np.all(np.isfinite(x))for x in (V,W,g,p,hp)):raise ValueError('finite aligned saved actions')
    ng,np_,nh=norm(g),norm(p),norm(hp)
    if min(ng,np_,nh)<=0:raise ValueError('nonzero actual gradient and correction/action')
    m=V.shape[1];tick();gram=V.T@V;projected=V.T@W;tick()
    r1=-g.copy();r2=r1.copy();beta=ng;oldb=0.;alphas=[];betas=[];errors=[];complete=True
    for i in range(m):
        tick()
        if not beta>0 or not np.isfinite(beta):complete=False;break
        errors.append(norm(V[:,i]-r2/beta))
        y=W[:,i].copy()
        if i>=1:y-=(beta/oldb)*r1
        alpha=float(np.dot(V[:,i],y));y-=(alpha/beta)*r2
        r1=r2;r2=y;oldb=beta;beta=norm(y);alphas.append(alpha);betas.append(beta)
    prefixes=[]
    for k in sorted(set([v for v in SETTINGS['fixed_prefixes']if v<=m]+[m])):
        orth=float(np.linalg.norm(gram[:k,:k]-np.eye(k),ord=np.inf));error=max(errors[:k],default=float('inf'));qualified=len(errors)>=k and error<=SETTINGS['recurrence_limit']and orth<=SETTINGS['orthogonality_limit']
        row=dict(columns=k,gram_inf_error=orth,maximum_recurrence_vector_error=None if not np.isfinite(error)else error,ritz_qualified=bool(qualified))
        if qualified:
            T=np.diag(alphas[:k])+np.diag(betas[:k-1],1)+np.diag(betas[:k-1],-1);row['ritz_values']=np.linalg.eigvalsh(T).tolist()
        prefixes.append(row)
    tick();coef,_,rankW,sw=np.linalg.lstsq(W,-g,rcond=SETTINGS['least_squares_rcond']);tick();span=V@coef;pred=g+W@coef
    fitted,_,rankV,sv=np.linalg.lstsq(V,p,rcond=SETTINGS['least_squares_rcond']);tick()
    rep=norm(V@fitted-p)/np_;action=norm(W@fitted-hp)/nh;actual=norm(g+hp)/ng;prediction=norm(pred)/ng
    # Actual first Lanczos vector is authenticated before this identity is used.
    first=errors and errors[0]<=SETTINGS['recurrence_limit'];hg=-ng*W[:,0] if first else None
    result=dict(settings=SETTINGS,rows=V.shape[0],columns=m,recurrence_complete=complete,alphas=alphas,next_betas=betas,recurrence_vector_errors=errors,prefixes=prefixes,actual_forcing=actual,span_prediction=dict(forcing=prediction,relative_to_actual=None if actual==0 else prediction/actual,rank=int(rankW),singular_values=sw.tolist(),coefficients=coef.tolist(),coefficient_l1=float(np.linalg.norm(coef,ord=1)),direction_norm=norm(span),requires_fresh_physical_action=True),actual_direction_representation=dict(relative_point_error=rep,relative_action_error=action,qualified=bool(max(rep,action)<=SETTINGS['span_representation_limit']),rank=int(rankV),singular_values=sv.tolist(),coefficients=fitted.tolist()),projected_symmetry_relative=norm(projected-projected.T)/max(norm(projected),np.finfo(float).tiny),merit_gradient_first_action=None if hg is None else dict(norm=norm(hg),direction_slope=float(np.dot(hg,p)),fresh_action_slope=float(np.dot(g,hp))),scope='Four historical exact states only. Saved action-span prediction is numerical and requires a fresh physical action before any correction. Qualified Ritz values describe only finite projected action, not a certified full spectrum, gap or current final state.')
    return result,dict(gram=gram,projected_operator=projected),span
