"""Fixed saved-state Krylov comparison with two-pass full reorthogonalization."""
import numpy as np
SETTINGS=dict(max_actions=256,svd_rcond=1e-12,initial_alignment=1e-12,orthogonality_limit=1e-8,new_vector_alignment=1e-10,breakdown_relative=32*np.finfo(float).eps,linear_action_relative=1e-8,material_forcing_ratio=.8,strict_forcing=.5)

def vector(a,n=None):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=1 or (n is not None and a.shape!=(n,))or not np.isfinite(a).all():raise ValueError('full finite vector')
    return a

def orthogonalize(V,w):
    """Two full classical Gram-Schmidt passes; coefficients are retained."""
    q=vector(w,V.shape[0]).copy();coefficients=[]
    for _ in range(2):
        c=V.T@q;q-=V@c;coefficients.append(c.tolist())
    return q,coefficients

def solve(g,first,action,*,event=lambda k,v:None,tick=lambda:None):
    g=vector(g);n=len(g);first=vector(first,n).copy();ng=float(np.linalg.norm(g));m=min(n,SETTINGS['max_actions'])
    if not np.isfinite(ng)or ng<=0 or abs(float(np.linalg.norm(first))-1)>SETTINGS['initial_alignment']or float(np.linalg.norm(first+g/ng))>SETTINGS['initial_alignment']:raise ValueError('same literal original first direction aligned with negative gradient')
    V=np.empty((n,m),order='F');W=np.empty_like(V);V[:,0]=first;count=0;status='maximum_actions'
    for k in range(m):
        tick();w=vector(action(V[:,k].copy()),n);W[:,k]=w;count=k+1
        q,cs=orthogonalize(V[:,:count],w);nq=float(np.linalg.norm(q));nw=float(np.linalg.norm(w));ratio=nq/max(1.,nw)
        row=dict(index=k,coefficients=cs,raw_action_norm=nw,residual_norm=nq,relative_residual=ratio)
        if k+1==m:row['status']='fixed_action_cap';event('reorthogonalization',row);break
        if nq<=SETTINGS['breakdown_relative']*max(1.,nw):
            status='numerical_breakdown';row['status']=status;event('reorthogonalization',row);break
        candidate=q/nq;error=float(np.max(np.abs(V[:,:count].T@candidate)));row['new_vector_alignment']=error
        if error>SETTINGS['new_vector_alignment']:
            status='orthogonality_failure';row['status']=status;event('reorthogonalization',row);break
        V[:,count]=candidate;row['status']='next_vector_qualified';event('reorthogonalization',row)
    V=V[:,:count];W=W[:,:count];tick();gram=V.T@V;orth=float(np.linalg.norm(gram-np.eye(count),ord=np.inf))
    coefficients,_,rank,singular=np.linalg.lstsq(W,-g,rcond=SETTINGS['svd_rcond']);tick();p=vector(V@coefficients,n);hp=vector(W@coefficients,n)
    report=dict(status=status,actions=count,equal256_action_count=count==256,gram_inf_error=orth,orthogonality_qualified=orth<=SETTINGS['orthogonality_limit'],rank=int(rank),singular_values=singular.tolist(),coefficients=coefficients.tolist(),coefficient_l1=float(np.linalg.norm(coefficients,ord=1)),predicted_forcing=float(np.linalg.norm(g+hp))/ng,requires_fresh_physical_action=True)
    return p,hp,gram,report


def qualify(g,predicted,hp,report,baseline_forcing):
    g=vector(g);predicted=vector(predicted,len(g));hp=vector(hp,len(g));ng=float(np.linalg.norm(g))
    if not np.isfinite(ng)or ng<=0 or not np.isfinite(baseline_forcing)or baseline_forcing<=0:raise ValueError('positive physical comparison scales')
    forcing=float(np.linalg.norm(g+hp))/ng;error=float(np.linalg.norm(predicted-hp))/max(ng,float(np.linalg.norm(hp)),1e-12)
    qualified=bool(report['equal256_action_count']and report['orthogonality_qualified']and report['status']=='maximum_actions'and error<=SETTINGS['linear_action_relative'])
    return dict(qualified=qualified,true_forcing=forcing,baseline_forcing=baseline_forcing,forcing_ratio=forcing/baseline_forcing,physical_action_relative_error=error,material_improvement=qualified and forcing<=SETTINGS['material_forcing_ratio']*baseline_forcing,strict_forcing_pass=qualified and forcing<=SETTINGS['strict_forcing'],scope='Numerical equal-action comparison at one fixed historical point only. Direction never applied; no current final state, stationary point, mode or orbit claim.')
