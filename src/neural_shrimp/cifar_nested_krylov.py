"""Extend a source-qualified fixed-point basis without repeating its native actions."""
import numpy as np
from neural_shrimp.cifar_reorthogonalized import vector,orthogonalize,SETTINGS


def extend(g,prefix_v,prefix_w,action,*,max_columns=512,event=lambda k,v:None,tick=lambda:None):
    g=vector(g);n=len(g);ng=float(np.linalg.norm(g))
    if not np.isfinite(ng)or ng<=0:raise ValueError('positive finite gradient norm')
    V0=np.asarray(prefix_v);W0=np.asarray(prefix_w)
    if V0.ndim!=2 or V0.shape!=W0.shape or V0.shape[0]!=n or V0.dtype!=np.float64 or W0.dtype!=np.float64 or not np.isfinite(V0).all()or not np.isfinite(W0).all():raise ValueError('literal finite binary64 prefix matrices')
    m=V0.shape[1]
    if not isinstance(max_columns,int)or not 1<=m<max_columns<=min(n,512):raise ValueError('strict bounded nested extension')
    if float(np.linalg.norm(V0[:,0]+g/ng))>SETTINGS['initial_alignment']:raise ValueError('same negative-gradient first vector')
    before=float(np.linalg.norm(V0.T@V0-np.eye(m),ord=np.inf))
    if before>SETTINGS['orthogonality_limit']:raise ValueError('qualified full prefix orthogonality')
    V=np.empty((n,max_columns),order='F');W=np.empty_like(V);V[:,:m]=V0;W[:,:m]=W0
    count=m;status='maximum_columns';fresh=0
    while count<max_columns:
        tick();w=W[:,count-1];q,cs=orthogonalize(V[:,:count],w);nq=float(np.linalg.norm(q));nw=float(np.linalg.norm(w));row=dict(from_column=count-1,coefficients=cs,residual_norm=nq,raw_action_norm=nw)
        if nq<=SETTINGS['breakdown_relative']*max(1.,nw):
            status='numerical_breakdown';row['status']=status;event('nested_reorthogonalization',row);break
        candidate=q/nq;error=float(np.max(np.abs(V[:,:count].T@candidate)));row['new_vector_alignment']=error
        if error>SETTINGS['new_vector_alignment']:
            status='orthogonality_failure';row['status']=status;event('nested_reorthogonalization',row);break
        row['status']='next_vector_qualified';event('nested_reorthogonalization',row);V[:,count]=candidate;W[:,count]=vector(action(candidate.copy()),n);count+=1;fresh+=1
    tick();V=V[:,:count];W=W[:,:count];gram=V.T@V;orth=float(np.linalg.norm(gram-np.eye(count),ord=np.inf))
    c,_,rank,singular=np.linalg.lstsq(W,-g,rcond=SETTINGS['svd_rcond']);tick();p=vector(V@c,n);hp=vector(W@c,n)
    report=dict(status=status,prefix_columns=m,total_columns=count,new_basis_actions=fresh,maximum_columns=max_columns,completed_fixed_extension=count==max_columns,prefix_gram_inf_error=before,gram_inf_error=orth,orthogonality_qualified=orth<=SETTINGS['orthogonality_limit'],rank=int(rank),singular_values=singular.tolist(),coefficients=c.tolist(),coefficient_l1=float(np.linalg.norm(c,ord=1)),predicted_forcing=float(np.linalg.norm(g+hp))/ng,requires_fresh_physical_action=True)
    return p,hp,gram,report


def qualify(g,predicted,actual,report,baseline):
    g=vector(g);predicted=vector(predicted,len(g));actual=vector(actual,len(g));ng=float(np.linalg.norm(g))
    if not np.isfinite(ng)or ng<=0 or not np.isfinite(baseline)or baseline<=0:raise ValueError('positive finite comparison scales')
    forcing=float(np.linalg.norm(g+actual))/ng;error=float(np.linalg.norm(predicted-actual))/max(ng,float(np.linalg.norm(actual)),1e-12)
    passed=bool(report['completed_fixed_extension']and report['orthogonality_qualified']and report['status']=='maximum_columns'and error<=SETTINGS['linear_action_relative'])
    return dict(qualified=passed,true_forcing=forcing,baseline_forcing=baseline,forcing_ratio=forcing/baseline,physical_action_relative_error=error,material_improvement=passed and forcing<=.8*baseline,strict_forcing_pass=passed and forcing<=.5,scope='Numerical nested larger-budget solve at one unchanged consumed state. No equal-action comparison, applied direction, current-final-state claim, stationary root or orbit certificate.')
