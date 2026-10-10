"""Full-time coordinate lines for the exact free-window normal operator.

This is a small-cocycle diagnostic. Exact column extraction costs n*p JVPs;
large-model use would require a separately assessed diagonal estimator.
"""
import time
import numpy as np
from .response_operator_blocks import Operator, ActionBudget, require


def factor_lines(diagonal, off):
    diagonal=np.asarray(diagonal,dtype=float);off=np.asarray(off,dtype=float)
    require(diagonal.ndim==2 and off.shape==(diagonal.shape[0]-1,diagonal.shape[1]),'line dimensions')
    require(np.isfinite(diagonal).all() and np.isfinite(off).all(),'finite lines')
    pivot=diagonal.copy();lower=np.zeros_like(off)
    for t in range(len(off)):
        require(np.all(pivot[t]>0),'positive line pivot')
        lower[t]=off[t]/pivot[t];pivot[t+1]-=lower[t]*off[t]
    require(np.isfinite(pivot).all() and np.all(pivot>0),'positive final line pivots')
    return lower,pivot


def solve_lines(lower,pivot,rhs):
    shape=np.asarray(pivot).shape;y=np.asarray(rhs,dtype=float).reshape(shape).copy()
    require(np.isfinite(y).all(),'finite line rhs')
    for t in range(1,len(y)):y[t]-=lower[t-1]*y[t-1]
    y/=pivot
    for t in range(len(y)-2,-1,-1):y[t]-=lower[t]*y[t+1]
    require(np.isfinite(y).all(),'finite line solution')
    return y.ravel()


def setup(op):
    op.admission(op.p*op.n)
    op.emit('line_setup_started',dict(phase_actions_upper=op.p*op.n,counts_before=dict(op.counts)))
    cols=np.empty((op.p,op.n,op.n));eye=np.eye(op.n)
    for phase in range(op.p):
        for j in range(op.n):
            op.tick();op.counts['JVP']+=1;cols[phase,j]=op.maps[phase]@eye[j]
            op.emit('line_setup_column',dict(phase=phase,coordinate=j,input=eye[j].copy(),output=cols[phase,j].copy(),counts=dict(op.counts)))
    # cols[phase,column,row]: retain exact column norm and diagonal only.
    cn=np.sum(cols*cols,axis=2);jd=np.diagonal(cols,axis1=1,axis2=2)
    diagonal=np.full((op.steps+1,op.n),op.eta**2);diagonal[1:]+=1
    diagonal[:-1]+=op.scale**2*cn[np.arange(op.steps)%op.p]
    off=-op.scale*jd[np.arange(op.steps)%op.p]
    lower,pivot=factor_lines(diagonal,off)
    op.emit('line_setup_return',dict(diagonal=diagonal,off=off,lower=lower,pivot=pivot,counts=dict(op.counts)))
    return lower,pivot


def probe(maps,ceiling,emit=lambda k,v:None,tick=lambda:None):
    started=time.monotonic();p,n,_=np.asarray(maps).shape;op=Operator(maps,16*p,ceiling,emit,tick)
    lower,pivot=setup(op);rhs,impulse=op.rhs();bnorm=float(np.linalg.norm(rhs))
    require(np.isfinite(bnorm) and bnorm>0,'nonzero finite rhs')
    x=np.zeros_like(rhs);r=rhs.copy();z=solve_lines(lower,pivot,r);direction=z.copy();rz=float(r@z)
    require(np.isfinite(rz) and rz>0,'positive initial preconditioned norm')
    emit('line_preconditioner',dict(iteration=0,input=r.copy(),output=z.copy()))
    status='iteration_limit';iterations=0;applications=1
    try:
        for k in range(1,301):
            tick();hp=op.apply(direction,tag=f'pcg{k}-direction');curvature=float(direction@hp)
            require(np.isfinite(curvature) and curvature>0,'positive true normal curvature')
            alpha=rz/curvature;candidate=x+alpha*direction;next_r=r-alpha*hp
            rel=float(np.linalg.norm(next_r)/bnorm)
            require(np.isfinite(candidate).all() and np.isfinite(next_r).all() and np.isfinite(rel),'finite PCG iterate')
            emit('line_pcg_iterate',dict(iteration=k,old_state=x.copy(),state=candidate.copy(),direction=direction.copy(),action=hp.copy(),residual=next_r.copy(),alpha=alpha,curvature=curvature,relative_recursive_residual=rel,counts=dict(op.counts)))
            x=candidate;r=next_r;iterations=k
            if rel<=1e-10:status='recursive_convergence';break
            z=solve_lines(lower,pivot,r);applications+=1;new=float(r@z)
            require(np.isfinite(new) and new>0,'positive preconditioned norm')
            emit('line_preconditioner',dict(iteration=k,input=r.copy(),output=z.copy(),rz=new))
            direction=z+(new/rz)*direction;rz=new
    except ActionBudget:status='action_budget_exhausted'
    terminal=op.apply(x,terminal=True,impulse=impulse,tag='terminal_true_residual');relative=float(np.linalg.norm(terminal)/bnorm)
    emit('terminal',dict(state=x.copy(),normal_residual=terminal.copy(),relative_residual=relative,counts=dict(op.counts),status=status))
    qualified=bool(status=='recursive_convergence' and np.isfinite(relative) and relative<=1e-8)
    return dict(method='coordinate_full_time_line_pcg',status=status,iterations=iterations,counts=dict(op.counts),action_ceiling=ceiling,
      true_normal_relative_residual=relative,solve_qualified=qualified,readout_column=x[(8*p+1)*n:(8*p+2)*n].tolist(),
      elapsed_seconds=time.monotonic()-started,line_applications=applications,scalar_line_solves=applications*n,
      setup_jvps=p*n,settings=dict(periods=16,shift=-.05,eta=.01,maxiter=300,rtol=1e-10),
      scope='Full-time scalar coordinate-line diagnostic; exact n*p JVP setup charged; no dense time factors or inner Krylov; no scalable LLM diagonal estimator or geometry utility established')
