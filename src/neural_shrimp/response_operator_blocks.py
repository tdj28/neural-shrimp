"""Flexible GCR with fixed eight-step operator-only temporal block CG.

Saved small Jacobians act as callbacks. Every JVP/VJP, including inner actions,
is charged; no dense temporal factorization or nonlinear-model call occurs.
"""
import time
import numpy as np

class ActionBudget(RuntimeError):pass

def require(ok,message):
    if not ok:raise ValueError(message)

class Operator:
    def __init__(self,maps,steps,ceiling,emit=lambda k,v:None,tick=lambda:None,shift=-.05,eta=.01):
        maps=np.asarray(maps,dtype=np.float64)
        require(maps.ndim==3 and maps.shape[1]==maps.shape[2] and np.isfinite(maps).all(),'finite square periodic maps')
        self.maps=maps;self.p,self.n,_=maps.shape;self.scale=float(np.exp(-shift));self.steps=steps;self.eta=eta
        require(type(steps)is int and steps>0 and steps%self.p==0 and eta>0,'whole period positive regularizer')
        require(type(ceiling)is int and ceiling>=3*steps,'finite phase-action budget')
        self.ceiling=ceiling;self.counts={'JVP':0,'VJP':0};self.emit=emit;self.tick=tick;self.reserve=2*steps;self.events=0
    def admission(self,cost,terminal=False):
        self.tick()
        if sum(self.counts.values())+cost+(0 if terminal else self.reserve)>self.ceiling:raise ActionBudget('fixed phase-action ceiling with terminal reserve')
    def rhs(self):
        self.admission(self.steps);self.emit('rhs_started',dict(phase_actions_upper=self.steps,counts_before=dict(self.counts)))
        v=np.zeros((self.steps+1,self.n));impulse=np.zeros((self.steps,self.n));impulse[8*self.p,0]=1
        actions=[]
        for t in range(self.steps):
            self.counts['VJP']+=1;z=self.maps[t%self.p].T@impulse[t];actions.append(z.copy());v[t]-=self.scale*z;v[t+1]+=impulse[t]
        self.emit('rhs',dict(impulse=impulse,rhs=v.copy(),vjp_returns=np.asarray(actions),counts=dict(self.counts)))
        return v.ravel(),impulse
    def apply(self,x,start=0,end=None,terminal=False,impulse=None,tag='normal'):
        end=self.steps+1 if end is None else end
        require(0<=start<end<=self.steps+1,'principal block indices')
        x=np.asarray(x,dtype=np.float64).reshape(end-start,self.n);require(np.isfinite(x).all(),'finite operator input')
        active=max(0,min(end,self.steps)-start);self.admission(2*active,terminal)
        self.events+=1;self.emit('operator_started',dict(event=self.events,tag=tag,start=start,end=end,input=x.copy(),phase_actions_upper=2*active,counts_before=dict(self.counts),terminal=terminal))
        out=self.eta**2*x.copy();first=max(0,start-1);last=min(end,self.steps);edges=[];jvs=[];vjps=[]
        for t in range(first,last):
            y=x[t+1-start].copy() if t+1<end else np.zeros(self.n)
            if t>=start:
                self.counts['JVP']+=1;jv=self.maps[t%self.p]@x[t-start];jvs.append(jv.copy());y-=self.scale*jv
            if impulse is not None:
                require(start==0 and end==self.steps+1,'data residual only for whole operator');y-=impulse[t]
            edges.append(y.copy())
            if t>=start:
                self.counts['VJP']+=1;z=self.maps[t%self.p].T@y;vjps.append(z.copy());out[t-start]-=self.scale*z
            if t+1<end:out[t+1-start]+=y
        self.emit('operator_return',dict(event=self.events,tag=tag,start=start,end=end,input=x.copy(),output=out.copy(),edge_residuals=np.asarray(edges),jvp_returns=np.asarray(jvs).reshape(-1,self.n),vjp_returns=np.asarray(vjps).reshape(-1,self.n),counts=dict(self.counts),terminal=terminal))
        require(np.isfinite(out).all(),'finite operator output');return out.ravel()

def inner_cg(op,rhs,start,end,tag):
    """Zero start, <=8 iterations, fixed1e-2 relative stopping."""
    x=np.zeros_like(rhs);r=rhs.copy();p=r.copy();rr=float(r@r);initial=rr
    require(np.isfinite(rr) and rr>=0,'finite inner rhs')
    if rr==0:return x,0
    for iteration in range(1,9):
        hp=op.apply(p,start,end,tag=tag+f'-inner{iteration}');curvature=float(p@hp)
        require(np.isfinite(curvature) and curvature>0,'positive inner curvature')
        alpha=rr/curvature;x+=alpha*p;r-=alpha*hp;new=float(r@r)
        op.emit('inner_iterate',dict(tag=tag,start=start,end=end,iteration=iteration,state=x.copy(),residual=r.copy(),direction=p.copy(),curvature=curvature,alpha=alpha,residual_squared=new))
        require(np.isfinite(x).all() and np.isfinite(new) and new>=0,'finite inner return')
        if new<=1e-4*initial:return x,iteration
        p=r+(new/rr)*p;rr=new
    return x,8

def precondition(op,rhs,iteration):
    out=np.zeros_like(rhs);reports=[];chunk=4*op.p
    for start in range(0,op.steps+1,chunk):
        end=min(start+chunk,op.steps+1);part,steps=inner_cg(op,rhs[start*op.n:end*op.n],start,end,f'outer{iteration}-block{start}')
        out[start*op.n:end*op.n]=part;reports.append({'start':start,'end':end,'iterations':steps})
    op.emit('preconditioner_return',dict(iteration=iteration,input=rhs.copy(),output=out.copy(),blocks=reports,counts=dict(op.counts)))
    return out,reports

def probe(maps,ceiling,emit=lambda k,v:None,tick=lambda:None):
    start=time.monotonic();p,n,_=np.asarray(maps).shape;op=Operator(maps,16*p,ceiling,emit,tick);rhs,impulse=op.rhs();bnorm=float(np.linalg.norm(rhs));require(np.isfinite(bnorm) and bnorm>0,'nonzero finite rhs')
    x=np.zeros_like(rhs);r=rhs.copy();basis=[];status='iteration_limit';iterations=0;inner_iterations=0
    try:
        for k in range(1,301):
            tick();z,blocks=precondition(op,r,k);inner_iterations+=sum(a['iterations']for a in blocks)
            w=op.apply(z,tag=f'outer{k}-fresh_direction');initial=float(w@w)
            require(np.isfinite(initial)and initial>0,'finite nonzero outer action')
            # Flexible GCR stores each actual (z,H z), not a presumed fixed
            # linear preconditioner. Two MGS passes reduce accumulated loss.
            for _ in range(2):
                for oldz,oldw in basis:
                    beta=float(oldw@w);z-=beta*oldz;w-=beta*oldw
            norm=float(np.linalg.norm(w));require(np.isfinite(norm)and norm>1e-14*np.sqrt(initial),'outer action rank loss')
            z/=norm;w/=norm;alpha=float(w@r);candidate=x+alpha*z
            emit('outer_candidate',dict(iteration=k,old_state=x.copy(),state=candidate.copy(),direction=z.copy(),orthogonal_action=w.copy(),alpha=alpha,basis_length=len(basis)))
            require(np.isfinite(candidate).all(),'finite outer candidate')
            # Before accepting, use a fresh original unscaled normal residual.
            fresh=op.apply(candidate,impulse=impulse,tag=f'outer{k}-true_residual');x=candidate;r=-fresh;iterations=k
            rel=float(np.linalg.norm(r)/bnorm);emit('outer_iterate',dict(iteration=k,state=x.copy(),true_residual=r.copy(),relative_residual=rel,counts=dict(op.counts)))
            require(np.isfinite(rel),'finite outer residual')
            if rel<=1e-10:status='converged';break
            basis.append((z.copy(),w.copy()))
            if len(basis)==20:basis=[]
    except ActionBudget:status='action_budget_exhausted'
    terminal=op.apply(x,terminal=True,impulse=impulse,tag='terminal_true_residual');relative=float(np.linalg.norm(terminal)/bnorm)
    emit('terminal',dict(state=x.copy(),normal_residual=terminal.copy(),relative_residual=relative,counts=dict(op.counts),status=status))
    t0=8*p;readout=x[(t0+1)*n:(t0+2)*n]
    return {'method':'operator_four_period_flexible_gcr','status':status,'iterations':iterations,'completed_inner_iterations':inner_iterations,'counts':dict(op.counts),'action_ceiling':ceiling,'true_normal_relative_residual':relative,'solve_qualified':bool(status=='converged'and np.isfinite(relative)and relative<=1e-8),'readout_column':readout.tolist(),'elapsed_seconds':time.monotonic()-start,'settings':{'periods':16,'shift':-.05,'eta':.01,'inner_cg_maxiter':8,'inner_rtol':.01,'outer_restart':20,'outer_maxiter':300,'outer_rtol':1e-10,'terminal_actions_reserved':2*16*p},'scope':'saved small Jacobian callbacks, all inner/outer phase actions charged; no dense temporal inverse, no measured large-model or LLM speedup'}
