"""Finite-N fixed random RNN map exploration; no shrimp/proof inference."""
from __future__ import annotations
import numpy as np


def inputs(n, seed):
    if n < 1 or 29000 <= seed <= 29999:
        raise ValueError("dimension or protected seed")
    j = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,n,0]))).normal(size=(n,n))/np.sqrt(n)
    x = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,n,1]))).normal(size=(2,n))
    v = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,n,2]))).normal(size=(2,n))
    return j, x, v


def step(x, j, g, delta):
    return (1-delta)*x + delta*g*(np.tanh(x) @ j.T)


def jacobian(x, j, g, delta):
    return (1-delta)*np.eye(len(x)) + delta*g*j*(1-np.tanh(x)**2)[None,:]


def iterate(x, j, g, delta, period):
    states=[]; matrices=[]; m=np.eye(len(x)); z=np.array(x,copy=True)
    for _ in range(period):
        states.append(z.copy()); a=jacobian(z,j,g,delta);matrices.append(a);m=a@m
        z=step(z,j,g,delta)
    return z,m,np.array(states),np.array(matrices)


def divisors(p):
    return [d for d in range(1,p) if p%d==0]


def period_proposals(tail, max_period=64, tol=1e-8):
    """Smallest full-state lag closing all last 4p pairs (not a proof)."""
    if len(tail)<5*max_period+1:raise ValueError("insufficient tail")
    count=tail.shape[1];scale=np.maximum(1,np.max(np.abs(tail),axis=(0,2)))
    periods=np.zeros(count,dtype=int);residual=np.full(count,np.nan)
    for p in range(1,max_period+1):
        near=np.max(np.abs(tail[-1]-tail[-1-p]),axis=1)/scale
        inds=np.flatnonzero((periods==0)&(near<=tol))
        if len(inds):
            errs=np.max(np.abs(tail[-4*p:,inds,:]-tail[-5*p:-p,inds,:]),axis=(0,2))/scale[inds]
            good=errs<=tol;periods[inds[good]]=p;residual[inds[good]]=errs[good]
    return periods,residual


def simulate_batch(j, initials, tangents, gains, deltas, *, burn=6000, align=1000,
                   measured=4096, keep=513, block=256, check=lambda:None, checkpoint=lambda t,x,v:None):
    """Independent fresh starts for every cell; one normalized tangent per orbit."""
    if measured%block or measured<keep:raise ValueError("measurement shape")
    g,d=np.meshgrid(gains,deltas,indexing="ij");grid=np.column_stack([g.ravel(),d.ravel()])
    grid=np.repeat(grid,2,axis=0);g=grid[:,0,None];d=grid[:,1,None]
    x=np.tile(initials,(len(grid)//2,1));v=np.tile(tangents,(len(grid)//2,1));v/=np.linalg.norm(v,axis=1)[:,None]
    initial_v=v.copy();collapsed=np.zeros(len(grid),dtype=int)
    tail=np.empty((keep,len(grid),len(j)));growth=np.zeros((measured//block,len(grid)))
    first_end=None
    for t in range(burn+align+measured):
        if t%128==0:
            if t%2048==0:checkpoint(t,x,v)
            try:check()
            except BaseException:
                checkpoint(t,x,v);raise
        if t>=burn:
            w=(1-d)*v+d*g*((v*(1-np.tanh(x)**2))@j.T)
            norm=np.linalg.norm(w,axis=1)
            if np.any(~np.isfinite(norm)):raise FloatingPointError("nonfinite tangent")
            zero=norm==0;collapsed+=zero
            v=w/np.where(zero,1,norm)[:,None];v[zero]=initial_v[zero]
            with np.errstate(divide="ignore"):
                if t>=burn+align:growth[(t-burn-align)//block]+=np.log(norm)
        x=step(x,j,g,d)
        if not np.all(np.isfinite(x)):raise FloatingPointError("nonfinite state")
        if t==burn+align+measured//2-1:first_end=x.copy()
        if t>=burn+align+measured-keep:tail[t-(burn+align+measured-keep)]=x
    check();periods,recurrence=period_proposals(tail)
    lam=growth.sum(axis=0)/measured;lam[collapsed>0]=np.nan
    return dict(grid=grid,tail=tail,tangent_collapse_count=collapsed,first_half_end=first_end,final=x,tangent=v,
                growth_blocks=growth,lambda_per_step=lam,
                lambda_per_time=lam/grid[:,1],
                period_proposals=periods,recurrence=recurrence)


def qualify_cycle(seed,j,g,delta,p,check=lambda:None):
    """Bounded floating correction plus both adjacent phases; never a certificate."""
    x=np.array(seed,copy=True);history=[];reason="iteration_limit"
    try:
        for k in range(7):
            check();end,m,cycle,js=iterate(x,j,g,delta,p);res=(end-x);scale=max(1,float(np.max(np.abs(cycle))))
            err=float(np.max(np.abs(res))/scale);history.append(err)
            if not np.isfinite(err):raise FloatingPointError("nonfinite closure")
            if err<=1e-11:reason="closed";break
            if k==6:break
            dx=np.linalg.solve(m-np.eye(len(x)),-res)
            if not np.all(np.isfinite(dx)) or np.max(np.abs(dx))>.05*scale:reason="correction_too_large";break
            x=x+dx
        out={"period":p,"status":reason,"history":history,"theta":x,"cycle":cycle,"phase_jacobians":js,"monodromy":m,"closure":err}
        if reason!="closed":return out
        end1,m1,_,_=iterate(cycle[1%p],j,g,delta,p)
        closure1=float(np.max(np.abs(end1-cycle[1%p]))/scale)
        eig,vec=np.linalg.eig(m);eig1,vec1=np.linalg.eig(m1)
        def eres(a,e,v):return float(np.max(np.linalg.norm(a@v-v*e[None,:],axis=0)/np.maximum(1,np.linalg.norm(a,ord=np.inf)*np.linalg.norm(v,axis=0))))
        separation={str(d):float(np.max(np.abs(cycle[d]-cycle[0]))/scale) for d in divisors(p)}
        primitive=all(v>1e-6 for v in separation.values())
        radius=float(max(np.max(np.abs(eig)),np.max(np.abs(eig1))))
        er=max(eres(m,eig,vec),eres(m1,eig1,vec1))
        symmetry=float(np.min(np.max(np.abs(cycle+cycle[0]),axis=1))/scale)
        good=primitive and radius<1-1e-6 and closure1<=1e-10 and er<=1e-10
        out.update(status="numerical_attracting_primitive" if good else "closed_unqualified",closure_adjacent=closure1,monodromy_adjacent=m1,eigenvalues=eig,eigenvectors=vec,eigenvalues_adjacent=eig1,eigenvectors_adjacent=vec1,radius=radius,eigen_residual=er,proper_divisor_separations=separation,primitive=primitive,odd_self_pair_distance=symmetry)
        return out
    except (np.linalg.LinAlgError,FloatingPointError) as e:
        return {"period":p,"status":"numerical_failure","error":str(e),"theta":x,"history":history}
